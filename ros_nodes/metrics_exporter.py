#!/usr/bin/env python3
# Minimal ROS1 metrics exporter for probe features.
# Computes lightweight stats over a window and writes JSON to /tmp/rlvo/probe.json.
# Exposes:
#   /rl_metrics/reset  (std_srvs/Trigger)  -> clears accumulators
#   /rl_metrics/commit (std_srvs/Trigger)  -> writes JSON file and returns success
#
# Params (ROS ~private params override CLI):
#   ~out_file: path to write JSON (default: /tmp/rlvo/probe.json)
#   ~dropout_thresh: seconds; odom Δt above this count toward pose_dropouts_s
#   ~scan_topic: PointCloud2 topic for scans (default: /lio_sam/deskew/cloud_deskewed)
#   ~surf_topic: PointCloud2 topic for surface features
#   ~corner_topic: PointCloud2 topic for corner features
#   ~odom_topic: nav_msgs/Odometry topic
#   ~imu_topic: sensor_msgs/Imu topic
#
# NOTE: No heavy point decoding — uses PointCloud2.width*height only.

#!/usr/bin/env python3
# ... (header comments unchanged)

import os
import json
import math
from dataclasses import dataclass
from collections import deque
import bisect

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry


@dataclass
class Welford:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    def add(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def value(self):
        if self.n < 2:
            return self.mean, 0.0
        var = self.M2 / (self.n - 1)
        return self.mean, math.sqrt(max(var, 0.0))


class MetricsExporter:
    def __init__(self):
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", default="/tmp/rlvo/probe.json")
        ap.add_argument("--dropout_thresh", type=float, default=0.3)
        ap.add_argument("--scan_topic", default="/lio_sam/deskew/cloud_deskewed")
        ap.add_argument("--surf_topic", default="/lio_sam/feature/cloud_surface")
        ap.add_argument("--corner_topic", default="/lio_sam/feature/cloud_corner")
        ap.add_argument("--odom_topic", default="/lio_sam/mapping/odometry_incremental")
        ap.add_argument("--imu_topic", default="/imu/data_raw")
        # NEW: variable-token controls
        ap.add_argument("--max_tokens", type=int, default=64)
        ap.add_argument("--match_tol_s", type=float, default=0.05)  # time match tolerance for tokens
        args, _ = ap.parse_known_args()

        rospy.init_node("rl_metrics_exporter", anonymous=True)

        self.out_file = rospy.get_param("~out_file", args.out)
        self.dropout_thresh = float(rospy.get_param("~dropout_thresh", args.dropout_thresh))
        self.scan_topic = rospy.get_param("~scan_topic", args.scan_topic)
        self.surf_topic = rospy.get_param("~surf_topic", args.surf_topic)
        self.corner_topic = rospy.get_param("~corner_topic", args.corner_topic)
        self.odom_topic = rospy.get_param("~odom_topic", args.odom_topic)
        self.imu_topic = rospy.get_param("~imu_topic", args.imu_topic)
        self.max_tokens = int(rospy.get_param("~max_tokens", args.max_tokens))
        self.match_tol_s = float(rospy.get_param("~match_tol_s", args.match_tol_s))

        # Accumulators
        self.reset_accumulators()

        # Subs
        self.sub_scan = rospy.Subscriber(self.scan_topic, PointCloud2, self.cb_scan, queue_size=20, tcp_nodelay=True)
        self.sub_surf = rospy.Subscriber(self.surf_topic, PointCloud2, self.cb_surf, queue_size=20, tcp_nodelay=True)
        self.sub_corner = rospy.Subscriber(self.corner_topic, PointCloud2, self.cb_corner, queue_size=20, tcp_nodelay=True)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=100, tcp_nodelay=True)
        self.sub_imu = rospy.Subscriber(self.imu_topic, Imu, self.cb_imu, queue_size=200, tcp_nodelay=True)

        # Services
        self.srv_reset = rospy.Service("/rl_metrics/reset", Trigger, self.on_reset)
        self.srv_commit = rospy.Service("/rl_metrics/commit", Trigger, self.on_commit)

        rospy.loginfo("rl_metrics_exporter up. Writing to %s", self.out_file)

    # ---------- Core ----------
    def reset_accumulators(self):
        self.t0 = None
        self.t_end = None

        # fixed stats
        self.scan_pts = Welford(); self.scan_times = []
        self.surf_pts = Welford()
        self.corner_pts = Welford()
        self.odom_times = []
        self.vel_norm = Welford()
        self.acc_jolt_sum = 0.0; self.acc_jolt_n = 0
        self.pose_dropouts_s = 0.0
        self._last_odom = None  # (t, x, y, z, v)
        self.imu_ang_sq_sum = 0.0; self.imu_lin_sq_sum = 0.0; self.imu_n = 0

        # NEW: time-series buffers for variable tokens (bounded deques for memory safety)
        self._scan_rec = deque(maxlen=5000)    # (t, pts_total)
        self._surf_rec = deque(maxlen=5000)    # (t, pts_surf)
        self._corner_rec = deque(maxlen=5000)  # (t, pts_corner)
        self._vel_rec = deque(maxlen=20000)    # (t, v_norm)

    def _update_time_bounds(self, t):
        if self.t0 is None: self.t0 = t
        self.t_end = t

    @staticmethod
    def _pts_count(pc2: PointCloud2) -> int:
        w = pc2.width; h = pc2.height if pc2.height > 0 else 1
        return int(w * h)

    # ---------- Callbacks ----------
    def cb_scan(self, msg: PointCloud2):
        t = msg.header.stamp.to_sec(); self._update_time_bounds(t)
        n = self._pts_count(msg)
        self.scan_times.append(t)
        self.scan_pts.add(n)
        self._scan_rec.append((t, float(n)))

    def cb_surf(self, msg: PointCloud2):
        t = msg.header.stamp.to_sec(); self._update_time_bounds(t)
        n = self._pts_count(msg)
        self.surf_pts.add(n)
        self._surf_rec.append((t, float(n)))

    def cb_corner(self, msg: PointCloud2):
        t = msg.header.stamp.to_sec(); self._update_time_bounds(t)
        n = self._pts_count(msg)
        self.corner_pts.add(n)
        self._corner_rec.append((t, float(n)))

    def cb_imu(self, msg: Imu):
        t = msg.header.stamp.to_sec(); self._update_time_bounds(t)
        wx, wy, wz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        self.imu_ang_sq_sum += (wx*wx + wy*wy + wz*wz)
        self.imu_lin_sq_sum += (ax*ax + ay*ay + az*az)
        self.imu_n += 1

    def cb_odom(self, msg: Odometry):
        t = msg.header.stamp.to_sec(); self._update_time_bounds(t)
        self.odom_times.append(t)
        p = msg.pose.pose.position
        if self._last_odom is not None:
            t_prev, x_prev, y_prev, z_prev, v_prev = self._last_odom
            dt = max(t - t_prev, 1e-6)
            vx = (p.x - x_prev)/dt; vy = (p.y - y_prev)/dt; vz = (p.z - z_prev)/dt
            v = math.sqrt(vx*vx + vy*vy + vz*vz)
            self.vel_norm.add(v)
            self._vel_rec.append((t, float(v)))
            dv = abs(v - v_prev); self.acc_jolt_sum += dv/dt; self.acc_jolt_n += 1
            if dt > self.dropout_thresh: self.pose_dropouts_s += dt
            self._last_odom = (t, p.x, p.y, p.z, v)
        else:
            self._last_odom = (t, p.x, p.y, p.z, 0.0)
            self._vel_rec.append((t, 0.0))

    # ---------- Services ----------
    def on_reset(self, _req):
        self.reset_accumulators()
        return TriggerResponse(success=True, message="reset")

    def on_commit(self, _req):
        d = self._make_dict()
        try:
            odir = os.path.dirname(self.out_file)
            if odir and not os.path.exists(odir): os.makedirs(odir)
            with open(self.out_file, "w") as f: json.dump(d, f, indent=2)
            rospy.loginfo("wrote %s", self.out_file)
            return TriggerResponse(success=True, message=self.out_file)
        except Exception as e:
            rospy.logerr("write failed: %s", e)
            return TriggerResponse(success=False, message=str(e))

    # ---------- Helpers ----------
    def _rate_from_times(self, ts):
        if len(ts) < 2: return 0.0
        duration = max(ts[-1] - ts[0], 1e-6)
        return float((len(ts) - 1) / duration)

    def _nearest_value(self, rec_deque, t, tol):
        """rec_deque: deque[(t,val)] sorted by time; return val nearest to t within tol, else 0.0."""
        if not rec_deque: return 0.0
        times = [x[0] for x in rec_deque]
        i = bisect.bisect_left(times, t)
        cand = []
        if i < len(times): cand.append((abs(times[i]-t), rec_deque[i][1]))
        if i > 0: cand.append((abs(times[i-1]-t), rec_deque[i-1][1]))
        if not cand: return 0.0
        best = min(cand, key=lambda x: x[0])
        return best[1] if best[0] <= tol else 0.0

    def _build_variable_tokens(self):
        """
        Build tokens aligned on scan timestamps.
        Each token: [surf_pts_at_t, corner_pts_at_t, vel_norm_at_t].
        Change here if you want different token channels.
        """
        tokens = []
        if len(self._scan_rec) == 0: return tokens
        # Use all scan stamps, then downsample if > max_tokens
        scan_ts = [t for (t, _) in self._scan_rec]
        stride = max(1, int(math.ceil(len(scan_ts) / float(self.max_tokens))))
        for idx in range(0, len(scan_ts), stride):
            t = scan_ts[idx]
            tok = [
                float(self._nearest_value(self._surf_rec, t, self.match_tol_s)),
                float(self._nearest_value(self._corner_rec, t, self.match_tol_s)),
                float(self._nearest_value(self._vel_rec, t, self.match_tol_s)),
            ]
            tokens.append(tok)
            if len(tokens) >= self.max_tokens:
                break
        return tokens

    def _make_dict(self):
        # fixed stats
        scan_mean, scan_std = self.scan_pts.value()
        surf_mean, surf_std = self.surf_pts.value()
        corner_mean, corner_std = self.corner_pts.value()
        vel_mean, vel_std = self.vel_norm.value()
        odom_rate = self._rate_from_times(self.odom_times)
        scan_rate = self._rate_from_times(self.scan_times)
        acc_jolt_mean = (self.acc_jolt_sum / self.acc_jolt_n) if self.acc_jolt_n > 0 else 0.0
        imu_ang_rms = math.sqrt(self.imu_ang_sq_sum / self.imu_n) if self.imu_n > 0 else 0.0
        imu_lin_rms = math.sqrt(self.imu_lin_sq_sum / self.imu_n) if self.imu_n > 0 else 0.0

        d = {
            "pts_per_scan_mean": float(scan_mean),
            "pts_per_scan_std": float(scan_std),
            "scan_rate_hz": float(scan_rate),
            "surf_pts_mean": float(surf_mean),
            "surf_pts_std": float(surf_std),
            "corner_pts_mean": float(corner_mean),
            "corner_pts_std": float(corner_std),
            "odom_rate_hz": float(odom_rate),
            "pose_dropouts_s": float(self.pose_dropouts_s),
            "vel_norm_mean": float(vel_mean),
            "vel_norm_std": float(vel_std),
            "acc_jolt_mean": float(acc_jolt_mean),
            "imu_ang_vel_rms": float(imu_ang_rms),
            "imu_lin_acc_rms": float(imu_lin_rms),
            "planarity_ratio_mean": 0.0,
            "action_last": 0.0,
        }

        # NEW: variable tokens + metadata
        tokens = self._build_variable_tokens()
        d["variable_tokens"] = tokens                 # list[list[float]]
        d["variable_tokens_n"] = int(len(tokens))     # pre-pad count
        d["token_feature_names"] = ["surf_pts", "corner_pts", "vel_norm"]

        # NEW: critique tail (4 dims)
        d["critique_tail"] = [
            float(odom_rate),
            float(self.pose_dropouts_s),
            float(scan_rate),
            float(scan_mean),
        ]
        return d


def main():
    MetricsExporter()
    rospy.spin()


if __name__ == "__main__":
    main()
