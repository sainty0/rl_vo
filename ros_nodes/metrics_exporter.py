#!/usr/bin/env python3
"""
DEPRECATED STUB

Training uses the ROS package exporter at:
  metric_pkg/scripts/metrics_exporter.py

This file is intentionally retained only to avoid confusion and accidental edits.
Do not modify or use this file. If executed or imported, it will raise an error.
"""

import sys

def main():
    sys.stderr.write(
        "Deprecated: use metric_pkg/scripts/metrics_exporter.py (launched by rl_vo/scripts/launch_sclsam.launch)\\n"
    )
    raise RuntimeError(
        "Deprecated metrics_exporter: use metric_pkg/scripts/metrics_exporter.py"
    )

if __name__ == "__main__":
    main()
