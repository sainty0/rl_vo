# env/rosbridge_client.py
import json, time
try:
    import websocket  # pip install websocket-client
except Exception:
    websocket = None

class RosbridgeClient:
    def __init__(self, url: str, lazy: bool = True, connect_timeout_s: float = 5.0):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed")
        self.url = url
        self.ws = None
        self.lazy = lazy
        self.connect_timeout_s = connect_timeout_s
        if not self.lazy:
            self._connect()

    def _connect(self):
        if self.ws is not None:
            return
        deadline = time.time() + self.connect_timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                self.ws = websocket.create_connection(self.url, timeout=3.0)
                return
            except Exception as e:
                last_err = e
                time.sleep(0.25)
        raise last_err or RuntimeError("Failed to connect to rosbridge")

    def ensure_connected(self):
        if self.ws is None:
            self._connect()

    def publish_float(self, topic: str, value: float):
        self.ensure_connected()
        msg = {"op": "publish", "topic": topic, "msg": {"data": float(value)}}
        self.ws.send(json.dumps(msg))

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass
        finally:
            self.ws = None

class MockRosbridge:
    def publish_float(self, topic: str, value: float):
        print(f"[MOCK rosbridge] publish {topic} = {value:.4f}")
