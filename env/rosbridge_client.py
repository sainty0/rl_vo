import json
try:
    import websocket  # pip install websocket-client
except Exception:
    websocket = None

class RosbridgeClient:
    def __init__(self, url: str):
        if websocket is None:
            raise RuntimeError("websocket-client is not installed")
        self.ws = websocket.create_connection(url, timeout=3.0)

    def publish_float(self, topic: str, value: float):
        msg = {"op": "publish", "topic": topic, "msg": {"data": float(value)}}
        self.ws.send(json.dumps(msg))

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


class MockRosbridge:
    def publish_float(self, topic: str, value: float):
        print(f"[MOCK rosbridge] publish {topic} = {value:.4f}")
