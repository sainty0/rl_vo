from env.rosbridge_client import MockRosbridge

def test_mock_publish():
    rb = MockRosbridge()
    rb.publish_float("/foo", 0.1234)  # should not raise
