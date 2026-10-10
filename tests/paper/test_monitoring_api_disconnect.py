from __future__ import annotations

from src.paper.monitoring_api import create_monitoring_server


class _ClosedClientSocket:
    def write(self, _data: bytes) -> None:
        raise BrokenPipeError("client disconnected during response")


def test_monitoring_api_ignores_client_disconnect_while_sending_response():
    server = create_monitoring_server("results/paper", token="test-monitoring-token-123456", port=0)
    try:
        handler_type = server.RequestHandlerClass
        handler = handler_type.__new__(handler_type)
        handler.wfile = _ClosedClientSocket()
        handler._headers_buffer = []
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"
        handler.requestline = "GET /v1/dashboard HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.server = server

        # A dropped browser/mobile connection is routine and must not escape
        # into ThreadingHTTPServer's request-thread traceback path.
        handler._send(200, {"state": "RUNNING"})
    finally:
        server.server_close()
