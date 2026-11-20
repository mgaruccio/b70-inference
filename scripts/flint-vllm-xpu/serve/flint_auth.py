"""ASGI middleware for the Flint vLLM server: require the API key on every route.

vLLM's --api-key only guards /v1, /v2, /inference and /cohere; other routes such as
/invocations run inference unauthenticated (docs.vllm.ai/en/latest/usage/security.html).
This keeps llama.cpp's posture: everything needs `Authorization: Bearer <key>` except
/health, and /metrics is also open to loopback (B70 LCD collector on the guest).
Fails closed when no key is configured. Key source: FLINT_API_KEY or VLLM_API_KEY."""

import hmac
import json
import os

OPEN_PATHS = {"/health"}
LOOPBACK_PATHS = {"/metrics"}
LOOPBACK = {"127.0.0.1", "::1"}


class FlintAuth:
    def __init__(self, app):
        self.app = app
        key = os.environ.get("FLINT_API_KEY") or os.environ.get("VLLM_API_KEY") or ""
        self.expected = f"Bearer {key}".encode() if key else None

    def _allowed(self, scope) -> bool:
        path = scope.get("path", "")
        if path in OPEN_PATHS:
            return True
        client = scope.get("client") or ("", 0)
        if path in LOOPBACK_PATHS and client[0] in LOOPBACK:
            return True
        if self.expected is None:
            return False
        auth = b""
        for name, value in scope.get("headers", ()):
            if name == b"authorization":
                auth = value
                break
        return hmac.compare_digest(auth, self.expected)

    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind not in ("http", "websocket") or self._allowed(scope):
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": {"message": "Unauthorized", "type": "authentication_error",
                                     "code": 401}}).encode()
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})
