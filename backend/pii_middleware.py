"""Fail-closed user-text boundary for AG-UI, before endpoint parsing/storage.

Only user-message strings and text parts are redacted, including history.
State/context, other message roles, media URIs/metadata, and image pixels or
other media payloads are outside this text boundary and are not inspected.
"""

import asyncio
import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.social_poster import guardrails

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_TOTAL_TEXT_BYTES = 128 * 1024
MAX_MESSAGES = 256
MAX_PARTS = 512
MAX_TEXTS = 64
MAX_BODY_CHUNKS = 2048
BODY_TIMEOUT_SECONDS = 10.0
PROCESSING_TIMEOUT_SECONDS = 30.0
_MEDIA_TYPES = {"image", "audio", "video", "document"}


class _InvalidRequest(Exception):
    def __init__(self, status: int = 400):
        self.status = status


class _Disconnected(Exception):
    pass


def _invalid_constant(value: str) -> None:
    # Python's JSON decoder otherwise accepts non-JSON NaN/Infinity tokens.
    raise _InvalidRequest()


def _check_media(part: dict) -> None:
    """Validate known bridge media shapes without changing/scanning their data."""
    if part["type"] == "binary":
        if set(part) - {"type", "mimeType", "mime_type", "data", "url", "id", "filename"}:
            raise _InvalidRequest()
        for key in ("mimeType", "mime_type", "data", "url", "id", "filename"):
            if part.get(key) is not None and not isinstance(part[key], str):
                raise _InvalidRequest()
        if not (part.get("mimeType") or part.get("mime_type")) or not any(
            part.get(key) for key in ("data", "url", "id")
        ):
            raise _InvalidRequest()
        return
    if set(part) - {"type", "source", "metadata"}:
        raise _InvalidRequest()
    source = part.get("source")
    if not isinstance(source, dict) or set(source) - {"type", "value", "mimeType", "mime_type"}:
        raise _InvalidRequest()
    if source.get("type") not in ("data", "url") or not isinstance(source.get("value"), str):
        raise _InvalidRequest()
    if not source["value"]:
        raise _InvalidRequest()
    for key in ("mimeType", "mime_type"):
        if source.get(key) is not None and not isinstance(source[key], str):
            raise _InvalidRequest()
    if source["type"] == "data" and not (source.get("mimeType") or source.get("mime_type")):
        raise _InvalidRequest()


def _text_fields(data: object) -> list[tuple[dict, str]]:
    # Validate the whole protected shape before sending any text to DLP.
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise _InvalidRequest()
    if len(data["messages"]) > MAX_MESSAGES:
        raise _InvalidRequest(413)
    fields = []
    parts_count = 0
    for message in data["messages"]:
        if not isinstance(message, dict) or message.get("role") not in (
            "user", "assistant", "system", "developer", "tool", "activity"
        ):
            raise _InvalidRequest()
        if message["role"] != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            fields.append((message, "content"))
        elif isinstance(content, list):
            parts_count += len(content)
            if parts_count > MAX_PARTS:
                raise _InvalidRequest(413)
            for part in content:
                if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                    raise _InvalidRequest()
                if part["type"] == "text":
                    if set(part) != {"type", "text"} or not isinstance(part["text"], str):
                        raise _InvalidRequest()
                    fields.append((part, "text"))
                elif part["type"] in _MEDIA_TYPES or part["type"] == "binary":
                    _check_media(part)
                else:
                    raise _InvalidRequest()
        else:
            raise _InvalidRequest()
    if len(fields) > MAX_TEXTS:
        raise _InvalidRequest(413)
    total = 0
    for container, key in fields:
        size = len(container[key].encode("utf-8"))
        if size > guardrails.MAX_PII_TEXT_BYTES:
            raise _InvalidRequest(413)
        total += size
    if total > MAX_TOTAL_TEXT_BYTES:
        raise _InvalidRequest(413)
    return fields


async def _reject(send: Send, status: int) -> None:
    body = b'{"error":"Request text could not be processed."}'
    await send({"type": "http.response.start", "status": status, "headers": [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]})
    await send({"type": "http.response.body", "body": body})


class RedactUserMessages:
    """Pure ASGI; register inside CORS only when PII_REDACTION is enabled."""

    def __init__(self, app: ASGIApp, path: str = "/api/adk"):
        self.app, self.path = app, path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            not guardrails.PII_REDACTION or scope["type"] != "http"
            or scope["method"] != "POST" or scope["path"] != self.path
        ):
            await self.app(scope, receive, send)
            return
        try:
            for name, value in scope.get("headers", []):
                if name.lower() == b"content-encoding" and value.strip().lower() != b"identity":
                    raise _InvalidRequest(415)
                if name.lower() == b"content-type":
                    media_type = value.split(b";", 1)[0].strip().lower()
                    if media_type != b"application/json" and not (
                        media_type.startswith(b"application/") and media_type.endswith(b"+json")
                    ):
                        raise _InvalidRequest(415)
                if name.lower() == b"content-length":
                    if not value.isdigit():
                        raise _InvalidRequest()
                    if len(value) > 10 or int(value) > MAX_BODY_BYTES:
                        raise _InvalidRequest(413)
            body = bytearray()
            async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
                for _ in range(MAX_BODY_CHUNKS):
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        raise _Disconnected()
                    if message["type"] != "http.request" or not isinstance(message.get("body", b""), bytes):
                        raise _InvalidRequest()
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        raise _InvalidRequest(413)
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
                else:
                    raise _InvalidRequest(413)
        except TimeoutError:
            await _reject(send, 408)
            return
        except (_Disconnected, OSError):
            return
        except _InvalidRequest as error:
            await _reject(send, error.status)
            return

        try:
            async with asyncio.timeout(PROCESSING_TIMEOUT_SECONDS):
                try:
                    data = json.loads(body, parse_constant=_invalid_constant)
                    fields = _text_fields(data)
                except (ValueError, UnicodeError, RecursionError):
                    raise _InvalidRequest() from None
                for container, key in fields:
                    container[key] = await guardrails.redact_boundary(container[key])
                rewritten = json.dumps(data, allow_nan=False).encode("utf-8")
                if len(rewritten) > MAX_BODY_BYTES:
                    raise _InvalidRequest(413)
        except _InvalidRequest as error:
            await _reject(send, error.status)
            return
        except Exception:
            # Includes DLP/SDK/configuration failures and the processing deadline.
            # No exception details, submitted text, or partially cleaned body escape.
            await _reject(send, 503)
            return

        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": rewritten, "more_body": False}
            # SSE listens here for the real disconnect. Never fabricate one or
            # repeatedly return an empty/request frame (which can busy-loop).
            return await receive()

        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(rewritten)).encode("ascii")))
        await self.app({**scope, "headers": headers}, replay, send)
