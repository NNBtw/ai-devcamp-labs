"""Await bounded Memory Bank submission; never publish or infer searchability."""

from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import re

import httpx
import vertexai
from google.auth.transport import mtls
from google.genai.types import HttpOptions, HttpRetryOptions

from .memory_ingestion_retry import submit_with_retry


_LOG = logging.getLogger(__name__)
_RESOURCE_PATTERN = re.compile(
    r"(?:^|/)projects/(?P<project>\d+)/"
    r"locations/(?P<location>[a-z0-9-]+)/"
    r"reasoningEngines/(?P<engine>\d+)(?:/|$)"
)
_PART_FIELDS = (
    "text", "inline_data", "file_data", "function_call",
    "function_response", "executable_code", "code_execution_result",
    "tool_call", "tool_response",
)


@dataclass(frozen=True)
class PreparedIngestion:
    project: str
    location: str
    name: str
    app_name: str
    user_id: str
    stream_id: str
    events_json: str
    event_count: int

    def request_kwargs(self):
        # Fresh models/dicts prevent a request from mutating the next retry.
        return {
            "name": self.name,
            "scope": {"app_name": self.app_name, "user_id": self.user_id},
            "stream_id": self.stream_id,
            "direct_contents_source": (
                vertexai.types.IngestionDirectContentsSource.model_validate_json(
                    self.events_json
                )
            ),
            "config": {"wait_for_completion": False},
        }


@dataclass(frozen=True)
class IngestionReceipt:
    status: str
    api_accepted: bool
    operation_name: str | None = None
    operation_completed: bool | None = None
    operation_succeeded: bool | None = None
    operation_error_code: int | None = None
    memory_searchable: bool | None = None


def parse_runtime_resource(value):
    if not isinstance(value, str):
        raise ValueError("Runtime resource must be a string")
    match = _RESOURCE_PATTERN.search(value.strip())
    if match is None:
        raise ValueError("Invalid Agent Runtime resource")
    project, location, engine = (
        match["project"], match["location"], match["engine"]
    )
    name = (
        f"projects/{project}/locations/{location}/reasoningEngines/{engine}"
    )
    return project, location, name


def prepare_session_ingestion(session, runtime_resource):
    project, location, name = parse_runtime_resource(runtime_resource)
    identity = {
        "app_name": session.app_name,
        "user_id": session.user_id,
        "session_id": session.id,
    }
    if any(
        not isinstance(value, str) or not value.strip()
        for value in identity.values()
    ):
        raise ValueError("Stream identity must contain non-empty strings")
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    stream_id = hashlib.sha256(encoded).hexdigest()[:32]

    direct_events = []
    for event in tuple(session.events):
        content = event.content
        if not content or not content.parts:
            continue
        if not any(
            getattr(part, field, None)
            for part in content.parts for field in _PART_FIELDS
        ):
            continue
        if not isinstance(event.id, str) or not event.id.strip():
            raise ValueError("A retained event must have a stable event ID")
        event_time = None
        if event.timestamp is not None:
            event_time = datetime.datetime.fromtimestamp(
                event.timestamp, tz=datetime.timezone.utc
            )
        direct_events.append(vertexai.types.IngestionDirectContentsSourceEvent(
            content=content, event_id=event.id, event_time=event_time
        ))

    payload = vertexai.types.IngestionDirectContentsSource(events=direct_events)
    return PreparedIngestion(
        project=project, location=location, name=name,
        app_name=session.app_name, user_id=session.user_id,
        stream_id=stream_id,
        events_json=payload.model_dump_json(exclude_none=True),
        event_count=len(direct_events),
    )


def classify_operation(operation):
    name = operation.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Missing operation name; submission outcome is uncertain")
    completed = operation.done is True
    error = operation.error
    if error is not None and not completed:
        raise ValueError("Inconsistent operation: error before done=True")
    error_code = error.get("code") if error is not None else None
    if type(error_code) is not int:
        error_code = None
    succeeded = (error is None) if completed else None
    status = "api_accepted"
    if completed:
        status = "operation_completed" if succeeded else "operation_failed"
    return IngestionReceipt(
        status=status, api_accepted=True, operation_name=name,
        operation_completed=completed, operation_succeeded=succeeded,
        operation_error_code=error_code,
    )


async def _close_clients(*clients):
    # A cleanup error must not turn an accepted POST into a submission retry,
    # or replace the original exception. Still propagate cancellation.
    cancelled = None
    for client in clients:
        if client is None:
            continue
        try:
            await client.aclose()
        except asyncio.CancelledError as error:
            cancelled = error
        except Exception:
            _LOG.exception("Memory ingestion client cleanup failed")
    if cancelled is not None:
        raise cancelled


async def ingest_session_with_retry(
    session, *, runtime_resource, attempt_timeout=30.0,
    sleep=asyncio.sleep, credentials=None, transport=None,
):
    """At most three submissions per call; do not poll or retry failed LROs.

    credentials/transport/sleep allow offline tests. Production defaults use
    the SDK's certificate-bound native session when requested; otherwise
    an HTTPX transport with connect retries disabled.
    """
    if not math.isfinite(attempt_timeout) or attempt_timeout <= 0:
        raise ValueError("attempt_timeout must be finite and positive")
    prepared = prepare_session_ingestion(session, runtime_resource)
    if prepared.event_count == 0:
        return IngestionReceipt(status="skipped_no_events", api_accepted=False)

    if hasattr(mtls, "should_use_client_cert") and mtls.should_use_client_cert():
        if transport is not None:
            raise ValueError("Custom HTTPX cannot bypass required client-certificate authentication")
        from .memory_ingestion_mtls import ingest_prepared_with_mtls

        return await ingest_prepared_with_mtls(
            prepared, attempt_timeout=attempt_timeout, sleep=sleep, credentials=credentials,
        )

    if transport is None:
        transport = httpx.AsyncHTTPTransport(retries=0)
    http_client = httpx.AsyncClient(transport=transport)
    client = None
    try:
        client = vertexai.Client(
            project=prepared.project,
            location=prepared.location,
            credentials=credentials,
            http_options=HttpOptions(
                httpx_async_client=http_client,
                retry_options=HttpRetryOptions(attempts=1),
                timeout=max(1, int(attempt_timeout * 1000)),
            ),
        ).aio
        async def submit():
            return await client.agent_engines.memories.ingest_events(
                **prepared.request_kwargs()
            )

        operation = await submit_with_retry(
            submit, attempt_timeout=attempt_timeout, sleep=sleep
        )
        # Classification is outside the POST retry boundary.
        return classify_operation(operation)
    finally:
        await _close_clients(client, http_client)
