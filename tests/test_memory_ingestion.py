"""Offline-only tests: no real Agent import, credentials, or network."""

import ast
import asyncio
from contextlib import suppress
import datetime
import importlib
import json
from pathlib import Path
import socket
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import google.auth
from google.auth.credentials import Credentials
from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import errors, types
import httpx
import pytest
import vertexai


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = (
    "projects/123456789012/locations/us-central1/"
    "reasoningEngines/1234567890"
)
BINARY = b"\x00\xff\x10\x80offline-test"


@pytest.fixture(autouse=True)
def forbid_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real network or default credentials are forbidden")
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(google.auth, "default", forbidden)


@pytest.fixture(scope="module")
def ingestion():
    # Namespace package avoids backend/social_poster/__init__.py and Agent/DB.
    name = "_lab2_offline_memory_tests"
    assert name not in sys.modules
    package = ModuleType(name)
    package.__path__ = [str(ROOT / "backend/social_poster")]
    sys.modules[name] = package
    try:
        yield importlib.import_module(name + ".memory_ingestion")
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def session():
    return Session(
        id="offline-session", app_name="social_poster", user_id="offline-user",
        state={}, events=[
            Event(id="text-1", author="user", timestamp=1700000000.0,
                  content=types.Content(role="user", parts=[
                      types.Part(text="offline preference")
                  ])),
            Event(id="tool-1", author="agent", timestamp=1700000001.0,
                  content=types.Content(role="model", parts=[
                      types.Part(function_call=types.FunctionCall(
                          name="offline_tool", args={"value": "demo"}
                      ))
                  ])),
            Event(id="image-1", author="user", timestamp=1700000002.0,
                  content=types.Content(role="user", parts=[
                      types.Part(inline_data=types.Blob(
                          mime_type="image/png", data=BINARY
                      ))
                  ])),
            Event(id="empty-1", author="user", timestamp=1700000003.0,
                  content=types.Content(role="user", parts=[])),
        ],
    )


class OfflineCredentials(Credentials):
    def __init__(self):
        super().__init__()
        self.token = "offline-test-token"

    def refresh(self, request):
        raise AssertionError("Credential refresh is forbidden")


class TrackingTransport(httpx.MockTransport):
    def __init__(self, handler, *, fail_close=False):
        super().__init__(handler)
        self.closed = False
        self.fail_close = fail_close

    async def aclose(self):
        self.closed = True
        await super().aclose()
        if self.fail_close:
            raise RuntimeError("offline close failure")


@pytest.mark.parametrize("codes,count,error", [
    ([200], 1, None), ([429, 503, 200], 3, None), ([408, 200], 2, None),
    ([400], 1, 400), ([401], 1, 401), ([403], 1, 403),
    ([503, 503, 503], 3, 503),
])
def test_core_http_budget(ingestion, codes, count, error):
    async def run():
        calls, delays = [], []
        sequence = iter(codes)
        async def submit():
            code = next(sequence)
            calls.append(code)
            if code == 200:
                return "accepted"
            cls = errors.ClientError if code < 500 else errors.ServerError
            raise cls(code, {"error": {"message": "offline"}})
        async def sleep(delay):
            delays.append(delay)
        if error is None:
            assert await ingestion.submit_with_retry(submit, sleep=sleep) == "accepted"
        else:
            with pytest.raises(errors.APIError) as caught:
                await ingestion.submit_with_retry(submit, sleep=sleep)
            assert caught.value.code == error
        assert len(calls) == count
        assert len(delays) == count - 1
        assert all(0 <= value <= 2 ** index for index, value in enumerate(delays))
    asyncio.run(run())


@pytest.mark.parametrize("cls,retryable", [
    (httpx.ConnectError, True), (httpx.ReadError, True), (httpx.WriteError, True),
    (httpx.ConnectTimeout, True), (httpx.ReadTimeout, True),
    (httpx.WriteTimeout, True), (httpx.PoolTimeout, True),
    (httpx.InvalidURL, False), (httpx.UnsupportedProtocol, False),
    (httpx.LocalProtocolError, False),
])
def test_core_network_policy(ingestion, cls, retryable):
    async def run():
        calls, delays = [], []
        async def submit():
            calls.append(1)
            if retryable and len(calls) == 3:
                return "accepted"
            raise cls("offline")
        async def sleep(delay):
            delays.append(delay)
        if retryable:
            assert await ingestion.submit_with_retry(submit, sleep=sleep) == "accepted"
        else:
            with pytest.raises(cls):
                await ingestion.submit_with_retry(submit, sleep=sleep)
        assert len(calls) == (3 if retryable else 1)
        assert len(delays) == (2 if retryable else 0)
    asyncio.run(run())


def test_core_deadline_cleanup(ingestion):
    async def run():
        calls, cleanups, delays = [], [], []
        blocker = asyncio.Event()
        async def submit():
            calls.append(1)
            try:
                await blocker.wait()
            finally:
                cleanups.append(1)
        async def sleep(delay):
            delays.append(delay)
        with pytest.raises(TimeoutError):
            await ingestion.submit_with_retry(
                submit, attempt_timeout=0.01, sleep=sleep
            )
        assert len(calls) == len(cleanups) == 3
        assert len(delays) == 2
    asyncio.run(run())


def test_core_unrelated_timeout_is_not_retried(ingestion):
    async def run():
        calls = []
        async def submit():
            calls.append(1)
            raise TimeoutError("inner timeout")
        async def sleep(delay):
            pytest.fail("Unexpected retry")
        with pytest.raises(TimeoutError, match="inner timeout"):
            await ingestion.submit_with_retry(submit, sleep=sleep)
        assert len(calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize("during_backoff", [False, True])
def test_core_cancellation(ingestion, during_backoff):
    async def run():
        entered, blocker, calls = asyncio.Event(), asyncio.Event(), []
        async def submit():
            calls.append(1)
            if during_backoff:
                raise errors.ClientError(429, {"error": {"message": "offline"}})
            entered.set()
            await blocker.wait()
        async def sleep(delay):
            entered.set()
            await blocker.wait()
        task = asyncio.create_task(ingestion.submit_with_retry(submit, sleep=sleep))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(calls) == 1
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    asyncio.run(run())


@pytest.mark.parametrize("value", [
    RESOURCE, "  " + RESOURCE + "  ",
    "https://offline.invalid/v1beta1/" + RESOURCE, RESOURCE + "/",
])
def test_runtime_resource(ingestion, value):
    assert ingestion.parse_runtime_resource(value) == (
        "123456789012", "us-central1", RESOURCE
    )


@pytest.mark.parametrize("value", [
    "", None, "projects/offline/locations/us-central1/reasoningEngines/123",
    "projects/123/locations/us-central1/reasoningEngines/",
])
def test_invalid_runtime(ingestion, value):
    with pytest.raises(ValueError):
        ingestion.parse_runtime_resource(value)


@pytest.mark.parametrize("field", ["id", "app_name", "user_id"])
def test_stream_identity_isolation(ingestion, field):
    original = session()
    first = ingestion.prepare_session_ingestion(original, RESOURCE)
    repeated = ingestion.prepare_session_ingestion(original, RESOURCE)
    changed = original.model_copy(update={field: getattr(original, field) + "-other"})
    assert first.stream_id == repeated.stream_id
    assert first.stream_id != ingestion.prepare_session_ingestion(changed, RESOURCE).stream_id


def test_snapshot_preserves_content_and_resists_mutation(ingestion):
    original = session()
    prepared = ingestion.prepare_session_ingestion(original, RESOURCE)
    first = prepared.request_kwargs()
    events = first["direct_contents_source"].events
    assert [event.event_id for event in events] == ["text-1", "tool-1", "image-1"]
    assert events[0].event_time.utcoffset() == datetime.timedelta(0)
    assert events[0].event_time.timestamp() == 1700000000.0
    assert events[1].content.parts[0].function_call.name == "offline_tool"
    assert events[2].content.parts[0].inline_data.data == BINARY
    events[0].event_id = "tampered"
    first["scope"]["user_id"] = "tampered"
    original.events[0].content.parts[0].text = "tampered"
    second = prepared.request_kwargs()
    assert second["scope"] == {"app_name": "social_poster", "user_id": "offline-user"}
    assert second["direct_contents_source"].events[0].event_id == "text-1"
    assert second["direct_contents_source"].events[0].content.parts[0].text == "offline preference"
    assert second["direct_contents_source"].events[2].content.parts[0].inline_data.data == BINARY


def test_missing_event_id_is_fail_closed(ingestion):
    original = session()
    original.events[0] = original.events[0].model_copy(update={"id": ""})
    with pytest.raises(ValueError, match="stable event ID"):
        ingestion.prepare_session_ingestion(original, RESOURCE)


def test_no_events_never_constructs_client(ingestion, monkeypatch):
    original = session()
    original.events = []
    def forbidden(*args, **kwargs):
        pytest.fail("Empty events must not create a client")
    monkeypatch.setattr(ingestion.vertexai, "Client", forbidden)
    receipt = asyncio.run(ingestion.ingest_session_with_retry(
        original, runtime_resource=RESOURCE
    ))
    assert receipt.status == "skipped_no_events"
    assert receipt.api_accepted is False
    assert receipt.memory_searchable is None


@pytest.mark.parametrize("done,error,status,succeeded", [
    (False, None, "api_accepted", None), (None, None, "api_accepted", None),
    (True, None, "operation_completed", True),
    (True, {"code": 13}, "operation_failed", False),
    (True, {}, "operation_failed", False),
])
def test_operation_states(ingestion, done, error, status, succeeded):
    operation = vertexai.types.MemoryBankIngestEventsOperation(
        name=RESOURCE + "/operations/offline", done=done, error=error
    )
    result = ingestion.classify_operation(operation)
    assert result.status == status
    assert result.api_accepted is True
    assert result.operation_completed is (done is True)
    assert result.operation_succeeded is succeeded
    assert result.memory_searchable is None


@pytest.mark.parametrize("fields", [
    {"done": False},
    {"name": "operations/offline", "done": False, "error": {"code": 13}},
])
def test_invalid_operation_response(ingestion, fields):
    with pytest.raises(ValueError):
        ingestion.classify_operation(
            vertexai.types.MemoryBankIngestEventsOperation(**fields)
        )


@pytest.mark.parametrize("codes,error,failed_lro", [
    ([429, 503, 200], None, False), ([503, 503, 503], 503, False),
    ([403], 403, False), ([200], None, True),
])
def test_sdk_wire_budget_and_cleanup(ingestion, codes, error, failed_lro):
    async def run():
        sequence, bodies, delays = iter(codes), [], []
        def handler(request):
            assert request.method == "POST"
            assert request.url.path.endswith("/memories:ingestEvents")
            bodies.append(request.content)
            code = next(sequence)
            if code != 200:
                return httpx.Response(code, json={"error": {"code": code, "message": "offline"}})
            body = {"name": RESOURCE + "/operations/offline", "done": failed_lro}
            if failed_lro:
                body["error"] = {"code": 13, "message": "offline operation failure"}
            return httpx.Response(200, json=body)
        async def sleep(delay):
            delays.append(delay)
        transport = TrackingTransport(handler)
        kwargs = dict(runtime_resource=RESOURCE, credentials=OfflineCredentials(),
                      transport=transport, sleep=sleep)
        if error is None:
            result = await ingestion.ingest_session_with_retry(session(), **kwargs)
            assert result.status == ("operation_failed" if failed_lro else "api_accepted")
            assert result.memory_searchable is None
        else:
            with pytest.raises(errors.APIError) as caught:
                await ingestion.ingest_session_with_retry(session(), **kwargs)
            assert caught.value.code == error
        assert len(bodies) == len(codes)
        assert len(delays) == len(codes) - 1
        assert all(body == bodies[0] for body in bodies)
        assert json.loads(bodies[0])["scope"] == {
            "app_name": "social_poster", "user_id": "offline-user"
        }
        assert transport.closed
    asyncio.run(run())


@pytest.mark.parametrize("close_fails", [False, True])
def test_sdk_cancellation_closes_transport(ingestion, close_fails):
    async def run():
        entered, blocker, calls = asyncio.Event(), asyncio.Event(), []
        async def handler(request):
            calls.append(1)
            entered.set()
            await blocker.wait()
        transport = TrackingTransport(handler, fail_close=close_fails)
        task = asyncio.create_task(ingestion.ingest_session_with_retry(
            session(), runtime_resource=RESOURCE,
            credentials=OfflineCredentials(), transport=transport,
        ))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(calls) == 1
            assert transport.closed
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    asyncio.run(run())


def test_cleanup_failure_preserves_accepted_receipt(ingestion, caplog):
    async def run():
        calls = []
        def handler(request):
            calls.append(1)
            return httpx.Response(200, json={
                "name": RESOURCE + "/operations/offline", "done": False,
            })
        transport = TrackingTransport(handler, fail_close=True)
        receipt = await ingestion.ingest_session_with_retry(
            session(), runtime_resource=RESOURCE,
            credentials=OfflineCredentials(), transport=transport,
        )
        assert receipt.api_accepted is True
        assert receipt.status == "api_accepted"
        assert len(calls) == 1
        assert transport.closed
        assert any("cleanup failed" in record.getMessage() for record in caplog.records)
    asyncio.run(run())


def test_cleanup_cancellation_still_closes_next_client(ingestion):
    async def run():
        closed = []
        class CancelledClose:
            async def aclose(self):
                raise asyncio.CancelledError()
        class NextClose:
            async def aclose(self):
                closed.append(1)
        with pytest.raises(asyncio.CancelledError):
            await ingestion._close_clients(CancelledClose(), NextClose())
        assert closed == [1]
    asyncio.run(run())


def callback(helper, *, enabled=True):
    path = ROOT / "backend/social_poster/agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_stage_after_agent")
    namespace = {
        "CallbackContext": object, "asyncio": asyncio,
        "use_memory_bank": enabled,
        "os": SimpleNamespace(environ={"APP_URL": RESOURCE}),
        "log": Mock(), "ingest_session_with_retry": helper,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_stage_after_agent"], namespace["log"]


@pytest.mark.parametrize("stage,enabled,flag,expected_stage", [
    ("simulated", True, None, "simulated"),
    ("posted", False, None, "posted"),
    ("posted", True, "memory_ingestion_scheduled", "posted"),
    ("posted", True, "memory_ingestion_attempted", "posted"),
    ("researching", True, None, "awaiting_approval"),
    ("drafting", True, None, "awaiting_approval"),
])
def test_callback_gates(stage, enabled, flag, expected_stage):
    helper = AsyncMock()
    function, logger = callback(helper, enabled=enabled)
    state = {"pipeline_stage": stage}
    if flag:
        state[flag] = True
    asyncio.run(function(SimpleNamespace(state=state, session=session())))
    helper.assert_not_awaited()
    logger.exception.assert_not_called()
    assert state["pipeline_stage"] == expected_stage


@pytest.mark.parametrize("status,accepted,done,succeeded", [
    ("api_accepted", True, False, None),
    ("operation_completed", True, True, True),
    ("operation_failed", True, True, False),
    ("skipped_no_events", False, None, None),
])
def test_callback_receipt_and_no_repeat(ingestion, status, accepted, done, succeeded):
    receipt = ingestion.IngestionReceipt(
        status=status, api_accepted=accepted,
        operation_name=(RESOURCE + "/operations/offline") if accepted else None,
        operation_completed=done, operation_succeeded=succeeded,
    )
    helper = AsyncMock(return_value=receipt)
    function, logger = callback(helper)
    context = SimpleNamespace(state={"pipeline_stage": "posted"}, session=session())
    async def run():
        await function(context)
        await function(context)
    asyncio.run(run())
    helper.assert_awaited_once_with(session=context.session, runtime_resource=RESOURCE)
    logger.exception.assert_not_called()
    state = context.state
    assert state["pipeline_stage"] == "posted"
    assert state["memory_ingestion_status"] == status
    assert state["memory_ingestion_api_accepted"] is accepted
    assert state["memory_ingestion_operation_completed"] is done
    assert state["memory_ingestion_operation_succeeded"] is succeeded
    assert state["memory_ingestion_searchable"] is None
    assert state.get("memory_ingestion_scheduled", False) is accepted


@pytest.mark.parametrize("cancelled", [False, True])
def test_callback_failure_never_restarts_or_changes_posted(cancelled):
    failure = asyncio.CancelledError() if cancelled else RuntimeError("offline failure")
    helper = AsyncMock(side_effect=failure)
    function, logger = callback(helper)
    context = SimpleNamespace(state={"pipeline_stage": "posted"}, session=session())
    async def run():
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await function(context)
        else:
            await function(context)
        await function(context)
    asyncio.run(run())
    helper.assert_awaited_once()
    assert context.state["pipeline_stage"] == "posted"
    assert context.state["memory_ingestion_status"] == "submission_unconfirmed"
    assert context.state["memory_ingestion_api_accepted"] is None
    assert not context.state.get("memory_ingestion_scheduled")


def test_callback_does_not_log_api_error_body():
    secret = "DO-NOT-LOG-OFFLINE-ERROR-BODY"
    helper = AsyncMock(side_effect=errors.ClientError(
        403, {"error": {"message": secret}}
    ))
    function, logger = callback(helper)
    context = SimpleNamespace(state={"pipeline_stage": "posted"}, session=session())
    asyncio.run(function(context))
    assert context.state["memory_ingestion_error_type"] == "ClientError"
    assert context.state["memory_ingestion_http_status"] == 403
    logger.exception.assert_not_called()
    assert secret not in repr(logger.mock_calls)
