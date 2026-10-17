"""Real SDK + real native authorized session; fake certificate setup and I/O."""

import asyncio
import importlib
import json
from pathlib import Path
import socket
import ssl
import sys
from types import MethodType, ModuleType, SimpleNamespace

import aiohttp
import google.auth
from google.auth.aio.transport import aiohttp as auth_aiohttp
from google.auth.aio.transport import sessions as auth_sessions
from google.auth.credentials import Credentials
from google.auth.transport import mtls
from google.genai import errors, types
import pytest


ROOT = Path(__file__).resolve().parents[1]
RESOURCE = ('projects/817286458936/locations/us-central1/'
            'reasoningEngines/7513438216494514176')
OPERATION = RESOURCE + '/operations/12345'
REAL_AUTHORIZED_SESSION = auth_sessions.AsyncAuthorizedSession


@pytest.fixture(autouse=True)
def isolation(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network, credential refresh or SDK internal resend forbidden')
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(google.auth, 'default', blocked)
    monkeypatch.setattr(mtls, 'should_use_client_cert', lambda: True)
    monkeypatch.setattr(mtls, 'has_default_client_cert_source', lambda: True)
    monkeypatch.setattr(mtls, 'default_client_cert_source', lambda: blocked)
    # The SDK's manual retry calls random.randint. Reaching it is a test failure.
    import google.genai._api_client as api_module
    monkeypatch.setattr(api_module.random, 'randint', blocked)


@pytest.fixture(scope='module')
def modules():
    prefix = '_lab2_native_mtls_tests'
    package = ModuleType(prefix)
    package.__path__ = [str(ROOT / 'backend/social_poster')]
    sys.modules[prefix] = package
    try:
        yield (importlib.import_module(prefix + '.memory_ingestion'),
               importlib.import_module(prefix + '.memory_ingestion_mtls'))
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + '.'):
                del sys.modules[name]


class OfflineCredentials(Credentials):
    def __init__(self):
        super().__init__()
        self.token = 'offline-placeholder'

    def refresh(self, request):
        raise AssertionError('Default credentials and refresh are forbidden')


class RawResponse(aiohttp.ClientResponse):
    def __init__(self, status, payload, read_error=None):
        # A real response subclass satisfies the SDK's response-type contract.
        # No connection, socket, certificate or external server is created.
        self.status = status
        self.payload = payload
        self.read_error = read_error
        self._closed = True
        self._body = None

    @property
    def headers(self):
        return {'Content-Type': 'application/json'}

    @property
    def reason(self):
        return 'Synthetic offline response'

    async def read(self):
        if self.read_error is not None:
            raise self.read_error
        if self._body is None:
            self._body = json.dumps(self.payload).encode('utf-8')
        return self._body

    async def json(self, **kwargs):
        return json.loads(await self.read())

    async def text(self, **kwargs):
        return (await self.read()).decode('utf-8')

    def close(self):
        self._closed = True


def make_session():
    from google.adk.events import Event
    from google.adk.sessions import Session
    return Session(
        id='mtls-offline-session', app_name='social_poster', user_id='offline-user',
        state={}, events=[Event(
            id='stable-event', author='user', timestamp=1700000000.0,
            content=types.Content(role='user', parts=[types.Part(text='Synthetic Python preference')]),
        )],
    )


def install_native(monkeypatch, handler, *, establish_mtls=True):
    requests, sessions = [], []

    class FakeAuthRequest:
        def __init__(self):
            self._closed = False

        async def __call__(self, url, method='GET', body=None, headers=None, timeout=None, **kwargs):
            assert method.upper() == 'POST'
            assert 'us-central1-aiplatform.mtls.googleapis.com/' in url
            assert kwargs.get('allow_redirects') is False
            assert headers.get('Authorization') == 'Bearer offline-placeholder'
            requests.append({'body': body, 'url': url, 'kwargs': kwargs, 'timeout': timeout})
            response = await handler(len(requests), body)
            return auth_aiohttp.Response(response)

        async def close(self):
            self._closed = True

    def factory(credentials):
        auth_request = FakeAuthRequest()
        native = REAL_AUTHORIZED_SESSION(credentials, auth_request=auth_request)
        native.configure_calls = 0
        native.request_limits = []
        original_request = native.request

        async def tracked_request(self, *args, **kwargs):
            self.request_limits.append({
                'total_attempts': kwargs.get('total_attempts'),
                'max_allowed_time': kwargs.get('max_allowed_time'),
            })
            return await original_request(*args, **kwargs)

        native.request = MethodType(tracked_request, native)

        async def configure(self, callback=None):
            assert callable(callback)
            self.configure_calls += 1
            self._is_mtls = establish_mtls

        native.configure_mtls_channel = MethodType(configure, native)
        sessions.append(native)
        return native

    monkeypatch.setattr(auth_sessions, 'AsyncAuthorizedSession', factory)
    return requests, sessions


def response(status=200, *, failed_lro=False):
    if status == 200:
        payload = {'name': OPERATION, 'done': failed_lro}
        if failed_lro:
            payload['error'] = {'code': 13, 'message': 'Synthetic failed operation'}
    else:
        payload = {'error': {'code': status, 'message': 'Synthetic offline API error'}}
    return RawResponse(status, payload)


@pytest.mark.parametrize('statuses', [(200,), (429, 503, 200), (408, 200)])
def test_native_submission_budget_and_snapshot(modules, monkeypatch, statuses):
    core, _ = modules
    source = make_session()

    async def handler(index, body):
        source.events[0].content.parts[0].text = 'Source changed after snapshot'
        return response(statuses[index - 1])

    requests, native = install_native(monkeypatch, handler)
    backoffs = []

    async def sleep(delay):
        backoffs.append(delay)

    receipt = asyncio.run(core.ingest_session_with_retry(
        source, runtime_resource=RESOURCE, credentials=OfflineCredentials(), sleep=sleep,
    ))
    assert receipt.api_accepted and receipt.operation_name == OPERATION
    assert receipt.memory_searchable is None
    assert len(requests) == len(statuses) and len(backoffs) == len(statuses) - 1
    assert all(item['body'] == requests[0]['body'] for item in requests)
    assert all(item['timeout'] == 30.0 for item in requests)
    assert all(limit == {'total_attempts': 1, 'max_allowed_time': float('inf')}
               for limit in native[0].request_limits)
    assert all(item._auth_request._closed for item in native)
    assert native[0].configure_calls == len(statuses)


@pytest.mark.parametrize('status', [400, 401, 403, 503, 307])
def test_native_http_errors_are_not_nested_retries(modules, monkeypatch, status):
    core, _ = modules

    async def handler(index, body):
        return response(status)

    requests, native = install_native(monkeypatch, handler)
    backoffs = []

    async def sleep(delay):
        backoffs.append(delay)

    with pytest.raises(errors.APIError) as caught:
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(), sleep=sleep,
        ))
    expected = 3 if status == 503 else 1
    assert caught.value.code == status
    assert len(requests) == expected and len(backoffs) == expected - 1
    assert all(item._auth_request._closed for item in native)


@pytest.mark.parametrize('kind', ['disconnect', 'auth_transport', 'auth_timeout', 'payload'])
def test_native_network_failures_only_use_outer_retry(modules, monkeypatch, kind):
    core, native_module = modules

    async def handler(index, body):
        if kind == 'disconnect':
            raise aiohttp.ServerDisconnectedError('Synthetic disconnect')
        if kind == 'auth_transport':
            raise native_module.auth_errors.TransportError('Synthetic transport failure')
        if kind == 'auth_timeout':
            raise native_module.auth_errors.TimeoutError('Synthetic request timeout')
        return RawResponse(200, {}, read_error=aiohttp.ClientPayloadError('Synthetic partial body'))

    requests, native = install_native(monkeypatch, handler)
    backoffs = []

    async def sleep(delay):
        backoffs.append(delay)

    import httpx
    with pytest.raises((httpx.ReadError, httpx.ReadTimeout)):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(), sleep=sleep,
        ))
    assert len(requests) == 3 and len(backoffs) == 2
    assert all(item._auth_request._closed for item in native)


def test_native_tls_verification_failure_stops(modules, monkeypatch):
    core, native_module = modules

    async def handler(index, body):
        raise native_module.auth_errors.TransportError('Synthetic transport failure') from ssl.SSLError('Synthetic TLS failure')

    requests, native = install_native(monkeypatch, handler)
    with pytest.raises(ValueError, match='TLS verification failed'):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(),
        ))
    assert len(requests) == 1 and native[0]._auth_request._closed


def test_native_failed_lro_is_not_resubmitted(modules, monkeypatch):
    core, _ = modules

    async def handler(index, body):
        return response(failed_lro=True)

    requests, native = install_native(monkeypatch, handler)
    receipt = asyncio.run(core.ingest_session_with_retry(
        make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(),
    ))
    assert receipt.status == 'operation_failed' and receipt.operation_error_code == 13
    assert len(requests) == 1 and native[0]._auth_request._closed


@pytest.mark.parametrize('phase', ['submission', 'backoff'])
def test_native_cancellation_propagates_and_closes(modules, monkeypatch, phase):
    core, _ = modules

    async def cancel():
        asyncio.current_task().cancel()
        await asyncio.sleep(0)

    async def handler(index, body):
        if phase == 'submission':
            await cancel()
        return response(429)

    async def sleep(delay):
        await cancel()

    requests, native = install_native(monkeypatch, handler)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(), sleep=sleep,
        ))
    assert len(requests) == 1 and native[0]._auth_request._closed


def test_native_deadline_cleans_active_requests(modules, monkeypatch):
    core, _ = modules
    active = 0

    async def handler(index, body):
        nonlocal active
        active += 1
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    async def sleep(delay):
        pass

    requests, native = install_native(monkeypatch, handler)
    with pytest.raises(TimeoutError):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(),
            sleep=sleep, attempt_timeout=0.01,
        ))
    assert len(requests) == 3 and active == 0 and native[0]._auth_request._closed
    assert all(item['timeout'] == 0.01 for item in requests)
    assert all(limit == {'total_attempts': 1, 'max_allowed_time': float('inf')}
               for limit in native[0].request_limits)


def test_native_refuses_standard_tls_fallback(modules, monkeypatch):
    core, _ = modules

    async def handler(index, body):
        raise AssertionError('No request is permitted before mTLS')

    requests, native = install_native(monkeypatch, handler, establish_mtls=False)
    with pytest.raises(ValueError, match='refusing a TLS fallback'):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(),
        ))
    assert not requests and native[0]._auth_request._closed


def test_native_missing_cert_source_stops_before_client(modules, monkeypatch):
    core, _ = modules
    monkeypatch.setattr(mtls, 'has_default_client_cert_source', lambda: False)
    with pytest.raises(ValueError, match='no native source'):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(),
        ))


def test_custom_httpx_is_refused_when_cert_required(modules):
    core, _ = modules
    with pytest.raises(ValueError, match='Custom HTTPX'):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(), transport=object(),
        ))


def test_native_dependency_drift_stops_before_network(modules, monkeypatch):
    core, native_module = modules
    monkeypatch.setattr(native_module, 'version', lambda name: 'unreviewed')
    with pytest.raises(ValueError, match='reviewed dependency'):
        asyncio.run(core.ingest_session_with_retry(
            make_session(), runtime_resource=RESOURCE, credentials=OfflineCredentials(),
        ))


@pytest.mark.parametrize('mode,expected_requests', [
    ('wrong_host', 0), ('wrong_method', 0), ('wrong_scope', 0),
    ('wrong_stream', 0), ('sdk_resend', 1), ('changed_payload', 1), ('fourth_attempt', 3),
])
def test_native_wire_guards_stop_before_unexpected_submission(modules, mode, expected_requests):
    core, native_module = modules
    prepared = core.prepare_session_ingestion(make_session(), RESOURCE)
    calls = []

    async def request(**kwargs):
        calls.append(kwargs)
        return auth_aiohttp.Response(response())

    async def close():
        pass

    native = SimpleNamespace(_is_mtls=True, request=request, close=close)
    wrapped = native_module.SingleAttemptMtlsSession(native, prepared, 30)
    base = {'scope': {'app_name': prepared.app_name, 'user_id': prepared.user_id},
            'streamId': prepared.stream_id}

    async def scenario():
        method = 'POST'
        url = 'https://us-central1-aiplatform.mtls.googleapis.com/v1beta1/' + RESOURCE + '/memories:ingestEvents'
        body = dict(base)
        if mode == 'wrong_host':
            url = url.replace('.mtls.googleapis.com', '.googleapis.com')
        elif mode == 'wrong_method':
            method = 'GET'
        elif mode == 'wrong_scope':
            body['scope'] = {'user_id': 'another-user'}
        elif mode == 'wrong_stream':
            body['streamId'] = 'different-stream'
        else:
            for _ in range(3 if mode == 'fourth_attempt' else 1):
                wrapped.start_attempt()
                await wrapped.request(method=method, url=url, data=json.dumps(body).encode())
            if mode != 'sdk_resend':
                wrapped.start_attempt()
            if mode == 'changed_payload':
                body['unexpected_change'] = True
        with pytest.raises((ValueError, RuntimeError)):
            await wrapped.request(method=method, url=url, data=json.dumps(body).encode())

    asyncio.run(scenario())
    assert len(calls) == expected_requests
