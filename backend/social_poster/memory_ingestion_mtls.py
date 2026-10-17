"""Use the SDK's native certificate-bound session with one POST per attempt.

This adapter relies on reviewed private SDK session access. Pinned versions and
offline wire-contract tests are required before upgrading these dependencies.
It never reads, exports, or installs certificate/key material itself.
"""

from __future__ import annotations

import asyncio
import hashlib
from importlib.metadata import version
import json
import ssl
from urllib.parse import urlsplit

import aiohttp
from google.auth.aio.transport import aiohttp as auth_aiohttp
from google.auth.aio.transport.sessions import AsyncAuthorizedSession
from google.auth.transport import mtls
from google.genai.types import HttpOptions, HttpRetryOptions
import httpx
import vertexai

from .memory_ingestion import _close_clients, classify_operation
from .memory_ingestion_retry import submit_with_retry


# Match the exception classes actually used by the reviewed auth transport.
auth_errors = auth_aiohttp.exceptions


def _network_error(error):
    """Normalize before the SDK's aiohttp fallback can resubmit the POST."""
    chain = []
    current = error
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(item, (ssl.SSLError, aiohttp.ClientSSLError)) for item in chain):
        return ValueError('TLS verification failed; no submission retry is allowed')
    if any(isinstance(item, (asyncio.TimeoutError, auth_errors.TimeoutError)) for item in chain):
        return httpx.ReadTimeout('Native authenticated request timed out')
    if any(isinstance(item, aiohttp.ClientConnectorError) for item in chain):
        return httpx.ConnectError('Native authenticated connection failed')
    return httpx.ReadError('Native authenticated request or response failed')


class SingleAttemptMtlsSession:
    """Delegate authentication/mTLS; suppress redirects and nested retries."""

    def __init__(self, native, prepared, attempt_timeout):
        self.native = native
        self.prepared = prepared
        self.attempt_timeout = attempt_timeout
        self.used = False
        self.body_hash = None
        self.requests_started = 0

    @property
    def _is_mtls(self):
        return getattr(self.native, '_is_mtls', False) is True

    async def configure_mtls_channel(self, client_cert_callback=None):
        await self.native.configure_mtls_channel(client_cert_callback)
        if not self._is_mtls:
            raise ValueError('Native mTLS could not be established; refusing a TLS fallback')

    def start_attempt(self):
        self.used = False

    async def request(self, *, method, url, data=None, headers=None, **kwargs):
        parsed = urlsplit(str(url))
        expected_host = self.prepared.location + '-aiplatform.mtls.googleapis.com'
        if (
            str(method).upper() != 'POST' or parsed.scheme != 'https'
            or parsed.hostname != expected_host or parsed.port not in (None, 443)
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment
            or not parsed.path.endswith('/' + self.prepared.name + '/memories:ingestEvents')
        ):
            raise ValueError('Only the expected mTLS ingestion endpoint is permitted')
        if not self._is_mtls:
            raise ValueError('Certificate-bound transport is required before submitting')
        if self.used or self.requests_started >= 3:
            raise RuntimeError('Unexpected SDK resubmission refused before networking')
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError('Ingestion request must contain serialized bytes')
        body = json.loads(data)
        if (
            body.get('scope') != {'app_name': self.prepared.app_name, 'user_id': self.prepared.user_id}
            or body.get('streamId', body.get('stream_id')) != self.prepared.stream_id
        ):
            raise ValueError('Ingestion scope or stream differs from the snapshot')
        body_hash = hashlib.sha256(data).hexdigest()
        if self.body_hash is not None and self.body_hash != body_hash:
            raise ValueError('Retry payload changed; refusing submission')
        self.body_hash = body_hash
        self.used = True
        self.requests_started += 1
        # Do not replace the native auth_request: it owns the certificate context.
        # submit_with_retry owns the whole-attempt deadline, including auth and
        # buffering. Keep the SDK's finite HTTP timeout, but avoid a competing
        # auth-session total timer obscuring that deadline as a ReadTimeout.
        kwargs.update(total_attempts=1, max_allowed_time=float('inf'),
                      allow_redirects=False)
        raw = None
        try:
            response = await self.native.request(
                method=method, url=url, data=data, headers=headers, **kwargs,
            )
            raw = getattr(response, '_response', None)
            if not isinstance(raw, aiohttp.ClientResponse):
                raise TypeError('Unsupported native response; SDK contract needs review')
            # Buffer inside this boundary so body failures cannot trigger SDK retries.
            await raw.read()
            return response
        except asyncio.CancelledError:
            if raw is not None:
                raw.close()
            raise
        except (
            aiohttp.ClientError, auth_errors.TransportError,
            auth_errors.ResponseError, auth_errors.TimeoutError, asyncio.TimeoutError,
        ) as error:
            if raw is not None:
                raw.close()
            raise _network_error(error) from None

    async def aclose(self):
        request = getattr(self.native, '_auth_request', None)
        if request is not None and not getattr(request, '_closed', False):
            await self.native.close()


async def ingest_prepared_with_mtls(prepared, *, attempt_timeout, sleep, credentials=None):
    for package, expected in {
        'google-genai': '2.11.0', 'google-auth': '2.55.2',
        'google-cloud-aiplatform': '1.161.0', 'aiohttp': '3.14.1',
    }.items():
        if version(package) != expected:
            raise ValueError('Native mTLS adapter requires a reviewed dependency version: ' + package)
    if not mtls.has_default_client_cert_source():
        raise ValueError('A client certificate was requested but no native source is available')
    client = None
    api = None
    wrapped = None
    previous = None
    try:
        client = vertexai.Client(
            project=prepared.project, location=prepared.location, credentials=credentials,
            http_options=HttpOptions(
                retry_options=HttpRetryOptions(attempts=1),
                timeout=max(1, int(attempt_timeout * 1000)),
            ),
        ).aio
        api = client._api_client
        if not api._use_google_auth_async() or not api._use_aiohttp():
            raise ValueError('SDK native asynchronous certificate authentication is unavailable')
        native = await api._get_aiohttp_session()
        if not isinstance(native, AsyncAuthorizedSession):
            raise TypeError('SDK did not create the reviewed native authorized session')
        wrapped = SingleAttemptMtlsSession(native, prepared, attempt_timeout)
        if api._http_options.aiohttp_client is not None:
            raise ValueError('An unexpected custom native session was already configured')
        previous = api._get_aiohttp_session

        async def get_wrapped_session():
            return wrapped

        # Scope the hook to this dedicated client. Keep the SDK's session registry
        # and auth transport intact so native ownership and mTLS still work.
        api._get_aiohttp_session = get_wrapped_session

        async def submit():
            wrapped.start_attempt()
            return await client.agent_engines.memories.ingest_events(**prepared.request_kwargs())

        operation = await submit_with_retry(submit, attempt_timeout=attempt_timeout, sleep=sleep)
        return classify_operation(operation)
    finally:
        if api is not None and previous is not None:
            api._get_aiohttp_session = previous
        # The SDK owns the original native session. The wrapper is a fallback
        # closer if SDK cleanup fails before reaching its native session registry.
        await _close_clients(client, wrapped)
