import asyncio
import random

import httpx
from google.genai import errors

RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRYABLE_NETWORK = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
)


async def submit_with_retry(
    submit, *, attempt_timeout=30.0, sleep=asyncio.sleep
):
    """Attempt submission at most three times; return without polling LRO."""
    for attempt in range(1, 4):
        deadline = asyncio.timeout(attempt_timeout)
        try:
            async with deadline:
                return await submit()
        except TimeoutError:
            if not deadline.expired() or attempt == 3:
                raise
        except errors.APIError as exc:
            if exc.code not in RETRYABLE_CODES or attempt == 3:
                raise
        except RETRYABLE_NETWORK:
            if attempt == 3:
                raise

        ceiling = min(8.0, 2.0 ** (attempt - 1))
        await sleep(random.uniform(0.0, ceiling))
