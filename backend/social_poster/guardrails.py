"""Inline Model Armor configuration and readable block responses."""

import asyncio
import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types


def _template_id(value: str) -> str:
    value = value.strip()
    return "" if value.lower() in ("", "off", "none", "disabled") else value


_REQUEST_TEMPLATE_ID = _template_id(os.environ.get("MODEL_ARMOR_TEMPLATE_ID", ""))
_OUTPUT_TEMPLATE_ID = _template_id(
    os.environ.get("MODEL_ARMOR_OUTPUT_TEMPLATE_ID", f"{_REQUEST_TEMPLATE_ID}-output")
)
# Only the request template controls the shared flag used by agents and bridge.
MODEL_ARMOR_ENABLED = bool(_REQUEST_TEMPLATE_ID)
if MODEL_ARMOR_ENABLED and not _OUTPUT_TEMPLATE_ID:
    raise ValueError(
        "MODEL_ARMOR_OUTPUT_TEMPLATE_ID must be a valid template ID when "
        "MODEL_ARMOR_TEMPLATE_ID is enabled; blank, off, none, and disabled "
        "values are not allowed"
    )

REFUSAL = (
    "I can't help with that request — it was flagged by our content safety "
    "checks. Happy to help with your next post idea instead."
)


def content_config(screen_input: bool = True) -> genai_types.GenerateContentConfig | None:
    """Screen root input and every agent's output; return None when disabled."""
    if not MODEL_ARMOR_ENABLED:
        return None
    project_id = os.environ.get("PROJECT_ID", "").strip()
    if not project_id:
        raise ValueError("PROJECT_ID is required when Model Armor is enabled")
    region = os.environ.get("MODEL_ARMOR_REGION", "us-central1").strip()
    template_prefix = f"projects/{project_id}/locations/{region}/templates/"
    return genai_types.GenerateContentConfig(
        model_armor_config=genai_types.ModelArmorConfig(
            prompt_template_name=f"{template_prefix}{_REQUEST_TEMPLATE_ID}" if screen_input else None,
            response_template_name=f"{template_prefix}{_OUTPUT_TEMPLATE_ID}",
        )
    )


def friendly_block(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Replace a Model Armor block without exposing the rejected content."""
    if not MODEL_ARMOR_ENABLED or llm_response.error_code != "MODEL_ARMOR":
        return None
    return LlmResponse(
        content=genai_types.Content(
            role="model", parts=[genai_types.Part.from_text(text=REFUSAL)]
        )
    )


def model(name: str) -> str | Gemini:
    """Keep unguarded models unchanged; retry guarded requests within the SDK."""
    if not MODEL_ARMOR_ENABLED:
        return name
    return Gemini(
        model=name,
        retry_options=genai_types.HttpRetryOptions(
            attempts=8,  # Includes the initial request.
            initial_delay=0.3,
            max_delay=8.0,
            exp_base=2.0,
            jitter=0.3,
            http_status_codes=[400, 429],
        ),
    )


# Independent of Model Armor: disabling its templates must not disable DLP.
PII_REDACTION = os.environ.get("PII_REDACTION", "off").strip().lower() in ("on", "true", "1")
DLP_TIMEOUT_SECONDS = 5.0  # Per SDK call; automatic retries are disabled.
DLP_PROCESSING_SECONDS = 12.0  # Both passes, including lazy client creation.
MAX_PII_TEXT_BYTES = 32 * 1024
_BOUNDARY_PASSES = (
    (("EMAIL_ADDRESS", "PHONE_NUMBER"), "POSSIBLE"),
    (("PERSON_NAME",), "LIKELY"),
)
_FULL_PASSES = (
    _BOUNDARY_PASSES[0],
    (("PERSON_NAME", "LOCATION"), "LIKELY"),
)
_INTERNATIONAL_PHONE_TEXT = re.compile(
    r"(?<![\w+./-])\+(?:"
    r"[1-9][0-9]{7,14}|"
    r"[1-9][0-9]{0,2}(?:[ \u00a0\u202f-]+(?:[0-9]{1,8}|\([0-9]{1,6}\))){1,5}|"
    r"[1-9][0-9]{0,2}[ \u00a0\u202f-]*\([0-9]{1,6}\)"
    r"[ \u00a0\u202f-]*[0-9]{1,8}(?:[ \u00a0\u202f-]+[0-9]{1,8}){0,4}"
    r")(?![\w+./-]|[ \u00a0\u202f-]*[0-9(])"
)


def _redact_international_phone_text(text: str) -> str:
    """Conservatively mask explicit + phone shapes; not numbering-plan validation."""
    def replace(match: re.Match[str]) -> str:
        digits = sum(char in "0123456789" for char in match[0])
        return "[PHONE_NUMBER]" if 8 <= digits <= 15 else match[0]

    return _INTERNATIONAL_PHONE_TEXT.sub(replace, text)

_NATIONAL_PHONE_TEXT = re.compile(
    r"(?<![\w+./-])(?:"
    r"0[1-9][0-9]{8,9}|"
    r"0[1-9][0-9]{1,4}(?:[ \u00a0\u202f-]+[0-9]{1,8}){1,4}|"
    r"\(0[1-9][0-9]{1,4}\)[ \u00a0\u202f-]*[0-9]{1,8}(?:[ \u00a0\u202f-]+[0-9]{1,8}){0,3}"
    r")(?![\w+/-]|\.[0-9a-zA-Z]|[ \u00a0\u202f-]*[0-9(])"
)


def _redact_national_phone_text(text: str) -> str:
    """Conservatively mask UK national 0x phone shapes; not numbering-plan validation."""
    def replace(match: re.Match[str]) -> str:
        digits = sum(char in "0123456789" for char in match[0])
        return "[PHONE_NUMBER]" if 10 <= digits <= 11 else match[0]

    return _NATIONAL_PHONE_TEXT.sub(replace, text)



class PiiRedactionError(RuntimeError):
    """A generic failure safe to surface without submitted text or SDK details."""


def _redact_sync(text: str, passes: tuple, project_id: str) -> str:
    # No credentials, SDK import, client, or raw-text cache at module import time.
    from google.cloud import dlp_v2

    with dlp_v2.DlpServiceClient() as client:
        for info_types, likelihood in passes:
            response = client.deidentify_content(
                request={
                    "parent": f"projects/{project_id}/locations/global",
                    "inspect_config": {
                        "info_types": [{"name": name} for name in info_types],
                        "min_likelihood": likelihood,
                    },
                    "deidentify_config": {
                        "info_type_transformations": {
                            "transformations": [{
                                "primitive_transformation": {"replace_with_info_type_config": {}}
                            }],
                        },
                    },
                    "item": {"value": text},
                },
                timeout=DLP_TIMEOUT_SECONDS,
                retry=None,
            )
            cleaned = response.item.value
            if (
                not isinstance(cleaned, str)
                or (text and not cleaned)
                or len(cleaned.encode("utf-8")) > MAX_PII_TEXT_BYTES
            ):
                raise PiiRedactionError("PII redaction is unavailable")
            text = cleaned
    # Supplement detector misses only after every DLP pass and client cleanup
    # succeeds. Never substitute local matching for a failed DLP operation.
    text = _redact_international_phone_text(text)
    text = _redact_national_phone_text(text)
    if len(text.encode("utf-8")) > MAX_PII_TEXT_BYTES:
        raise PiiRedactionError("PII redaction is unavailable")
    return text


async def _redact(text: str, passes: tuple) -> str:
    if not isinstance(text, str):
        raise PiiRedactionError("PII redaction is unavailable")
    try:
        project_id = os.environ.get("PROJECT_ID", "").strip()
        if re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project_id) is None:
            raise PiiRedactionError("PII redaction is unavailable")
        if len(text.encode("utf-8")) > MAX_PII_TEXT_BYTES:
            raise PiiRedactionError("PII redaction is unavailable")
        if not text:
            return text
        async with asyncio.timeout(DLP_PROCESSING_SECONDS):
            return await asyncio.to_thread(_redact_sync, text, passes, project_id)
    except Exception:
        # Do not expose exception text/chains or return the unredacted input.
        raise PiiRedactionError("PII redaction is unavailable") from None


async def redact_boundary(text: str) -> str:
    """Redact email/phone/name text, leaving locations; raise on any failure."""
    return await _redact(text, _BOUNDARY_PASSES)


async def redact_pii(text: str) -> str:
    """Redact email/phone/name/location text; raise on any failure."""
    return await _redact(text, _FULL_PASSES)


async def redact_pii_text(text: str) -> dict[str, Any]:
    """Remove names, emails, phone numbers, and locations from source/post text.

    This explicit tool always attempts redaction, independently of PII_REDACTION.
    A failed check returns an error without the submitted text.
    """
    try:
        return {"status": "success", "redacted_text": await redact_pii(text)}
    except Exception:
        return {"isError": True, "status": "error", "error": "PII redaction is unavailable."}


# Buffer's local integration expects top-level text; its authoritative schema
# has not been verified. Missing/non-string text must never bypass protection.
_POST_TOOLS = frozenset({"linkedin_create_post", "create_post"})
_POST_REFUSAL = "I couldn't complete the privacy checks, so no posting actions were submitted."


def _post_text(args: dict[str, Any] | None) -> str:
    text = args.get("text") if isinstance(args, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise PiiRedactionError("Post text is invalid")
    return text


async def redact_post_calls(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    """Clean the entire posting batch before ADK stores/requests confirmation.

    Work on a complete deep copy: failure discards every call in this response,
    rather than dispatching a partially cleaned batch. Later callbacks would be
    short-circuited by this replacement; keep this after friendly_block.
    """
    if not PII_REDACTION or not llm_response.content:
        return None
    try:
        response = llm_response.model_copy(deep=True)
        has_post = False
        for part in response.content.parts or []:
            call = part.function_call
            if call and call.name in _POST_TOOLS:
                has_post = True
                text = _post_text(call.args)
                call.args["text"] = await redact_pii(text)
        return response if has_post else None
    except Exception:
        # No function calls, rejected text, or original metadata on failure.
        return LlmResponse(content=genai_types.Content(
            role="model", parts=[genai_types.Part.from_text(text=_POST_REFUSAL)]
        ))


async def redact_before_post(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Recheck initial and resumed calls; never change an approved payload."""
    if not PII_REDACTION or tool.name not in _POST_TOOLS:
        return None
    try:
        text = _post_text(args)
        cleaned = await redact_pii(text)
    except Exception:
        tool_context.state["pipeline_stage"] = "awaiting_approval"
        return {"isError": True, "status": "error", "error": "Post privacy checks could not be completed."}
    if cleaned == text:
        return None
    tool_context.state["pipeline_stage"] = "awaiting_approval"
    # Returning a nonempty response skips the remaining callbacks and tool.
    # Only a new call may request fresh approval through the existing MCP gate.
    return {"isError": True, "status": "requires_review", "redacted_text": cleaned}
