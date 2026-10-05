import os
import pathlib
import sys
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.skills import load_skill_from_dir
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
from google.adk.tools.skill_toolset import SkillToolset
from google.adk.tools.tool_context import ToolContext
from mcp import StdioServerParameters

from .tools import generate_image

SKILLS_DIR = pathlib.Path(__file__).resolve().parents[2] / "skills"
POST_VIA = os.environ.get("POST_VIA", "").strip().lower()
if POST_VIA not in ("", "buffer", "linkedin"):
    raise ValueError("POST_VIA must be 'buffer', 'linkedin', or blank for both.")

web_researcher = Agent(
    name="web_researcher",
    model="gemini-2.5-flash",
    description="Researches facts, dates, and context on the web for social posts.",
    instruction="""Use Google Search to research the given idea or question.

Workflow:
Return a concise factual summary relevant to the requested post, with source
links and dates where relevant. Prefer authoritative, primary sources. Flag
uncertainty or conflicting evidence instead of inventing facts. Return research
findings only; the calling agent will draft the social media post.""",
    tools=[google_search],
    output_key="research_summary",
)

skill_toolset = SkillToolset(
    skills=[
        load_skill_from_dir(SKILLS_DIR / "post-formatter"),
        load_skill_from_dir(SKILLS_DIR / "platform-style"),
        load_skill_from_dir(SKILLS_DIR / "brand-voice"),
        load_skill_from_dir(SKILLS_DIR / "poster-style"),
    ],
)

buffer_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.buffer.com/mcp",
        headers={"Authorization": f"Bearer {os.environ.get('BUFFER_API_KEY', '')}"},
    ),
    tool_filter=["get_account", "list_channels", "create_post"],
    require_confirmation=True,
)

linkedin_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(SKILLS_DIR.parent / "mcp" / "linkedin_server.py")],
            env={
                "DRY_RUN": os.environ.get("DRY_RUN", "true"),
                "LINKEDIN_ACCESS_TOKEN": os.environ.get("LINKEDIN_ACCESS_TOKEN", ""),
            },
        ),
    ),
    tool_filter=["get_profile", "create_post"],
    # Buffer also exposes create_post; Gemini requires unique tool names.
    tool_name_prefix="linkedin",
    require_confirmation=True,
)

if POST_VIA == "buffer":
    posting_toolsets = [buffer_toolset]
    posting_routing = """Publish exclusively via Buffer (POST_VIA=buffer).
The direct LinkedIn tools are unavailable in this configuration."""
elif POST_VIA == "linkedin":
    posting_toolsets = [linkedin_toolset]
    posting_routing = """Publish exclusively via the direct LinkedIn tools
(POST_VIA=linkedin). Buffer is unavailable. These tools support immediate
LinkedIn posts only; explain that scheduling or other platforms require Buffer
and do not publish immediately as a substitute."""
else:
    posting_toolsets = [buffer_toolset, linkedin_toolset]
    posting_routing = """Both publishers are available:
- Use linkedin_create_post for an immediate LinkedIn post with an image.
- Use Buffer for scheduling or posting to other platforms.
- For an immediate text-only LinkedIn post, use linkedin_create_post unless
  the user explicitly requests Buffer."""

# idea -> researching -> drafting -> awaiting_approval -> posted
_TOOL_STAGES = {
    "web_researcher": "researching",
    "load_skill": "drafting",
    "generate_image": "drafting",
}


def _init_stage(callback_context: CallbackContext) -> None:
    if "pipeline_stage" not in callback_context.state:
        callback_context.state["pipeline_stage"] = "idea"


def _track_stage(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict
) -> dict | None:
    if tool.name in _TOOL_STAGES:
        tool_context.state["pipeline_stage"] = _TOOL_STAGES[tool.name]
    elif (
        tool.name in ("create_post", "linkedin_create_post")
        and tool_response.get("isError") is False
    ):
        tool_context.state["pipeline_stage"] = "posted"  # only a completed MCP call
    return None  # Preserve the tool result.  # never modify the tool result


def _stage_after_agent(callback_context: CallbackContext) -> None:
    # A turn that ends mid-pipeline is waiting on the user (draft approval).
    if callback_context.state.get("pipeline_stage") in ("researching", "drafting"):
        callback_context.state["pipeline_stage"] = "awaiting_approval"


root_agent = Agent(
    name="social_poster",
    model="gemini-2.5-flash",
    description="Helps the user turn an idea into a polished social media post.",
    instruction=f"""You help the user turn an idea into a social media post.

Workflow:
1.  If the idea needs facts, dates, statistics, or current context, call
    web_researcher first, before drafting. Use its findings to ground the post
    and include relevant source links. If research cannot verify a claim, leave
    it out or clearly state the uncertainty.
2.  Before drafting or revising a post, use load_skill to load and follow
    post-formatter, platform-style, and brand-voice. If the user has not named a
    platform, ask which one they want. Use load_skill_resource to read the matching
    platform-style reference file before drafting.
3.  When the user asks for an image, first use load_skill to load poster-style and
    follow its visual rules. Then call generate_image with a detailed visual
    description covering the subject, composition, style, colours, and lighting,
    incorporating any preferences the user provides. If the tool returns success,
    tell the user where the file was saved using the returned image_path. If it
    returns an error, report the error instead of claiming an image was saved.
4.  ALWAYS show the finished draft to the user and get explicit approval before
    any posting action, following the publishing route below.

Publishing route:
{posting_routing}
- When using LinkedIn, call linkedin_get_profile if needed, then
  linkedin_create_post with the approved text and the generated local image_path
  when an image is wanted.
- When using Buffer, resolve the organization with get_account and its channel
  with list_channels; ask the user to choose if multiple match. For an immediate
  post, call create_post with mode="shareNow" and schedulingType="automatic",
  rather than schedule.

Posting rules (strict):
- Only post AFTER the user explicitly approves the exact finished draft shown
  in this conversation. "Yes", "post it", or "approved" counts when responding
  to the approval request; silence or enthusiasm about the idea does not.
- If the draft changes after approval, show it again and get fresh approval.
- Always honour the tool confirmation gate. Rejection means do not post.
- Schedule or add to the queue only when the user explicitly requests it and
  approves the draft and timing.
- Buffer posts must be text only; never pass a local image_path as an asset.
  If posting via Buffer with an image generated, tell the user the image is
  available locally only. LinkedIn can attach it using image_path.
- After posting, share the tool's returned confirmation and status. Report
  errors honestly and do not claim a scheduled or queued post is published.
  If the LinkedIn result has dry_run=true, say it was a simulation and nothing
  was posted, even if the result includes a post_url.""",
    tools=[
        skill_toolset,
        generate_image,
        AgentTool(agent=web_researcher),
        *posting_toolsets,
    ],
    # ...name, model, description, instruction, tools as before...
    before_agent_callback=_init_stage,
    after_tool_callback=_track_stage,
    after_agent_callback=_stage_after_agent,
)
