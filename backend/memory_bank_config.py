"""Configure what Memory Bank remembers for a deployed engine.

Memory Bank defaults extract "whatever seems worth keeping". For a social-post
agent that is too vague, so this pins it to the things that should change the
next draft: stated preferences, explicit instructions, and the user's posting
style. It also expires memories, so a workshop project does not hoard them.

  uv run python backend/memory_bank_config.py show
  uv run python backend/memory_bank_config.py apply

Idempotent. Affects only future memory generation; existing memories stay.
Applied with a PATCH on the engine's contextSpec, so it needs no redeploy —
but check it survives yours: `show` after every `agents-cli deploy`.

Heads-up: deleting the engine deletes its memories (Lab 2, teardown).
"""
import json
import os
import sys

import google.auth
import google.auth.transport.requests
import httpx

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or sys.exit(
    "Set GOOGLE_CLOUD_PROJECT to your GCP project ID (export GOOGLE_CLOUD_PROJECT=...)."
)
REGION = os.environ.get("ENGINE_REGION", "us-central1")
ENGINE = os.environ.get("AGENT_ENGINE_ID") or sys.exit(
    "Set AGENT_ENGINE_ID to your deployed engine's numeric ID (see `agents-cli deploy --list`)."
)
URL = (f"https://{REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}"
       f"/locations/{REGION}/reasoningEngines/{ENGINE}")

CONFIG = {
    "customizationConfigs": [{
        "memoryTopics": [
            {"managedMemoryTopic": {"managedTopicEnum": "USER_PREFERENCES"}},
            {"managedMemoryTopic": {"managedTopicEnum": "EXPLICIT_INSTRUCTIONS"}},
            {"customMemoryTopic": {
                "label": "posting_style",
                "description": "How the user likes their social posts written: tone, "
                               "openers, sign-offs, hashtag habits, words to avoid.",
            }},
        ],
    }],
    "ttlConfig": {"defaultTtl": "7776000s"},  # 90 days
}


def _headers():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(google.auth.transport.requests.Request())
    return {"Authorization": f"Bearer {creds.token}", "x-goog-user-project": PROJECT}


def show():
    r = httpx.get(URL, headers=_headers(), timeout=30)
    r.raise_for_status()
    print(json.dumps(r.json().get("contextSpec", {}), indent=1))


def apply():
    # Read-modify-write so we keep whatever else the engine has under memoryBankConfig.
    cur = httpx.get(URL, headers=_headers(), timeout=30).json()
    mb = cur.get("contextSpec", {}).get("memoryBankConfig", {})
    mb.update(CONFIG)
    r = httpx.patch(URL, headers=_headers(), timeout=60,
                    params={"updateMask": "contextSpec.memoryBankConfig"},
                    json={"contextSpec": {"memoryBankConfig": mb}})
    if r.status_code >= 400:
        sys.exit(f"{r.status_code}: {r.text[:600]}")
    print("applied; operation:", r.json().get("name", "?"))


if __name__ == "__main__":
    {"show": show, "apply": apply}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
