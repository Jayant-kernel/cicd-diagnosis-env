# inference.py — run this from the project root
"""
LLM agent for the CI/CD Failure Diagnosis environment.

Env vars:
    API_BASE_URL  OpenAI-compatible base URL (e.g. https://api.openai.com/v1)
    MODEL_NAME    model to call (e.g. gpt-4o-mini)
    HF_TOKEN      HuggingFace token — used as API key when running on HF Spaces
    ENV_URL       running server URL (default: http://localhost:8000)
    NUM_EPISODES  how many episodes to run (default: 10)
"""
import json
import os
import sys
import time

from openai import OpenAI

from cicd_diagnosis_env.client import CICDEnv
from cicd_diagnosis_env.models import DiagnoseAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", "10"))

# HF_TOKEN doubles as API key when running on Spaces — fall back to OPENAI_API_KEY locally
_api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "no-key")
llm = OpenAI(api_key=_api_key, base_url=API_BASE_URL)

_SYSTEM = """You are an expert CI/CD engineer diagnosing pipeline failures.
You will receive a failure log. Respond ONLY with valid JSON (no markdown):
{
  "failure_category": "<dependency|config|flaky|code_bug|infra>",
  "root_cause": "<concise 1-2 sentence explanation>",
  "suggested_fix": "<concrete action>",
  "confidence": <float 0.0-1.0>
}"""


def diagnose(log, summary):
    msg = f"Error summary: {summary}\n\nFull log:\n{log}"
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": msg}],
        temperature=0.2,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content.strip()
    # strip markdown fences — some models add them even when told not to
    # this is a bit fragile but works for the models we're targeting
    if raw.startswith("```"):
        lines = raw.splitlines()
        # drop first line (```json or ```) and last line (```)
        raw = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(raw)
    return DiagnoseAction(
        failure_category=parsed["failure_category"],
        root_cause=parsed["root_cause"],
        suggested_fix=parsed["suggested_fix"],
        confidence=float(parsed.get("confidence", 0.8)),
    )


def run_episode(env, ep):
    obs = env.reset()
    print(f"[STEP {ep}.0] reset task_id={obs.task_id} stage={obs.pipeline_stage}")

    total = 0.0
    for attempt in range(1, 4):  # max 3 attempts per episode
        try:
            action = diagnose(obs.pipeline_log, obs.error_summary)
        except Exception as e:
            print(f"[STEP {ep}.{attempt}] LLM error: {e}", file=sys.stderr)
            break

        obs = env.step(action)
        total = obs.score
        print(
            f"[STEP {ep}.{attempt}] "
            f"cat={action.failure_category} "
            f"score={obs.score:.3f} "
            f"done={obs.done}"
        )
        if obs.done:
            break

    return total


def main():
    print(f"[START] model={MODEL_NAME} episodes={NUM_EPISODES} env={ENV_URL}")
    env = CICDEnv(base_url=ENV_URL)
    scores = []
    t0 = time.time()

    for ep in range(1, NUM_EPISODES + 1):
        try:
            s = run_episode(env, ep)
            scores.append(s)
        except Exception as e:
            print(f"[STEP {ep}] episode error: {e}", file=sys.stderr)
            scores.append(0.0)

    elapsed = time.time() - t0
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"[END] episodes={NUM_EPISODES} avg_score={avg:.4f} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
