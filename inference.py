# inference.py — run this from the project root
import json
import os
import sys

from openai import OpenAI

from cicd_diagnosis_env.client import CICDEnv
from cicd_diagnosis_env.models import DiagnoseAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
NUM_EPISODES = int(os.getenv("NUM_EPISODES", "10"))
MAX_STEPS = 3

_api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "no-key")
llm = OpenAI(api_key=_api_key, base_url=API_BASE_URL)

_SYSTEM = """You are an expert CI/CD engineer diagnosing pipeline failures.
Respond ONLY with valid JSON (no markdown):
{
  "failure_category": "<dependency|config|flaky|code_bug|infra>",
  "root_cause": "<concise 1-2 sentence explanation>",
  "suggested_fix": "<concrete action>",
  "confidence": <float 0.0-1.0>
}"""


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rstr}", flush=True)


def call_llm(log, summary):
    msg = f"Error summary: {summary}\n\nFull log:\n{log}"
    resp = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": msg}],
        temperature=0.2,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(raw)
    return DiagnoseAction(
        failure_category=parsed["failure_category"],
        root_cause=parsed["root_cause"],
        suggested_fix=parsed["suggested_fix"],
        confidence=float(parsed.get("confidence", 0.8)),
    )


def run_episode(env, ep_num):
    obs = env.reset()
    task_name = f"task{obs.task_id}"
    log_start(task=task_name, env="cicd_diagnosis_env", model=MODEL_NAME)

    rewards = []
    steps = 0
    success = False
    score = 0.0

    for step in range(1, MAX_STEPS + 1):
        error = None
        try:
            action = call_llm(obs.pipeline_log, obs.error_summary)
            action_str = f"diagnose(category={action.failure_category})"
        except Exception as e:
            error = str(e)
            action_str = "diagnose(error)"
            log_step(step, action_str, 0.0, True, error)
            rewards.append(0.0)
            steps = step
            break

        obs = env.step(action)
        reward = obs.reward if obs.reward is not None else 0.0
        rewards.append(reward)
        steps = step
        score = obs.score

        log_step(step, action_str, reward, obs.done, error)

        if obs.done:
            success = score >= 0.5
            break

    log_end(success=success, steps=steps, score=score, rewards=rewards)
    return score


def main():
    env = CICDEnv(base_url=ENV_URL)
    for ep in range(1, NUM_EPISODES + 1):
        try:
            run_episode(env, ep)
        except Exception as e:
            print(f"[DEBUG] episode {ep} error: {e}", file=sys.stderr, flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])


if __name__ == '__main__':
    main()
