# CI/CD Failure Diagnosis Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete OpenEnv-compliant RL environment for CI/CD failure diagnosis with hybrid log generation, deterministic grading, and an LLM inference script.

**Architecture:** FastAPI server inside Docker exposes `/reset` and `/step` endpoints. `log_generator.py` produces synthetic logs by randomizing seed templates — the injected failure metadata is stored in server state so `graders.py` can score deterministically. Three difficulty tiers map to three grader functions.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, OpenEnv core SDK, OpenAI-compatible client (for inference.py), Docker, uvicorn

---

## File Map

| File | Role |
|------|------|
| `cicd_diagnosis_env/server/log_generator.py` | Hybrid template engine — seed templates + randomization, returns (log_str, metadata) |
| `cicd_diagnosis_env/models.py` | DiagnoseAction, PipelineObservation, PipelineState Pydantic dataclasses |
| `cicd_diagnosis_env/server/graders.py` | grade_task1, grade_task2, grade_task3 — return (score, feedback) |
| `cicd_diagnosis_env/server/environment.py` | CICDEnvironment(Environment) — reset/step/state property |
| `cicd_diagnosis_env/server/app.py` | FastAPI app — manual routes /reset /step /state /health |
| `cicd_diagnosis_env/client.py` | CICDEnv(EnvClient) — _step_payload, _parse_result, _parse_state |
| `inference.py` | LLM agent loop — [START]/[STEP]/[END] structured logs |
| `cicd_diagnosis_env/Dockerfile` | Server container — port 8000 |
| `cicd_diagnosis_env/openenv.yaml` | OpenEnv manifest |
| `cicd_diagnosis_env/README.md` | Environment documentation |

---

## Task 1: log_generator.py

**Files:**
- Create: `cicd_diagnosis_env/server/log_generator.py`

- [ ] **Step 1: Create the file with imports and helpers**

```python
# cicd_diagnosis_env/server/log_generator.py
import random
import uuid
from datetime import datetime, timedelta

# packages that can appear in task1 dependency errors
_PACKAGES = [
    "pytest-cov", "httpx", "pydantic", "sqlalchemy",
    "celery", "redis", "boto3", "fastapi", "uvicorn",
    "alembic", "mypy", "black", "ruff", "anyio",
]

_REPOS = [
    "api-gateway", "user-service", "payment-service",
    "notification-worker", "data-pipeline", "auth-service",
]

_BRANCHES = ["main", "develop", "feat/auth-refactor", "fix/retry-logic", "release/v2.1"]

def _ts(base: datetime, offset_s: int) -> str:
    return (base + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%SZ")

def _rand_line() -> int:
    return random.randint(12, 340)
```

- [ ] **Step 2: Add Task 1 generator — single ModuleNotFoundError**

```python
def _task1_log() -> tuple[str, dict]:
    """Easy: one missing package, one stage fails."""
    base = datetime.utcnow().replace(microsecond=0)
    pkg = random.choice(_PACKAGES)
    repo = random.choice(_REPOS)
    branch = random.choice(_BRANCHES)
    line = _rand_line()
    run_id = random.randint(1000, 9999)

    log = f"""##[group]Run details
Repository: {repo}
Branch:     {branch}
Run ID:     {run_id}
Triggered:  push
##[endgroup]

{_ts(base, 0)} [INFO ] Pipeline started
{_ts(base, 2)} [INFO ] Stage: checkout — OK
{_ts(base, 4)} [INFO ] Stage: setup-python — OK
{_ts(base, 6)} [INFO ] Stage: install-dependencies — OK
{_ts(base, 8)} [INFO ] Stage: lint — OK
{_ts(base, 10)} [INFO ] Stage: test — RUNNING
{_ts(base, 11)} [ERROR] Traceback (most recent call last):
{_ts(base, 11)} [ERROR]   File "tests/test_main.py", line {line}, in test_endpoint
{_ts(base, 11)} [ERROR]     import {pkg}
{_ts(base, 11)} [ERROR] ModuleNotFoundError: No module named '{pkg}'
{_ts(base, 12)} [ERROR] Stage: test — FAILED (exit code 1)
{_ts(base, 13)} [INFO ] Stage: deploy — SKIPPED
{_ts(base, 13)} [ERROR] Pipeline FAILED
"""

    meta = {
        "task_id": 1,
        "failure_category": "dependency",
        "root_cause": f"missing package: {pkg}",
        "expected_fix": f"add {pkg} to requirements.txt or pyproject.toml",
        "failed_stage": "test",
        "pkg": pkg,
    }
    return log.strip(), meta
```

- [ ] **Step 3: Add Task 2 generator — config error causing cascading test failures**

```python
def _task2_log() -> tuple[str, dict]:
    """Medium: bad env var config causes 3 tests to fail downstream."""
    base = datetime.utcnow().replace(microsecond=0)
    repo = random.choice(_REPOS)
    branch = random.choice(_BRANCHES)
    run_id = random.randint(1000, 9999)
    var_name = random.choice(["DATABASE_URL", "REDIS_URL", "SECRET_KEY", "API_BASE_URL"])
    test_names = random.sample(
        ["test_create_user", "test_login", "test_refresh_token",
         "test_get_profile", "test_update_settings", "test_delete_account"],
        3,
    )
    lines = [_rand_line() for _ in range(3)]

    failing_tests = ""
    for i, (t, ln) in enumerate(zip(test_names, lines)):
        failing_tests += f"""{_ts(base, 14 + i)} [ERROR] FAILED tests/test_api.py::{t}
{_ts(base, 14 + i)} [ERROR]   File "tests/test_api.py", line {ln}, in {t}
{_ts(base, 14 + i)} [ERROR]     assert response.status_code == 200
{_ts(base, 14 + i)} [ERROR] AssertionError: assert 500 == 200
"""

    log = f"""##[group]Run details
Repository: {repo}
Branch:     {branch}
Run ID:     {run_id}
Triggered:  push
##[endgroup]

{_ts(base, 0)} [INFO ] Pipeline started
{_ts(base, 2)} [INFO ] Stage: checkout — OK
{_ts(base, 4)} [INFO ] Stage: setup-python — OK
{_ts(base, 6)} [INFO ] Stage: install-dependencies — OK
{_ts(base, 8)} [INFO ] Stage: lint — OK
{_ts(base, 10)} [INFO ] Stage: test — RUNNING
{_ts(base, 12)} [WARN ] Environment variable {var_name} not set, using fallback value ''
{_ts(base, 13)} [INFO ] Connecting to service... using config: {var_name}=''
{failing_tests.rstrip()}
{_ts(base, 17)} [ERROR] 3 failed, 12 passed in 4.21s
{_ts(base, 18)} [ERROR] Stage: test — FAILED (exit code 1)
{_ts(base, 19)} [INFO ] Stage: deploy — SKIPPED
{_ts(base, 19)} [ERROR] Pipeline FAILED
"""

    meta = {
        "task_id": 2,
        "failure_category": "config",
        "root_cause": f"missing environment variable: {var_name}",
        "expected_fix": f"set {var_name} in CI/CD secrets or .env file",
        "failed_stage": "test",
        "var_name": var_name,
        "failing_tests": test_names,
    }
    return log.strip(), meta
```

- [ ] **Step 4: Add Task 3 generator — async timing / flaky test**

```python
def _task3_log() -> tuple[str, dict]:
    """Hard: async timing failure that looks like a code bug but is flaky."""
    base = datetime.utcnow().replace(microsecond=0)
    repo = random.choice(_REPOS)
    branch = random.choice(_BRANCHES)
    run_id = random.randint(1000, 9999)
    timeout_ms = random.choice([50, 100, 150, 200])
    test_name = random.choice(
        ["test_async_handler", "test_concurrent_requests",
         "test_background_task", "test_websocket_ping"]
    )
    line = _rand_line()

    log = f"""##[group]Run details
Repository: {repo}
Branch:     {branch}
Run ID:     {run_id}
Triggered:  push
##[endgroup]

{_ts(base, 0)} [INFO ] Pipeline started
{_ts(base, 2)} [INFO ] Stage: checkout — OK
{_ts(base, 4)} [INFO ] Stage: setup-python — OK
{_ts(base, 6)} [INFO ] Stage: install-dependencies — OK
{_ts(base, 8)} [INFO ] Stage: lint — OK
{_ts(base, 10)} [INFO ] Stage: test — RUNNING
{_ts(base, 11)} [INFO ] pytest -x tests/ --timeout={timeout_ms / 1000:.1f}
{_ts(base, 13)} [ERROR] FAILED tests/test_handlers.py::{test_name}
{_ts(base, 13)} [ERROR]   File "tests/test_handlers.py", line {line}, in {test_name}
{_ts(base, 13)} [ERROR]     result = await asyncio.wait_for(handler(), timeout={timeout_ms / 1000:.2f})
{_ts(base, 13)} [ERROR] asyncio.exceptions.TimeoutError
{_ts(base, 14)} [ERROR] 1 failed, 27 passed in 2.{random.randint(10,99)}s
{_ts(base, 15)} [WARN ] Note: This test passed on the last 4 runs
{_ts(base, 15)} [ERROR] Stage: test — FAILED (exit code 1)
{_ts(base, 16)} [INFO ] Stage: deploy — SKIPPED
{_ts(base, 16)} [ERROR] Pipeline FAILED
"""

    meta = {
        "task_id": 3,
        "failure_category": "flaky",
        "root_cause": f"async timeout in {test_name} — timing-sensitive, not a logic bug",
        "expected_fix": "increase timeout or add retry logic for this test; do not modify handler code",
        "failed_stage": "test",
        "test_name": test_name,
        "timeout_ms": timeout_ms,
    }
    return log.strip(), meta
```

- [ ] **Step 5: Add the public `generate_log()` entry point**

```python
_GENERATORS = {1: _task1_log, 2: _task2_log, 3: _task3_log}

def generate_log(task_id: int | None = None) -> tuple[str, dict]:
    """
    Returns (log_text, metadata).
    metadata always has: task_id, failure_category, root_cause, expected_fix, failed_stage
    task_id=None picks randomly.
    """
    if task_id is None:
        task_id = random.choice([1, 2, 3])
    if task_id not in _GENERATORS:
        raise ValueError(f"task_id must be 1, 2, or 3 — got {task_id}")
    return _GENERATORS[task_id]()
```

- [ ] **Step 6: Quick smoke test in terminal**

```bash
cd "c:/Users/jayan/Desktop/New folder (2)"
python -c "
import sys; sys.path.insert(0, '.')
from cicd_diagnosis_env.server.log_generator import generate_log
for tid in [1,2,3]:
    log, meta = generate_log(tid)
    print(f'--- Task {tid} ---')
    print(log[:300])
    print(meta)
    print()
"
```
Expected: 3 different log blocks printed, each with a meta dict containing `failure_category`.

---

## Task 2: models.py

**Files:**
- Create: `cicd_diagnosis_env/models.py`
- Create: `cicd_diagnosis_env/__init__.py`

- [ ] **Step 1: Write models.py**

```python
# cicd_diagnosis_env/models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    from dataclasses import dataclass, field

    @dataclass(kw_only=True)
    class Action:
        pass

    @dataclass(kw_only=True)
    class Observation:
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass(kw_only=True)
    class State:
        episode_id: str = ""
        step_count: int = 0


class DiagnoseAction(Action):
    """Agent's diagnosis for a CI/CD pipeline failure."""
    failure_category: str        # "dependency" | "config" | "flaky" | "code_bug" | "infra"
    root_cause: str              # free-text explanation
    suggested_fix: str           # what the agent recommends
    confidence: float = 0.8      # 0.0–1.0, used for calibration score


class PipelineObservation(Observation):
    pipeline_log: str = ""
    error_summary: str = ""      # one-line extracted error
    pipeline_stage: str = ""     # which stage failed
    task_id: int = 0             # 1=easy, 2=medium, 3=hard
    attempt: int = 0
    feedback: str = ""
    score: float = 0.0


class PipelineState(State):
    last_score: float = 0.0
    task_id: int = 0
    pipeline_name: str = ""
```

- [ ] **Step 2: Write `__init__.py`**

```python
# cicd_diagnosis_env/__init__.py
from .models import DiagnoseAction, PipelineObservation, PipelineState
from .client import CICDEnv

__all__ = ["DiagnoseAction", "PipelineObservation", "PipelineState", "CICDEnv"]
```

- [ ] **Step 3: Create server `__init__.py`**

```python
# cicd_diagnosis_env/server/__init__.py
```
(empty, just makes it a package)

---

## Task 3: graders.py

**Files:**
- Create: `cicd_diagnosis_env/server/graders.py`

- [ ] **Step 1: Write graders.py with all three grader functions**

```python
# cicd_diagnosis_env/server/graders.py
"""
Grading logic for all three CI/CD diagnosis tasks.
Each grader returns (score: float, feedback: str).
Score is in [0.0, 1.0].
"""

from __future__ import annotations


def _category_score(predicted: str, expected: str) -> float:
    return 0.20 if predicted.lower().strip() == expected.lower().strip() else 0.0


def _cause_score(predicted: str, expected_keywords: list[str]) -> float:
    pred = predicted.lower()
    # at least one keyword must appear
    hit = any(kw.lower() in pred for kw in expected_keywords)
    return 0.30 if hit else 0.0


def _fix_score(predicted: str, expected_keywords: list[str]) -> float:
    pred = predicted.lower()
    hit = any(kw.lower() in pred for kw in expected_keywords)
    return 0.30 if hit else 0.0


def _confidence_score(predicted: float) -> float:
    # calibration: reward confident-and-correct answers
    # called only when base score > 0; penalise extreme miscalibration
    clamped = max(0.0, min(1.0, float(predicted)))
    # simple linear: 0.8 confidence = full 0.20, lower = proportional
    return round(clamped * 0.20, 4)


def _relevance_penalty(mentioned_sections: list[str], irrelevant: list[str]) -> float:
    hits = sum(1 for s in mentioned_sections if any(irr in s.lower() for irr in irrelevant))
    return -0.10 * hits


def grade_task1(action, meta: dict) -> tuple[float, str]:
    """Task 1: single ModuleNotFoundError — dependency failure."""
    pkg = meta["pkg"]
    score = 0.0
    parts = []

    cat = _category_score(action.failure_category, "dependency")
    score += cat
    parts.append(f"category={'OK' if cat else 'MISS'}")

    cause = _cause_score(action.root_cause, [pkg, "ModuleNotFoundError", "missing"])
    score += cause
    parts.append(f"root_cause={'OK' if cause else 'MISS'}")

    fix = _fix_score(action.suggested_fix, ["requirements", "pyproject", "install", pkg])
    score += fix
    parts.append(f"fix={'OK' if fix else 'MISS'}")

    if score > 0:
        conf = _confidence_score(action.confidence)
        score += conf
        parts.append(f"confidence={conf:.2f}")

    # penalise if agent mentions unrelated stages (checkout/lint passed fine)
    mentioned = action.root_cause.lower() + action.suggested_fix.lower()
    irrelevant_stages = ["checkout", "lint", "setup-python"]
    for stage in irrelevant_stages:
        if stage in mentioned:
            score -= 0.10
            parts.append(f"penalty(-0.10 for {stage})")

    score = max(0.0, min(1.0, round(score, 4)))
    feedback = f"Task1 grader: {', '.join(parts)}. Final={score}"
    return score, feedback


def grade_task2(action, meta: dict) -> tuple[float, str]:
    """Task 2: config env var causes 3 cascading test failures."""
    var = meta["var_name"]
    score = 0.0
    parts = []

    cat = _category_score(action.failure_category, "config")
    score += cat
    parts.append(f"category={'OK' if cat else 'MISS'}")

    # root cause must mention the var name or "environment variable" / "config"
    cause = _cause_score(action.root_cause, [var, "environment variable", "config", "missing var"])
    score += cause
    parts.append(f"root_cause={'OK' if cause else 'MISS'}")

    fix = _fix_score(action.suggested_fix, [var, "secret", ".env", "CI", "environment"])
    score += fix
    parts.append(f"fix={'OK' if fix else 'MISS'}")

    if score > 0:
        conf = _confidence_score(action.confidence)
        score += conf
        parts.append(f"confidence={conf:.2f}")

    # penalise if agent blames the test code (that's a symptom, not root cause)
    if "test" in action.root_cause.lower() and var.lower() not in action.root_cause.lower():
        score -= 0.10
        parts.append("penalty(-0.10 for blaming tests instead of config)")

    score = max(0.0, min(1.0, round(score, 4)))
    feedback = f"Task2 grader: {', '.join(parts)}. Final={score}"
    return score, feedback


def grade_task3(action, meta: dict) -> tuple[float, str]:
    """Task 3: async timing flaky test — must classify as flaky not code_bug."""
    test_name = meta["test_name"]
    score = 0.0
    parts = []

    cat = _category_score(action.failure_category, "flaky")
    score += cat
    parts.append(f"category={'OK' if cat else 'MISS'}")

    # root cause: must mention timing/timeout/async/intermittent — NOT logic bug
    cause = _cause_score(
        action.root_cause,
        ["timeout", "timing", "async", "intermittent", "flaky", "race"],
    )
    score += cause
    parts.append(f"root_cause={'OK' if cause else 'MISS'}")

    fix = _fix_score(
        action.suggested_fix,
        ["timeout", "retry", "increase", "skip", "xfail", "flaky marker"],
    )
    score += fix
    parts.append(f"fix={'OK' if fix else 'MISS'}")

    if score > 0:
        conf = _confidence_score(action.confidence)
        score += conf
        parts.append(f"confidence={conf:.2f}")

    # heavy penalise if agent says it's a code bug
    if "code_bug" in action.failure_category.lower() or (
        "logic" in action.root_cause.lower() and "timeout" not in action.root_cause.lower()
    ):
        score -= 0.20
        parts.append("penalty(-0.20 for misclassifying as code_bug)")

    score = max(0.0, min(1.0, round(score, 4)))
    feedback = f"Task3 grader: {', '.join(parts)}. Final={score}"
    return score, feedback


GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}


def grade(action, meta: dict) -> tuple[float, str]:
    """Dispatch to the right grader based on meta['task_id']."""
    tid = meta.get("task_id")
    if tid not in GRADERS:
        return 0.0, f"Unknown task_id {tid}"
    return GRADERS[tid](action, meta)
```

---

## Task 4: environment.py

**Files:**
- Create: `cicd_diagnosis_env/server/environment.py`

- [ ] **Step 1: Write environment.py**

```python
# cicd_diagnosis_env/server/environment.py
import uuid

try:
    from openenv.core.env_server.interfaces import Action, Environment, Observation
except ImportError:
    from cicd_diagnosis_env.models import Action, Observation

    class Environment:
        pass

from cicd_diagnosis_env.models import DiagnoseAction, PipelineObservation, PipelineState
from cicd_diagnosis_env.server.log_generator import generate_log
from cicd_diagnosis_env.server.graders import grade

MAX_STEPS = 3  # agent gets at most 3 attempts per episode


class CICDEnvironment(Environment):
    """
    RL environment for CI/CD failure diagnosis.

    Each episode: random pipeline log is shown, agent diagnoses up to 3 times.
    Episode ends when score=1.0 or max steps reached.
    """

    def __init__(self):
        self._state = PipelineState()
        self._meta = {}       # injected failure ground truth for current episode
        self._log = ""

    def reset(self) -> Observation:
        """Start a new episode with a fresh pipeline failure."""
        self._log, self._meta = generate_log()
        self._state = PipelineState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self._meta["task_id"],
            pipeline_name=self._meta.get("failed_stage", "unknown"),
        )

        # extract one-line error summary from log
        summary = _extract_summary(self._log)

        return PipelineObservation(
            pipeline_log=self._log,
            error_summary=summary,
            pipeline_stage=self._meta["failed_stage"],
            task_id=self._meta["task_id"],
            attempt=0,
            feedback="",
            score=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: Action) -> Observation:
        if not isinstance(action, DiagnoseAction):
            raise ValueError(f"Expected DiagnoseAction, got {type(action)}")

        self._state.step_count += 1
        step_num = self._state.step_count

        score, feedback = grade(action, self._meta)
        self._state.last_score = score

        done = score >= 1.0 or step_num >= MAX_STEPS

        return PipelineObservation(
            pipeline_log=self._log,
            error_summary=_extract_summary(self._log),
            pipeline_stage=self._meta["failed_stage"],
            task_id=self._meta["task_id"],
            attempt=step_num,
            feedback=feedback,
            score=score,
            done=done,
            reward=score,
        )

    @property
    def state(self) -> PipelineState:
        return self._state


def _extract_summary(log: str) -> str:
    """Pull the first ERROR line as a one-liner summary."""
    for line in log.splitlines():
        if "[ERROR]" in line and "Traceback" not in line:
            return line.split("[ERROR]")[-1].strip()
    return "Unknown error"
```

---

## Task 5: app.py

**Files:**
- Create: `cicd_diagnosis_env/server/app.py`

- [ ] **Step 1: Write app.py with manual FastAPI routes (no create_app helper)**

```python
# cicd_diagnosis_env/server/app.py
"""
FastAPI server for the CI/CD Failure Diagnosis environment.

Usage:
    uvicorn cicd_diagnosis_env.server.app:app --reload --host 0.0.0.0 --port 8000
    uvicorn cicd_diagnosis_env.server.app:app --host 0.0.0.0 --port 8000 --workers 2
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from cicd_diagnosis_env.models import DiagnoseAction, PipelineObservation
from cicd_diagnosis_env.server.environment import CICDEnvironment

app = FastAPI(title="CI/CD Diagnosis Environment", version="0.1.0")

# one shared env instance — fine for single-user / hackathon scale
# TODO: add per-session isolation if needed for multi-agent training
_env = CICDEnvironment()


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/reset")
def reset():
    obs = _env.reset()
    return _obs_to_dict(obs)


@app.post("/step")
def step(req: StepRequest):
    try:
        action = DiagnoseAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    obs = _env.step(action)
    return _obs_to_dict(obs)


@app.get("/state")
def get_state():
    s = _env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
        "last_score": s.last_score,
        "task_id": s.task_id,
        "pipeline_name": s.pipeline_name,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def _obs_to_dict(obs: PipelineObservation) -> dict:
    return {
        "observation": {
            "pipeline_log": obs.pipeline_log,
            "error_summary": obs.error_summary,
            "pipeline_stage": obs.pipeline_stage,
            "task_id": obs.task_id,
            "attempt": obs.attempt,
            "feedback": obs.feedback,
            "score": obs.score,
        },
        "reward": obs.reward,
        "done": obs.done,
        "info": {},
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the server locally**

```bash
cd "c:/Users/jayan/Desktop/New folder (2)"
# Start server in background
python -m uvicorn cicd_diagnosis_env.server.app:app --port 8000 &
sleep 2

# Test health
curl -s http://localhost:8000/health

# Test reset
curl -s -X POST http://localhost:8000/reset | python -m json.tool | head -20

# Test step
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"failure_category": "dependency", "root_cause": "missing package", "suggested_fix": "add to requirements.txt", "confidence": 0.9}}' \
  | python -m json.tool
```
Expected: `/health` → `{"status":"healthy"}`, `/reset` → JSON with `pipeline_log`, `/step` → JSON with `score` and `feedback`.

---

## Task 6: client.py

**Files:**
- Create: `cicd_diagnosis_env/client.py`

- [ ] **Step 1: Write client.py**

```python
# cicd_diagnosis_env/client.py
"""
CICDEnv
-------
HTTP client for the CI/CD Failure Diagnosis environment.

Instantiate with base_url pointing to a running server:
    env = CICDEnv(base_url="http://localhost:8000")
    obs = env.reset()
    result = env.step(DiagnoseAction(...))
"""

from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    _has_openenv = True
except ImportError:
    _has_openenv = False

from cicd_diagnosis_env.models import DiagnoseAction, PipelineObservation, PipelineState

if _has_openenv:
    class CICDEnv(EnvClient[DiagnoseAction, PipelineObservation, PipelineState]):
        # --- EnvClient hooks ---

        def _step_payload(self, action: DiagnoseAction) -> dict:
            # wire format expected by /step under "action" key
            return {
                "failure_category": action.failure_category,
                "root_cause": action.root_cause,
                "suggested_fix": action.suggested_fix,
                "confidence": action.confidence,
            }

        def _parse_result(self, payload: dict) -> StepResult[PipelineObservation]:
            # Expecting: { "observation": {...}, "reward": float|null, "done": bool }
            obs = PipelineObservation(**payload["observation"])
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=bool(payload.get("done", False)),
            )

        def _parse_state(self, payload: dict) -> PipelineState:
            return PipelineState(
                episode_id=payload.get("episode_id", ""),
                step_count=payload.get("step_count", 0),
                last_score=payload.get("last_score", 0.0),
                task_id=payload.get("task_id", 0),
                pipeline_name=payload.get("pipeline_name", ""),
            )
else:
    # fallback HTTP client when openenv SDK is not installed
    import requests

    class CICDEnv:  # type: ignore[no-redef]
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url.rstrip("/")

        def reset(self) -> PipelineObservation:
            r = requests.post(f"{self.base_url}/reset")
            r.raise_for_status()
            return PipelineObservation(**r.json()["observation"])

        def step(self, action: DiagnoseAction):
            payload = {
                "action": {
                    "failure_category": action.failure_category,
                    "root_cause": action.root_cause,
                    "suggested_fix": action.suggested_fix,
                    "confidence": action.confidence,
                }
            }
            r = requests.post(f"{self.base_url}/step", json=payload)
            r.raise_for_status()
            data = r.json()
            obs = PipelineObservation(**data["observation"])
            return obs
```

---

## Task 7: inference.py (ROOT level)

**Files:**
- Create: `inference.py` (at repo root, NOT inside cicd_diagnosis_env/)

- [ ] **Step 1: Write inference.py**

```python
# inference.py
"""
LLM agent for CI/CD Failure Diagnosis environment.

Required env vars:
    API_BASE_URL  — OpenAI-compatible API base (e.g. https://api.openai.com/v1)
    MODEL_NAME    — model to use (e.g. gpt-4o-mini)
    HF_TOKEN      — HuggingFace token (used if env runs on HF Spaces)
    ENV_URL       — base URL of the running cicd_diagnosis_env server
                    (default: http://localhost:8000)

Structured log format: [START], [STEP n], [END]
Runtime: well under 20 minutes for 10 episodes
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

client = OpenAI(api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""), base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an expert CI/CD engineer. You will be shown a pipeline failure log.
Diagnose the failure by providing:
1. failure_category: one of [dependency, config, flaky, code_bug, infra]
2. root_cause: concise explanation (1-2 sentences)
3. suggested_fix: concrete action to resolve it
4. confidence: float 0.0-1.0

Respond ONLY with valid JSON matching this schema:
{"failure_category": "...", "root_cause": "...", "suggested_fix": "...", "confidence": 0.9}"""


def diagnose(log: str, error_summary: str) -> DiagnoseAction:
    user_msg = f"Error summary: {error_summary}\n\nFull log:\n{log}"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content.strip()
    # strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)
    return DiagnoseAction(
        failure_category=parsed["failure_category"],
        root_cause=parsed["root_cause"],
        suggested_fix=parsed["suggested_fix"],
        confidence=float(parsed.get("confidence", 0.8)),
    )


def run_episode(env: CICDEnv, ep: int) -> float:
    obs = env.reset()
    print(f"[STEP {ep}.0] reset task_id={obs.task_id} stage={obs.pipeline_stage}")

    total_reward = 0.0
    for step_num in range(1, 4):  # max 3 attempts
        action = diagnose(obs.pipeline_log, obs.error_summary)
        obs = env.step(action)
        total_reward = obs.score
        print(
            f"[STEP {ep}.{step_num}] "
            f"cat={action.failure_category} score={obs.score:.3f} "
            f"feedback={obs.feedback[:80]}"
        )
        if obs.done:
            break

    return total_reward


def main():
    print(f"[START] cicd_diagnosis_env inference | model={MODEL_NAME} episodes={NUM_EPISODES}")
    env = CICDEnv(base_url=ENV_URL)

    scores = []
    start = time.time()

    for ep in range(1, NUM_EPISODES + 1):
        try:
            score = run_episode(env, ep)
            scores.append(score)
        except Exception as e:
            print(f"[STEP {ep}] ERROR: {e}", file=sys.stderr)
            scores.append(0.0)

    elapsed = time.time() - start
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"[END] episodes={NUM_EPISODES} avg_score={avg:.4f} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
```

---

## Task 8: Dockerfile

**Files:**
- Create: `cicd_diagnosis_env/Dockerfile`
- Create: `cicd_diagnosis_env/requirements.txt`

- [ ] **Step 1: Write requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
openai>=1.0.0
requests>=2.31.0
```

- [ ] **Step 2: Write Dockerfile**

```dockerfile
# cicd_diagnosis_env/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# install deps first (better layer caching)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# copy source
COPY . /app/cicd_diagnosis_env
COPY cicd_diagnosis_env/server /app/cicd_diagnosis_env/server

ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "cicd_diagnosis_env.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

- [ ] **Step 3: Test Docker build**

```bash
cd "c:/Users/jayan/Desktop/New folder (2)"
docker build -t cicd-diagnosis-env:latest -f cicd_diagnosis_env/Dockerfile .
```
Expected: build completes without errors.

- [ ] **Step 4: Test Docker run**

```bash
docker run -d -p 8000:8000 --name cicd-test cicd-diagnosis-env:latest
sleep 3
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/reset | python -m json.tool | head -10
docker stop cicd-test && docker rm cicd-test
```
Expected: health check passes, reset returns a log.

---

## Task 9: openenv.yaml + README

**Files:**
- Create: `cicd_diagnosis_env/openenv.yaml`
- Create: `cicd_diagnosis_env/README.md`

- [ ] **Step 1: Write openenv.yaml**

```yaml
spec_version: 1
name: cicd_diagnosis_env
version: "0.1.0"
description: "RL environment for diagnosing CI/CD pipeline failures"
type: space
runtime: fastapi
app: cicd_diagnosis_env.server.app:app
port: 8000
action: DiagnoseAction
observation: PipelineObservation
```

- [ ] **Step 2: Write README.md**

```markdown
# CI/CD Failure Diagnosis Environment

An OpenEnv RL environment where an agent diagnoses broken CI/CD pipelines.

## What the Agent Does

Given a synthetic pipeline log, the agent must:
1. Classify the failure category (`dependency`, `config`, `flaky`, `code_bug`, `infra`)
2. Identify the root cause
3. Suggest a concrete fix

## Three Task Tiers

| Task | Difficulty | Failure Type |
|------|-----------|-------------|
| 1 | Easy | Single `ModuleNotFoundError` — one missing package |
| 2 | Medium | Misconfigured env var → 3 cascading test failures |
| 3 | Hard | Async timing flaky test (looks like code bug) |

## Reward Function

| Component | Points |
|-----------|--------|
| Correct failure category | +0.20 |
| Correct root cause | +0.30 |
| Valid fix suggested | +0.30 |
| Confidence calibration | +0.20 |
| Per irrelevant section mentioned | -0.10 |

## Quick Start

```bash
# Run server
docker build -t cicd-env -f cicd_diagnosis_env/Dockerfile .
docker run -p 8000:8000 cicd-env

# Run LLM agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:8000
python inference.py
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode, returns initial observation |
| `/step` | POST | Submit diagnosis action, returns scored observation |
| `/state` | GET | Current episode metadata |
| `/health` | GET | Health check |
```
```

---

## Task 10: Final Validation

- [ ] **Step 1: Verify all files exist**

```bash
find "c:/Users/jayan/Desktop/New folder (2)/cicd_diagnosis_env" -type f | sort
ls "c:/Users/jayan/Desktop/New folder (2)/inference.py"
```

Expected files:
```
cicd_diagnosis_env/__init__.py
cicd_diagnosis_env/models.py
cicd_diagnosis_env/client.py
cicd_diagnosis_env/openenv.yaml
cicd_diagnosis_env/Dockerfile
cicd_diagnosis_env/README.md
cicd_diagnosis_env/requirements.txt
cicd_diagnosis_env/server/__init__.py
cicd_diagnosis_env/server/app.py
cicd_diagnosis_env/server/environment.py
cicd_diagnosis_env/server/log_generator.py
cicd_diagnosis_env/server/graders.py
inference.py
```

- [ ] **Step 2: Full integration test**

```bash
cd "c:/Users/jayan/Desktop/New folder (2)"
# start server
python -m uvicorn cicd_diagnosis_env.server.app:app --port 8000 &
sleep 2

# hit all endpoints
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/reset > /tmp/reset_out.json
cat /tmp/reset_out.json | python -m json.tool

curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"failure_category":"dependency","root_cause":"missing pytest-cov package","suggested_fix":"add pytest-cov to requirements.txt","confidence":0.9}}' \
  | python -m json.tool

curl -s http://localhost:8000/state | python -m json.tool
```

- [ ] **Step 3: Run openenv validate (if SDK installed)**

```bash
cd "c:/Users/jayan/Desktop/New folder (2)/cicd_diagnosis_env"
openenv validate || echo "SDK not installed — skip"
```

- [ ] **Step 4: Kill background server**

```bash
pkill -f "uvicorn cicd_diagnosis_env" || true
```
