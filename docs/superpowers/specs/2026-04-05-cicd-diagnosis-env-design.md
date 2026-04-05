# CI/CD Failure Diagnosis Environment — Design Spec

**Team:** Technical Irony
**Hackathon:** Meta PyTorch x Scaler OpenEnv
**Deadline:** April 8, 2026 11:59 PM

---

## What We're Building

An OpenEnv-compliant RL environment where an agent reads synthetic CI/CD
pipeline failure logs and must diagnose: failure category, root cause, and
suggested fix. Three difficulty tiers. Dense reward function. 100%
deterministic grading via injected failure metadata.

---

## Hybrid Log Generation (Approved Approach)

- 3 "seed" failure types (easy / medium / hard), each with a hardcoded template
- Each `reset()` randomizes: package names, line numbers, repo names, branch names, 
  timestamp offsets within the template
- Grader always knows the injected failure (passed via internal state), so grading
  is deterministic regardless of cosmetic variation

---

## Three Tasks

| Task | Difficulty | Failure Type | Key Signal | Max Reward |
|------|-----------|--------------|------------|-----------|
| 1 | Easy | Single `ModuleNotFoundError` | One package, one stage | 1.0 |
| 2 | Medium | Config error → 3 cascading test failures | Root cause is config, not tests | 1.0 |
| 3 | Hard | Async timing flaky failure | Must classify "flaky" not "code_bug" | 1.0 |

---

## Reward Function

| Component | Points |
|-----------|--------|
| Correct failure category | +0.20 |
| Correct root cause identified | +0.30 |
| Valid fix suggested | +0.30 |
| Confidence calibration (0.0–1.0, scored by proximity) | +0.20 |
| Per irrelevant log section mentioned (penalty) | -0.10 |

---

## File Structure

```
cicd_diagnosis_env/          ← env package root
├── __init__.py
├── models.py                ← Action, Observation, State (Pydantic dataclasses)
├── client.py                ← CICDEnv(EnvClient) subclass
├── openenv.yaml             ← manifest
├── Dockerfile               ← server container
├── README.md
└── server/
    ├── __init__.py
    ├── environment.py       ← CICDEnvironment(Environment) — reset/step/state
    ├── app.py               ← FastAPI routes (manual, not scaffolded)
    ├── log_generator.py     ← hybrid template + randomization
    └── graders.py           ← grader functions for all 3 tasks
inference.py                 ← ROOT level, LLM agent
```

---

## Judging Alignment

- **Real-world utility (30%):** CI/CD diagnosis is a real pain point; dense reward encourages partial-credit learning
- **Task & grader quality (25%):** 3 difficulty tiers, deterministic grading, calibrated reward
- **Environment design (20%):** Clean OpenEnv spec compliance, async client, Docker-ready
- **Code quality (15%):** Human-written style, under 30 lines/function, sparse comments
- **Creativity (10%):** Hybrid log generation, confidence calibration penalty, flaky-test hard task

---

## Key Constraints

- Runtime under 20 minutes
- 2 vCPU / 8GB RAM
- `openenv validate` must pass
- Docker build + run must work
- HF Space deploys and responds to `reset()`
- `inference.py` uses `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars
- Structured logs: `[START]`, `[STEP]`, `[END]` format
