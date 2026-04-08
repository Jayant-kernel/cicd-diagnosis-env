---
title: CICD Diagnosis Env
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# CI/CD Failure Diagnosis Environment

An OpenEnv RL environment where an agent reads broken CI/CD pipeline logs and diagnoses what went wrong.

## What the Agent Does

Given a synthetic pipeline failure log, the agent must:
1. Classify the failure category (`dependency`, `config`, `flaky`, `code_bug`, `infra`)
2. Identify the root cause
3. Suggest a concrete fix

## Three Task Tiers

| Task | Difficulty | Failure Type |
|------|-----------|-------------|
| 1 | Easy | Single `ModuleNotFoundError` — one missing package |
| 2 | Medium | Misconfigured env var causes 3 cascading test failures |
| 3 | Hard | Async timing flaky test (looks like a code bug, isn't) |

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
# Build and run the server
docker build -t cicd-env .
docker run -p 7860:7860 cicd-env

# Run the LLM agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:7860
python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode, returns initial observation with pipeline log |
| `/step` | POST | Submit `DiagnoseAction`, returns scored observation |
| `/state` | GET | Current episode metadata |
| `/health` | GET | Health check |

## Action Schema

```json
{
  "action": {
    "failure_category": "dependency",
    "root_cause": "missing pytest-cov package",
    "suggested_fix": "add pytest-cov to requirements.txt",
    "confidence": 0.9
  }
}
```
