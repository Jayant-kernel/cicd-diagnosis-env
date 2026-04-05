# cicd_diagnosis_env/server/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from cicd_diagnosis_env.models import DiagnoseAction
from cicd_diagnosis_env.server.environment import CICDEnvironment

app = FastAPI(title="cicd_diagnosis_env", version="0.1.0")

# one shared env instance — fine for hackathon scale
# TODO: add per-session map if we need concurrent multi-agent training
_env = CICDEnvironment()


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/reset")
def reset():
    obs = _env.reset()
    return _obs_dict(obs)


@app.post("/step")
def step(req: StepRequest):
    try:
        action = DiagnoseAction(**req.action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    obs = _env.step(action)
    return _obs_dict(obs)


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


def _obs_dict(obs):
    # separate observation from reward/done so clients match the OpenEnv wire format
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
