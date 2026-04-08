# cicd_diagnosis_env/client.py
# HTTP client wrapper — tries openenv SDK first, falls back to raw requests
from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    _sdk = True
except ImportError:
    _sdk = False

from cicd_diagnosis_env.models import DiagnoseAction, PipelineObservation, PipelineState

if _sdk:
    class CICDEnv(EnvClient[DiagnoseAction, PipelineObservation, PipelineState]):
        # --- EnvClient abstract hooks ---

        def _step_payload(self, action: DiagnoseAction) -> dict:
            # wire format expected by /step endpoint under "action" key
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
    # fallback when openenv SDK is not installed — plain requests, sync only
    import requests

    class CICDEnv:  # type: ignore[no-redef]
        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")

        def reset(self) -> PipelineObservation:
            r = requests.post(f"{self.base_url}/reset", timeout=30)
            r.raise_for_status()
            return PipelineObservation(**r.json()["observation"])

        def step(self, action: DiagnoseAction) -> PipelineObservation:
            payload = {
                "action": {
                    "failure_category": action.failure_category,
                    "root_cause": action.root_cause,
                    "suggested_fix": action.suggested_fix,
                    "confidence": action.confidence,
                }
            }
            r = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
            r.raise_for_status()
            return PipelineObservation(**r.json()["observation"])
