from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from openenv.core.env_server.interfaces import Action, Observation, State
    from dataclasses import dataclass, field
except ImportError:
    # running outside the openenv SDK (local dev, tests) — define minimal base classes
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


@dataclass(kw_only=True)
class DiagnoseAction(Action):
    # one of: dependency, config, flaky, code_bug, infra
    failure_category: str
    root_cause: str
    suggested_fix: str
    # 0-1 self-reported confidence; grader scales the 0.20 bonus by this
    confidence: float = 0.8


@dataclass(kw_only=True)
class PipelineObservation(Observation):
    pipeline_log: str = ""
    error_summary: str = ""   # one-liner pulled from the log
    pipeline_stage: str = ""
    task_id: int = 0          # 1=easy 2=medium 3=hard
    attempt: int = 0
    feedback: str = ""
    score: float = 0.01
    # TODO: add structured fields for log sections once we have more task types


@dataclass(kw_only=True)
class PipelineState(State):
    last_score: float = 0.01
    task_id: int = 0
    pipeline_name: str = ""
