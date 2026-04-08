# cicd_diagnosis_env/server/environment.py
import uuid

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:
        pass

from cicd_diagnosis_env.models import DiagnoseAction, PipelineObservation, PipelineState
from cicd_diagnosis_env.server.log_generator import generate_log
from cicd_diagnosis_env.server.graders import grade

MAX_STEPS = 3  # agent gets up to 3 attempts per episode


class CICDEnvironment(Environment):
    """
    CI/CD failure diagnosis environment.
    Agent sees a broken pipeline log and must identify category, root cause, and fix.
    Episode ends on perfect score or after MAX_STEPS attempts.
    """

    def __init__(self):
        self._state = PipelineState()
        self._meta = {}
        self._log = ""

    def reset(self):
        self._log, self._meta = generate_log()
        self._state = PipelineState(episode_id=str(uuid.uuid4()), step_count=0)
        self._state.task_id = self._meta["task_id"]
        self._state.pipeline_name = self._meta.get("failed_stage", "unknown")
        self._state.last_score = 0.01
        summary = _extract_summary(self._log)
        obs = PipelineObservation(done=False, reward=0.01)
        obs.pipeline_log = self._log
        obs.error_summary = summary
        obs.pipeline_stage = self._meta["failed_stage"]
        obs.task_id = self._meta["task_id"]
        obs.attempt = 0
        obs.feedback = ""
        obs.score = 0.01
        return obs

    def step(self, action):
        if not isinstance(action, DiagnoseAction):
            raise ValueError(f"expected DiagnoseAction, got {type(action)}")

        self._state.step_count += 1
        step_num = self._state.step_count

        score, feedback = grade(action, self._meta)
        self._state.last_score = score

        # TODO: track per-episode score history for better feedback
        done = score >= 0.99 or step_num >= MAX_STEPS

        obs = PipelineObservation(done=done, reward=score)
        obs.pipeline_log = self._log
        obs.error_summary = _extract_summary(self._log)
        obs.pipeline_stage = self._meta["failed_stage"]
        obs.task_id = self._meta["task_id"]
        obs.attempt = step_num
        obs.feedback = feedback
        obs.score = score
        return obs

    @property
    def state(self):
        return self._state


def _extract_summary(log):
    # grab the actual error class line — skip Traceback/File/assert noise
    for line in log.splitlines():
        if "[ERROR]" not in line:
            continue
        tail = line.split("[ERROR]")[-1].strip()
        # skip frame lines and assertion lines — not useful as a summary
        if tail.startswith('File "') or tail.startswith("assert ") or tail.startswith("Traceback"):
            continue
        if "FAILED" in tail or "failed" in tail:
            continue
        return tail
    return "unknown error"
