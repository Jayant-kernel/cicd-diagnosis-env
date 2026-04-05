# cicd_diagnosis_env/server/log_generator.py
import random
from datetime import datetime, timedelta

# packages that show up as missing deps in task1
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


def _ts(base, offset_s):
    return (base + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rand_line():
    return random.randint(12, 340)


def _task1_log():
    # easy - one obvious ModuleNotFoundError, single stage fails
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
{_ts(base, 2)} [INFO ] Stage: checkout - OK
{_ts(base, 4)} [INFO ] Stage: setup-python - OK
{_ts(base, 6)} [INFO ] Stage: install-dependencies - OK
{_ts(base, 8)} [INFO ] Stage: lint - OK
{_ts(base, 10)} [INFO ] Stage: test - RUNNING
{_ts(base, 11)} [ERROR] Traceback (most recent call last):
{_ts(base, 11)} [ERROR]   File "tests/test_main.py", line {line}, in test_endpoint
{_ts(base, 11)} [ERROR]     import {pkg}
{_ts(base, 11)} [ERROR] ModuleNotFoundError: No module named '{pkg}'
{_ts(base, 12)} [ERROR] Stage: test - FAILED (exit code 1)
{_ts(base, 13)} [INFO ] Stage: deploy - SKIPPED
{_ts(base, 13)} [ERROR] Pipeline FAILED"""

    meta = {
        "task_id": 1,
        "failure_category": "dependency",
        "root_cause": f"missing package: {pkg}",
        "expected_fix": f"add {pkg} to requirements.txt or pyproject.toml",
        "failed_stage": "test",
        "pkg": pkg,
    }
    return log, meta


def _task2_log():
    # medium - missing env var causes 3 downstream test failures
    # the tricky part: symptoms look like test failures but root cause is config
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
        failing_tests += (
            f"{_ts(base, 14 + i)} [ERROR] FAILED tests/test_api.py::{t}\n"
            f"{_ts(base, 14 + i)} [ERROR]   File \"tests/test_api.py\", line {ln}, in {t}\n"
            f"{_ts(base, 14 + i)} [ERROR]     assert response.status_code == 200\n"
            f"{_ts(base, 14 + i)} [ERROR] AssertionError: assert 500 == 200\n"
        )

    log = f"""##[group]Run details
Repository: {repo}
Branch:     {branch}
Run ID:     {run_id}
Triggered:  push
##[endgroup]

{_ts(base, 0)} [INFO ] Pipeline started
{_ts(base, 2)} [INFO ] Stage: checkout - OK
{_ts(base, 4)} [INFO ] Stage: setup-python - OK
{_ts(base, 6)} [INFO ] Stage: install-dependencies - OK
{_ts(base, 8)} [INFO ] Stage: lint - OK
{_ts(base, 10)} [INFO ] Stage: test - RUNNING
{_ts(base, 12)} [WARN ] Environment variable {var_name} not set, using fallback value ''
{_ts(base, 13)} [INFO ] Connecting to service... using config: {var_name}=''
{failing_tests.rstrip()}
{_ts(base, 17)} [ERROR] 3 failed, 12 passed in 4.21s
{_ts(base, 18)} [ERROR] Stage: test - FAILED (exit code 1)
{_ts(base, 19)} [INFO ] Stage: deploy - SKIPPED
{_ts(base, 19)} [ERROR] Pipeline FAILED"""

    meta = {
        "task_id": 2,
        "failure_category": "config",
        "root_cause": f"missing environment variable: {var_name}",
        "expected_fix": f"set {var_name} in CI/CD secrets or .env file",
        "failed_stage": "test",
        "var_name": var_name,
        "failing_tests": test_names,
    }
    return log, meta


def _task3_log():
    # hard - async timeout that *looks* like a code bug but it's a flaky timing issue
    # the hint is "This test passed on the last 4 runs" buried in the log
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
    elapsed_ms = random.randint(10, 99)  # for the "2.XX s" runtime line

    log = f"""##[group]Run details
Repository: {repo}
Branch:     {branch}
Run ID:     {run_id}
Triggered:  push
##[endgroup]

{_ts(base, 0)} [INFO ] Pipeline started
{_ts(base, 2)} [INFO ] Stage: checkout - OK
{_ts(base, 4)} [INFO ] Stage: setup-python - OK
{_ts(base, 6)} [INFO ] Stage: install-dependencies - OK
{_ts(base, 8)} [INFO ] Stage: lint - OK
{_ts(base, 10)} [INFO ] Stage: test - RUNNING
{_ts(base, 11)} [INFO ] pytest -x tests/ --timeout={timeout_ms / 1000:.1f}
{_ts(base, 13)} [ERROR] FAILED tests/test_handlers.py::{test_name}
{_ts(base, 13)} [ERROR]   File "tests/test_handlers.py", line {line}, in {test_name}
{_ts(base, 13)} [ERROR]     result = await asyncio.wait_for(handler(), timeout={timeout_ms / 1000:.2f})
{_ts(base, 13)} [ERROR] asyncio.exceptions.TimeoutError
{_ts(base, 14)} [ERROR] 1 failed, 27 passed in 2.{elapsed_ms}s
{_ts(base, 15)} [WARN ] Note: This test passed on the last 4 runs
{_ts(base, 15)} [ERROR] Stage: test - FAILED (exit code 1)
{_ts(base, 16)} [INFO ] Stage: deploy - SKIPPED
{_ts(base, 16)} [ERROR] Pipeline FAILED"""

    meta = {
        "task_id": 3,
        "failure_category": "flaky",
        "root_cause": f"async timeout in {test_name} - timing-sensitive, not a logic bug",
        "expected_fix": "increase timeout or add retry logic for this test; do not modify handler code",
        "failed_stage": "test",
        "test_name": test_name,
        "timeout_ms": timeout_ms,
    }
    return log, meta


_GENERATORS = {1: _task1_log, 2: _task2_log, 3: _task3_log}


def generate_log(task_id=None):
    # task_id=None -> pick randomly, useful during training
    if task_id is None:
        task_id = random.choice([1, 2, 3])
    if task_id not in _GENERATORS:
        raise ValueError(f"task_id must be 1, 2, or 3 - got {task_id}")
    return _GENERATORS[task_id]()
