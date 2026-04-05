# cicd_diagnosis_env/server/graders.py
from __future__ import annotations


def _cat_score(predicted, expected):
    return 0.20 if predicted.lower().strip() == expected.lower() else 0.0


def _cause_score(predicted, keywords):
    pred = predicted.lower()
    # any keyword hit counts — agents phrase things differently
    return 0.30 if any(kw.lower() in pred for kw in keywords) else 0.0


def _fix_score(predicted, keywords):
    pred = predicted.lower()
    return 0.30 if any(kw.lower() in pred for kw in keywords) else 0.0


def _conf_score(conf):
    # scale 0.20 bonus by confidence — only called when base score > 0
    # real calibration would need a held-out set, this is good enough for now
    return round(max(0.0, min(1.0, float(conf))) * 0.20, 4)


def grade_task1(action, meta):
    # easy task: single ModuleNotFoundError, category must be "dependency"
    pkg = meta["pkg"]
    score = 0.0
    parts = []

    cat = _cat_score(action.failure_category, "dependency")
    score += cat
    parts.append(f"cat={'OK' if cat else 'MISS'}")

    cause = _cause_score(action.root_cause, [pkg, "ModuleNotFoundError", "missing"])
    score += cause
    parts.append(f"cause={'OK' if cause else 'MISS'}")

    fix = _fix_score(action.suggested_fix, ["requirements", "pyproject", "install", pkg])
    score += fix
    parts.append(f"fix={'OK' if fix else 'MISS'}")

    if score > 0:
        c = _conf_score(action.confidence)
        score += c
        parts.append(f"conf={c:.2f}")

    # penalise mentioning stages that passed fine
    for stage in ["checkout", "lint", "setup-python"]:
        if stage in (action.root_cause + action.suggested_fix).lower():
            score -= 0.10
            parts.append(f"penalty(irrelevant:{stage})")

    score = max(0.0, min(1.0, round(score, 4)))
    return score, f"task1: {', '.join(parts)} => {score}"


def grade_task2(action, meta):
    # medium: config env var missing -> 3 cascading failures; root cause is NOT the test code
    var = meta["var_name"]
    score = 0.0
    parts = []

    cat = _cat_score(action.failure_category, "config")
    score += cat
    parts.append(f"cat={'OK' if cat else 'MISS'}")

    cause = _cause_score(action.root_cause, [var, "environment variable", "config", "missing var"])
    score += cause
    parts.append(f"cause={'OK' if cause else 'MISS'}")

    fix = _fix_score(action.suggested_fix, [var, "secret", ".env", "CI", "environment"])
    score += fix
    parts.append(f"fix={'OK' if fix else 'MISS'}")

    if score > 0:
        c = _conf_score(action.confidence)
        score += c
        parts.append(f"conf={c:.2f}")

    # blaming the test code is wrong — that's a symptom
    if "test" in action.root_cause.lower() and var.lower() not in action.root_cause.lower():
        score -= 0.10
        parts.append("penalty(blamed tests not config)")

    score = max(0.0, min(1.0, round(score, 4)))
    return score, f"task2: {', '.join(parts)} => {score}"


def grade_task3(action, meta):
    # hard: async timeout flaky test — MUST classify as "flaky", not "code_bug"
    score = 0.0
    parts = []

    cat = _cat_score(action.failure_category, "flaky")
    score += cat
    parts.append(f"cat={'OK' if cat else 'MISS'}")

    cause = _cause_score(
        action.root_cause,
        ["timeout", "timing", "async", "intermittent", "flaky", "race"],
    )
    score += cause
    parts.append(f"cause={'OK' if cause else 'MISS'}")

    fix = _fix_score(
        action.suggested_fix,
        ["timeout", "retry", "increase", "skip", "xfail", "flaky marker"],
    )
    score += fix
    parts.append(f"fix={'OK' if fix else 'MISS'}")

    if score > 0:
        c = _conf_score(action.confidence)
        score += c
        parts.append(f"conf={c:.2f}")

    # heavy penalty for misclassifying as code_bug — that's the whole point of this task
    if "code_bug" in action.failure_category.lower() or (
        "logic" in action.root_cause.lower() and "timeout" not in action.root_cause.lower()
    ):
        score -= 0.20
        parts.append("penalty(misclassified as code_bug)")

    score = max(0.0, min(1.0, round(score, 4)))
    return score, f"task3: {', '.join(parts)} => {score}"


_GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}


def grade(action, meta):
    tid = meta.get("task_id")
    if tid not in _GRADERS:
        return 0.0, f"unknown task_id {tid}"
    return _GRADERS[tid](action, meta)
