from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_gauthey_lbm_remaining_identities.py"
SPEC = importlib.util.spec_from_file_location("recover_remaining", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
recover_remaining = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recover_remaining)


def test_transient_exit_is_retried_until_trial_completes(monkeypatch):
    returns = iter((75, 75, 0))
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    def fake_run(cmd, check=False):
        calls.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, next(returns))

    monkeypatch.setattr(recover_remaining.subprocess, "run", fake_run)
    monkeypatch.setattr(recover_remaining.time, "sleep", sleeps.append)

    completed = recover_remaining._run_trial_subprocess_with_transient_resume(
        ["python", "trial.py"],
        max_resume_attempts=4,
        retry_base_seconds=2.0,
        retry_cap_seconds=10.0,
    )

    assert completed.returncode == 0
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_nontransient_exit_is_not_retried(monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd, check=False):
        calls.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 2)

    monkeypatch.setattr(recover_remaining.subprocess, "run", fake_run)
    monkeypatch.setattr(recover_remaining.time, "sleep", lambda _: (_ for _ in ()).throw(AssertionError("unexpected sleep")))

    completed = recover_remaining._run_trial_subprocess_with_transient_resume(
        ["python", "trial.py"],
        max_resume_attempts=4,
        retry_base_seconds=2.0,
        retry_cap_seconds=10.0,
    )

    assert completed.returncode == 2
    assert len(calls) == 1


def test_transient_retry_budget_is_bounded(monkeypatch):
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    def fake_run(cmd, check=False):
        calls.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 75)

    monkeypatch.setattr(recover_remaining.subprocess, "run", fake_run)
    monkeypatch.setattr(recover_remaining.time, "sleep", sleeps.append)

    completed = recover_remaining._run_trial_subprocess_with_transient_resume(
        ["python", "trial.py"],
        max_resume_attempts=3,
        retry_base_seconds=3.0,
        retry_cap_seconds=5.0,
    )

    assert completed.returncode == 75
    assert len(calls) == 3
    assert sleeps == [3.0, 5.0]
