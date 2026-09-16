from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "retry_gauthey_lbm_remaining_identities.py"
SPEC = importlib.util.spec_from_file_location("retry_remaining", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
retry_remaining = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retry_remaining)


def _write_interruption(root: Path, trial: str, *, resumable: bool = True, marker: int = 0) -> None:
    path = (
        root
        / "trial_recovery"
        / trial
        / "trial_receipts"
        / f"gauthey_lbm_transport_interruption_{trial}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "gauthey_lbm_transport_interruption",
                "trial_id": trial,
                "resumable": resumable,
                "searched": False,
                "marker": marker,
            }
        ),
        encoding="utf-8",
    )


def test_fresh_resumable_transport_receipt_retries_until_success(tmp_path, monkeypatch):
    returns = iter((1, 1, 0))
    calls: list[tuple[str, ...]] = []
    sleeps: list[float] = []
    attempt = {"n": 0}

    def fake_run(cmd, check=False):
        attempt["n"] += 1
        calls.append(tuple(cmd))
        if attempt["n"] <= 2:
            _write_interruption(
                tmp_path,
                "04192024_6f_a1_r2",
                resumable=True,
                marker=attempt["n"],
            )
        return subprocess.CompletedProcess(cmd, next(returns))

    monkeypatch.setattr(retry_remaining.subprocess, "run", fake_run)
    monkeypatch.setattr(retry_remaining.time, "sleep", sleeps.append)

    completed = retry_remaining.run_with_resumable_transport_retries(
        ["python", "recover.py"],
        output_dir=tmp_path,
        max_attempts=4,
        retry_base_seconds=2.0,
        retry_cap_seconds=10.0,
    )

    assert completed.returncode == 0
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]


def test_nontransport_failure_is_not_retried(tmp_path, monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd, check=False):
        calls.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 2)

    monkeypatch.setattr(retry_remaining.subprocess, "run", fake_run)
    monkeypatch.setattr(
        retry_remaining.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )

    completed = retry_remaining.run_with_resumable_transport_retries(
        ["python", "recover.py"],
        output_dir=tmp_path,
        max_attempts=4,
        retry_base_seconds=2.0,
        retry_cap_seconds=10.0,
    )

    assert completed.returncode == 2
    assert len(calls) == 1


def test_nonresumable_transport_receipt_is_not_retried(tmp_path, monkeypatch):
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd, check=False):
        calls.append(tuple(cmd))
        _write_interruption(tmp_path, "04192024_6f_a1_r2", resumable=False, marker=1)
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(retry_remaining.subprocess, "run", fake_run)
    monkeypatch.setattr(
        retry_remaining.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )

    completed = retry_remaining.run_with_resumable_transport_retries(
        ["python", "recover.py"],
        output_dir=tmp_path,
        max_attempts=4,
        retry_base_seconds=2.0,
        retry_cap_seconds=10.0,
    )

    assert completed.returncode == 1
    assert len(calls) == 1
