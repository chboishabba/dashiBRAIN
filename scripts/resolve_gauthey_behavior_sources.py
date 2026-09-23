#!/usr/bin/env python3
"""Inspect the Gauthey preprocessed archive for behavior-source candidates.

Central-directory inspection only: no large archive member is downloaded.
Filename evidence is retained as candidate evidence and never promoted to exact
behavior artifact identity, trial binding, or synchronized timebase binding.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dashi.analysis.gauthey_lbm_experiment import GAUTHEY_DATA_ZIP_URL
from dashi.io.gauthey_behavior_source_resolution import (
    classify_behavior_archive_members,
)
from dashi.io.remote_zip import list_remote_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/gauthey_behavior_source_resolution.json",
    )
    parser.add_argument("--url", default=GAUTHEY_DATA_ZIP_URL)
    args = parser.parse_args()

    members = list_remote_zip(args.url)
    resolution = classify_behavior_archive_members(members)
    payload = {
        "archive_url": args.url,
        "archive_member_count": len(members),
        "strong_candidates": [asdict(candidate) for candidate in resolution.strong_candidates],
        "weak_candidates": [asdict(candidate) for candidate in resolution.weak_candidates],
        "exact_behavior_artifact_identity_paid": resolution.exact_behavior_artifact_identity_paid,
        "exact_trial_binding_paid": resolution.exact_trial_binding_paid,
        "synchronized_timebase_binding_paid": resolution.synchronized_timebase_binding_paid,
        "source_bounded_context": {
            "paper_reports_behavior_recording": True,
            "paper_reports_ball_tracking": True,
            "paper_reports_fictrac_locomotion": True,
            "paper_reports_stimulus_behavior_synchronization": True,
            "paper_doi": "10.1038/s41467-026-72437-1",
        },
        "boundary": (
            "central-directory filename evidence only; a candidate member does not establish "
            "same-trial identity or neural/behavior timebase binding"
        ),
    }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
