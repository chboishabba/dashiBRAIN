"""Fail-closed receipts for completed Gauthey LBM reconstructions.

A successful remote stream proves the requested ZIP member passed compressed-size,
uncompressed-size, and CRC-32 checks.  This module adds the persistent scientific
receipt needed after the temporary inner ZIP has been deleted: it binds the exact
Zenodo member metadata to the deposited selected matrix and the generated
identity/unresolved/summary outputs with SHA-256 digests.

The receipt is deliberately narrower than an empirical result.  It certifies that
an execution fixture is structurally/provenance ready for downstream consumers;
it does not promote source-trace identity to neuron identity or establish a
structure/function claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from dashi.io.remote_zip import RemoteZipMember


@dataclass(frozen=True)
class FileDigest:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GautheyReconstructionReceipt:
    trial_id: str
    remote_url: str
    remote_member: Mapping[str, object]
    deposited_selected: FileDigest
    reconstruction_summary: FileDigest
    identities: FileDigest
    unresolved_rows: FileDigest
    processed_trial_confirmed: bool
    exact_source_identities_recovered: int
    unresolved_selected_rows: int
    complete_identity_recovery: bool
    identity_semantics: str
    fixture_admissible: bool


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 << 20) -> FileDigest:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = sha256()
    size = 0
    with p.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return FileDigest(str(p), size, digest.hexdigest())


def build_gauthey_reconstruction_receipt(
    *,
    trial_id: str,
    remote_url: str,
    remote_member: RemoteZipMember,
    deposited_selected_path: str | Path,
    reconstruction_summary_path: str | Path,
) -> GautheyReconstructionReceipt:
    summary_path = Path(reconstruction_summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    processed = tuple(str(x) for x in summary.get("processed_trials", ()))
    if trial_id not in processed:
        raise ValueError(
            f"trial {trial_id!r} is not confirmed in reconstruction processed_trials={processed!r}"
        )

    identity_semantics = str(summary.get("identity_semantics", ""))
    if identity_semantics != "exact trace equality to deposited selected row; not neuron identity":
        raise ValueError(f"unexpected identity semantics: {identity_semantics!r}")

    identity_path = Path(str(summary.get("identity_output", "")))
    unresolved_path = Path(str(summary.get("unresolved_output", "")))
    if not identity_path.is_file():
        # Older summaries commonly store a relative path whose intended base is
        # the process working directory.  If that is unavailable, try the
        # summary directory using only the emitted filename before failing.
        candidate = summary_path.parent / identity_path.name
        if candidate.is_file():
            identity_path = candidate
        else:
            raise FileNotFoundError(identity_path)
    if not unresolved_path.is_file():
        candidate = summary_path.parent / unresolved_path.name
        if candidate.is_file():
            unresolved_path = candidate
        else:
            raise FileNotFoundError(unresolved_path)

    recovered = int(summary.get("exact_source_identities_recovered", -1))
    unresolved = int(summary.get("unresolved_selected_rows", -1))
    complete = bool(summary.get("complete_identity_recovery", False))
    if recovered < 0 or unresolved < 0:
        raise ValueError("reconstruction summary lacks non-negative identity counts")
    if complete != (unresolved == 0):
        raise ValueError("complete_identity_recovery disagrees with unresolved_selected_rows")

    # A partial exact recovery is still a valid fixture for consumers that retain
    # the unresolved rows explicitly.  Admissibility therefore means the exact
    # trial/provenance binding and outputs are verified, not that every selected
    # row has been recovered.
    fixture_admissible = recovered > 0

    return GautheyReconstructionReceipt(
        trial_id=trial_id,
        remote_url=remote_url,
        remote_member={
            "name": remote_member.name,
            "compressed_size": remote_member.compressed_size,
            "uncompressed_size": remote_member.uncompressed_size,
            "compression_method": remote_member.compression_method,
            "local_header_offset": remote_member.local_header_offset,
            "crc32": f"{remote_member.crc32:08x}",
        },
        deposited_selected=sha256_file(deposited_selected_path),
        reconstruction_summary=sha256_file(summary_path),
        identities=sha256_file(identity_path),
        unresolved_rows=sha256_file(unresolved_path),
        processed_trial_confirmed=True,
        exact_source_identities_recovered=recovered,
        unresolved_selected_rows=unresolved,
        complete_identity_recovery=complete,
        identity_semantics=identity_semantics,
        fixture_admissible=fixture_admissible,
    )


def write_gauthey_reconstruction_receipt(
    receipt: GautheyReconstructionReceipt, output_path: str | Path
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(receipt), indent=2) + "\n", encoding="utf-8")
    return out
