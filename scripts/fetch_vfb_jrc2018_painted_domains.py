#!/usr/bin/env python3
"""Fetch the 46 VFB-painted JRC2018Unisex neuropil domains as NRRDs.

Source dataset: Virtual Fly Brain JRC2018.  The HTML report for this dataset is
only a presentation view and can expose a truncated subset, so this executable
uses VFB's canonical API through ``vfb_connect`` to enumerate dataset instances,
filters those whose template list contains JRC2018Unisex, and then downloads the
corresponding VFB image volumes.

Each painted domain remains a separate binary image.  We deliberately do not
collapse overlapping domains/subdomains into a mutually-exclusive label atlas.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import urllib.request

DATASET_ID = "JRC2018"
TEMPLATE_NAME = "JRC2018Unisex"
TEMPLATE_ID = "VFB_00101567"
EXPECTED_DOMAIN_COUNT = 46
VFB_IMAGE_ROOT = "https://www.virtualflybrain.org/data/VFB/i"


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._'-]+", "_", text.strip())
    return value.strip("_") or "region"


def _vfb_image_url(vfb_id: str) -> str:
    if not re.fullmatch(r"VFB_\d{8}", vfb_id):
        raise ValueError(f"unexpected VFB identifier: {vfb_id!r}")
    digits = vfb_id.split("_", 1)[1]
    return f"{VFB_IMAGE_ROOT}/{digits[:4]}/{digits[4:]}/{TEMPLATE_ID}/volume.nrrd"


def _canonical_dataset_rows() -> list[dict]:
    try:
        from vfb_connect.cross_server_tools import VfbConnect
    except ImportError as exc:
        raise RuntimeError(
            "vfb_connect is required for canonical JRC2018 domain discovery; "
            "install it with `pip install vfb-connect`"
        ) from exc

    vc = VfbConnect()
    frame = vc.get_instances_by_dataset(DATASET_ID)
    if frame is None or len(frame) == 0:
        raise RuntimeError("VFB returned no JRC2018 dataset instances")
    required = {"id", "label", "templates"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"VFB dataset result lacks required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, item in frame.iterrows():
        templates = item["templates"]
        if not isinstance(templates, (list, tuple, set)) or TEMPLATE_NAME not in templates:
            continue
        vfb_id = str(item["id"]).strip()
        label = str(item["label"]).strip()
        if not vfb_id or not label:
            continue
        suffix = " on JRC2018Unisex adult brain"
        region = label[:-len(suffix)] if label.endswith(suffix) else label
        rows.append(
            {
                "region": region,
                "label": label,
                "vfb_id": vfb_id,
                "nrrd_url": _vfb_image_url(vfb_id),
            }
        )
    return rows


def _head_ok(url: str) -> bool:
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "dashiBRAIN-VFB/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return int(response.status) == 200
    except Exception:
        return False


def resolve_domains() -> tuple[dict, ...]:
    rows = _canonical_dataset_rows()
    by_id = {row["vfb_id"]: row for row in rows}
    rows = sorted(by_id.values(), key=lambda r: (r["region"], r["vfb_id"]))
    if len(rows) != EXPECTED_DOMAIN_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_DOMAIN_COUNT} JRC2018Unisex painted domains, found {len(rows)}"
        )
    regions = {row["region"] for row in rows}
    for required in ("AMMC", "WED"):
        if required not in regions:
            raise RuntimeError(f"required calibration region missing from VFB dataset: {required}")

    missing_urls = [row["vfb_id"] for row in rows if not _head_ok(row["nrrd_url"])]
    if missing_urls:
        raise RuntimeError(
            "VFB painted-domain volume URLs failed verification: " + ", ".join(missing_urls[:10])
        )
    return tuple(rows)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dashiBRAIN-VFB/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, part.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    part.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/vfb/jrc2018_painted_domains")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    domains = resolve_domains()

    manifest_rows: list[dict] = []
    for index, row in enumerate(domains):
        filename = f"{index:03d}_{_slug(row['region'])}_{row['vfb_id']}.nrrd"
        target = out / filename
        if not args.metadata_only and not target.exists():
            download(row["nrrd_url"], target)
        manifest_rows.append({"index": index, **row, "local_path": str(target)})

    manifest_csv = out / "jrc2018_painted_domains.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "region", "label", "vfb_id", "nrrd_url", "local_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "dataset": DATASET_ID,
        "template": TEMPLATE_NAME,
        "template_vfb_id": TEMPLATE_ID,
        "domain_count": len(manifest_rows),
        "contains_ammc": any(r["region"] == "AMMC" for r in manifest_rows),
        "contains_wed": any(r["region"] == "WED" for r in manifest_rows),
        "manifest": str(manifest_csv),
        "discovery": "VFB canonical API via vfb_connect.get_instances_by_dataset",
        "binary_domains_preserved_separately": True,
    }
    (out / "jrc2018_painted_domains.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
