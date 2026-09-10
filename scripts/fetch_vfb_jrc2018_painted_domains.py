#!/usr/bin/env python3
"""Fetch the 46 VFB-painted JRC2018Unisex neuropil domains as NRRDs.

Source dataset: Virtual Fly Brain JRC2018, whose painted domains are aligned to
JRC2018Unisex (VFB_00101567).  Each domain is retained as its own binary image;
we deliberately do not collapse overlapping domains/subdomains into a single
label volume.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import urllib.request

DATASET_JSON_URL = "https://www.virtualflybrain.org/data/VFB/json/JRC2018.html"
TEMPLATE_ID = "VFB_00101567"
EXPECTED_DOMAIN_COUNT = 46


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "dashiBRAIN-VFB/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError("VFB dataset response is not a JSON object")
    return payload


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._'-]+", "_", text.strip())
    return value.strip("_") or "region"


def resolve_domains(payload: dict) -> tuple[dict, ...]:
    rows: list[dict] = []
    for item in payload.get("anatomy_channel_image", []):
        anatomy = item.get("anatomy") or {}
        channel_image = item.get("channel_image") or {}
        image = channel_image.get("image") or {}
        template = image.get("template_anatomy") or {}
        if template.get("short_form") != TEMPLATE_ID:
            continue
        label = str(anatomy.get("label") or "").strip()
        vfb_id = str(anatomy.get("short_form") or "").strip()
        folder = str(image.get("image_folder") or "").strip().replace("http://", "https://")
        if not label or not vfb_id or not folder:
            continue
        suffix = " on JRC2018Unisex adult brain"
        region = label[:-len(suffix)] if label.endswith(suffix) else label
        indexes = image.get("index") or []
        index = int(indexes[0]) if indexes else None
        rows.append({
            "region": region,
            "label": label,
            "vfb_id": vfb_id,
            "index": index,
            "nrrd_url": folder.rstrip("/") + "/volume.nrrd",
        })
    by_id = {row["vfb_id"]: row for row in rows}
    rows = sorted(by_id.values(), key=lambda r: (999999 if r["index"] is None else r["index"], r["region"]))
    if len(rows) != EXPECTED_DOMAIN_COUNT:
        raise RuntimeError(f"expected {EXPECTED_DOMAIN_COUNT} JRC2018 painted domains, found {len(rows)}")
    regions = {row["region"] for row in rows}
    for required in ("AMMC", "WED"):
        if required not in regions:
            raise RuntimeError(f"required calibration region missing from VFB dataset: {required}")
    return tuple(rows)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dashiBRAIN-VFB/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/vfb/jrc2018_painted_domains")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _get_json(DATASET_JSON_URL)
    domains = resolve_domains(payload)

    manifest_rows: list[dict] = []
    for row in domains:
        filename = f"{row['index']:03d}_{_slug(row['region'])}_{row['vfb_id']}.nrrd" if row["index"] is not None else f"{_slug(row['region'])}_{row['vfb_id']}.nrrd"
        target = out / filename
        if not args.metadata_only and not target.exists():
            download(row["nrrd_url"], target)
        manifest_rows.append({**row, "local_path": str(target)})

    manifest_csv = out / "jrc2018_painted_domains.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "region", "label", "vfb_id", "nrrd_url", "local_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "dataset": "JRC2018",
        "template": "JRC2018Unisex",
        "template_vfb_id": TEMPLATE_ID,
        "domain_count": len(manifest_rows),
        "contains_ammc": any(r["region"] == "AMMC" for r in manifest_rows),
        "contains_wed": any(r["region"] == "WED" for r in manifest_rows),
        "manifest": str(manifest_csv),
        "binary_domains_preserved_separately": True,
    }
    (out / "jrc2018_painted_domains.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
