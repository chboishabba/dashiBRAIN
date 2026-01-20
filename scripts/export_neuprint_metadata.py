#!/usr/bin/env python3
"""Export neuron metadata from neuPrint for locality-preserving coarse grains."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from neuprint import Client, fetch_custom


def _parse_roi_info(raw_roi_info: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Return ROI info as a dict plus a sorted key list."""
    if raw_roi_info is None:
        return {}, []
    if isinstance(raw_roi_info, str):
        try:
            roi_map = json.loads(raw_roi_info)
        except json.JSONDecodeError:
            roi_map = {}
    elif isinstance(raw_roi_info, dict):
        roi_map = raw_roi_info
    else:
        roi_map = {}
    if roi_map is None:
        roi_map = {}
    return roi_map, sorted(roi_map.keys())


def _derive_side(soma_x: float | None, midline_x: float | None) -> str | None:
    if soma_x is None or midline_x is None:
        return None
    return "L" if soma_x < midline_x else "R"


def _unpack_soma(soma_location: Any) -> Tuple[float | None, float | None, float | None]:
    if not isinstance(soma_location, (list, tuple)) or len(soma_location) != 3:
        return None, None, None
    soma_x, soma_y, soma_z = soma_location
    return float(soma_x), float(soma_y), float(soma_z)


def _sanitize_status(status: str | None) -> str | None:
    if status is None:
        return None
    if "'" in status:
        raise ValueError("Status filter cannot contain single quotes.")
    return status


def _build_query(status: str | None) -> str:
    status_clause = f"WHERE n.status = '{status}'" if status else ""
    return f"""
    MATCH (n:Neuron)
    {status_clause}
    RETURN n.bodyId AS bodyId,
           n.type AS type,
           n.instance AS instance,
           n.status AS status,
           n.size AS size,
           n.pre AS pre,
           n.post AS post,
           n.cropped AS cropped,
           n.cellBodyFiber AS cell_body_fiber,
           n.somaLocation AS soma_location,
           n.neurotransmitter AS neurotransmitter,
           n.NTpred AS neurotransmitter_pred,
           n.roiInfo AS roi_info_raw
    """


def export_metadata(
    server: str,
    dataset: str,
    token: str,
    output_csv: str,
    status: str | None,
    midline_x: float | None,
) -> None:
    status = _sanitize_status(status)
    client = Client(server, dataset=dataset, token=token)
    query = _build_query(status)
    df = fetch_custom(query, client=client)

    roi_parsed = df["roi_info_raw"].apply(_parse_roi_info)
    df["roi_info_json"] = [json.dumps(roi_map) for roi_map, _ in roi_parsed]
    df["rois"] = ["|".join(keys) for _, keys in roi_parsed]
    df["roi_count"] = [len(keys) for _, keys in roi_parsed]
    df = df.drop(columns=["roi_info_raw"])

    soma_coords = df["soma_location"].apply(_unpack_soma)
    df["soma_x"] = [coords[0] for coords in soma_coords]
    df["soma_y"] = [coords[1] for coords in soma_coords]
    df["soma_z"] = [coords[2] for coords in soma_coords]
    df["side"] = [_derive_side(coords[0], midline_x) for coords in soma_coords]
    df = df.drop(columns=["soma_location"])

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(df)} neurons to {output_csv}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export neuPrint neuron metadata (ROI membership, soma location, NT pred) "
            "so coarse grains can preserve locality."
        )
    )
    parser.add_argument(
        "--server",
        default="https://neuprint.janelia.org",
        help="neuPrint server URL",
    )
    parser.add_argument(
        "--dataset",
        default="hemibrain:v1.2.1",
        help="neuPrint dataset name",
    )
    parser.add_argument(
        "--token-env",
        default="NEUPRINT_APPLICATION_CREDENTIALS",
        help="Env var with neuPrint API token.",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Optional neuron status filter (e.g., Traced).",
    )
    parser.add_argument(
        "--midline-x",
        type=float,
        default=200_000.0,
        help=(
            "Midline x to derive side label (L/R) from soma_x. "
            "Set to 0 or omit to skip side derivation."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/hemibrain/neurons_metadata.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"Set {args.token_env} with your neuPrint API token.")

    midline_x = args.midline_x if args.midline_x and args.midline_x != 0 else None
    export_metadata(
        server=args.server,
        dataset=args.dataset,
        token=token,
        output_csv=args.output,
        status=args.status,
        midline_x=midline_x,
    )


if __name__ == "__main__":
    main()
