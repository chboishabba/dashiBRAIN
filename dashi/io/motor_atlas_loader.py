"""Load coarse Azevedo/Lesser motor-module resources.

Sources:
- Anthony Azevedo et al., *Connectomic reconstruction of a female Drosophila
  ventral nerve cord*, DOI 10.1038/s41586-024-07389-x.
- Ellen Lesser, Anthony W. Azevedo, Jasper S. Phelps et al., *Synaptic
  architecture of leg and wing premotor control networks in Drosophila*,
  DOI 10.1038/s41586-024-07600-z.

These are cross-animal/female FANC/VNC resources. Loading them never proves
literal identity with a MaleCNS neuron from another specimen.
"""

from __future__ import annotations

from pathlib import Path
import json

from dashi.analysis.effector_behaviour_real import MotorModuleMap


def load_fanc_muscle_json_directory(path: str | Path) -> MotorModuleMap:
    """Build a segment -> muscle-module mapping from repository ``jsons/*.json``.

    JSON files are expected to carry ``segments`` and optionally
    ``hiddenSegments``. The filename stem is retained as the muscle/module label.
    """
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"motor JSON directory not found: {root}")

    neuron_to_module: dict[str, str] = {}
    module_to_effector: dict[str, str] = {}
    for p in sorted(root.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        module = p.stem
        module_to_effector[module] = module
        segments = list(payload.get("segments") or []) + list(payload.get("hiddenSegments") or [])
        for segment in segments:
            neuron_to_module[str(segment)] = module

    if not neuron_to_module:
        raise ValueError(f"no segment mappings found in {root}")
    return MotorModuleMap(
        neuron_to_module=neuron_to_module,
        module_to_effector=module_to_effector,
        same_animal_identity=False,
    )


def load_escape_dataframe(path: str | Path) -> MotorModuleMap:
    """Load the repository ``escape_df.pkl`` motor/premotor table.

    Pickle must only be loaded from the explicitly trusted, pinned repository
    artifact. ``post_pt_root_id`` is used as the identified motor-neuron carrier;
    ``cell_type`` is retained as the coarse effector/module label.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"escape dataframe not found: {p}")
    import pandas as pd
    df = pd.read_pickle(p)
    required = {"post_pt_root_id", "cell_type"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"escape dataframe missing columns: {sorted(missing)}")

    neuron_to_module: dict[str, str] = {}
    module_to_effector: dict[str, str] = {}
    for neuron, cell_type in zip(df["post_pt_root_id"], df["cell_type"]):
        if cell_type is None:
            continue
        module = str(cell_type)
        neuron_to_module[str(neuron)] = module
        module_to_effector[module] = module

    if not neuron_to_module:
        raise ValueError("escape dataframe contains no usable motor mappings")
    return MotorModuleMap(
        neuron_to_module=neuron_to_module,
        module_to_effector=module_to_effector,
        same_animal_identity=False,
    )
