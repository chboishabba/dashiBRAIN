"""Local fibre hyperfabric carrier for structure/function experiments.

This module deliberately separates:

* base/incidence geometry;
* arbitrary local fibre coordinates;
* local refinement at crossings;
* typed seam/gluing compatibility;
* symmetry actions; and
* consumer-facing chart projection.

The historical eight-column MaleCNS NDim family is represented only as one
finite chart of this carrier.  It is not treated as a universal fibre count or
as the topology of the biological network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np

from dashi.analysis.ndim_structure_function import StructuralFibreFamily


@dataclass(frozen=True, order=True)
class BaseLocality:
    """A point/cell/interface in the incidence base."""

    source: str
    target: str
    hop: int | None = None
    layer: str = "region-pair"


@dataclass(frozen=True)
class Incidence:
    """Typed relation between base localities; not a fibre genealogy edge."""

    left: BaseLocality
    right: BaseLocality
    relation: str
    provenance: str = ""


@dataclass(frozen=True)
class LocalFibreCoordinate:
    """One locally exposed coordinate over one base locality."""

    name: str
    value: float
    axis: str
    provenance: str = ""


@dataclass(frozen=True)
class Crossing:
    """A locality at which several fibres can coexist and be refined."""

    locality: BaseLocality
    participating_fibres: tuple[str, ...]
    crossing_kind: str = "coexistence"


@dataclass(frozen=True)
class Seam:
    """A typed local gluing interface.

    Equality of a destination label is intentionally insufficient.  `matches`
    is the application-specific interface receipt.
    """

    left: BaseLocality
    right: BaseLocality
    interface: str
    matches: bool
    receipt: str


@dataclass(frozen=True)
class PantsPatch:
    """Local 1->n or n->1 branch/merge patch over matched interfaces."""

    inputs: tuple[BaseLocality, ...]
    outputs: tuple[BaseLocality, ...]
    seams: tuple[Seam, ...]
    receipt: str

    @property
    def is_gluable(self) -> bool:
        return bool(self.inputs and self.outputs and all(seam.matches for seam in self.seams))


@dataclass(frozen=True)
class SymmetryAction:
    """A base permutation plus optional fibre-name permutation.

    This records an action only.  Quotienting additionally requires a declared
    consumer-invariance check.
    """

    name: str
    base_map: Mapping[BaseLocality, BaseLocality]
    fibre_map: Mapping[str, str]


@dataclass(frozen=True)
class LocalFibreHyperfabric:
    regions: tuple[str, ...]
    fibres: Mapping[BaseLocality, tuple[LocalFibreCoordinate, ...]]
    incidences: tuple[Incidence, ...] = ()
    crossings: tuple[Crossing, ...] = ()
    pants_patches: tuple[PantsPatch, ...] = ()
    symmetries: tuple[SymmetryAction, ...] = ()

    def fibre_names_at(self, locality: BaseLocality) -> tuple[str, ...]:
        return tuple(f.name for f in self.fibres.get(locality, ()))

    def refine_at_crossing(
        self,
        locality: BaseLocality,
        additions: Sequence[LocalFibreCoordinate],
        *,
        crossing_kind: str = "local-refinement",
    ) -> "LocalFibreHyperfabric":
        """Add any number of local fibres without imposing ancestry or time order."""
        if locality not in self.fibres:
            raise KeyError(f"unknown base locality: {locality}")
        old = tuple(self.fibres[locality])
        existing = {f.name for f in old}
        additions_t = tuple(additions)
        duplicate = existing.intersection(f.name for f in additions_t)
        if duplicate:
            raise ValueError(f"fibre(s) already present at locality: {sorted(duplicate)}")
        if len({f.name for f in additions_t}) != len(additions_t):
            raise ValueError("new local fibre names must be unique")
        updated = dict(self.fibres)
        updated[locality] = old + additions_t
        crossing = Crossing(locality, tuple(f.name for f in additions_t), crossing_kind)
        return replace(self, fibres=updated, crossings=self.crossings + (crossing,))

    def with_pants_patch(self, patch: PantsPatch) -> "LocalFibreHyperfabric":
        if not patch.is_gluable:
            raise ValueError("pants patch lacks a complete matching seam receipt")
        known = set(self.fibres)
        missing = [p for p in patch.inputs + patch.outputs if p not in known]
        if missing:
            raise KeyError(f"pants patch references unknown localities: {missing}")
        return replace(self, pants_patches=self.pants_patches + (patch,))

    def with_symmetry(self, action: SymmetryAction) -> "LocalFibreHyperfabric":
        known = set(self.fibres)
        if set(action.base_map) - known or set(action.base_map.values()) - known:
            raise ValueError("symmetry action must map known base localities to known base localities")
        return replace(self, symmetries=self.symmetries + (action,))

    def project_chart(self, names: Sequence[str]) -> StructuralFibreFamily:
        """Project named local coordinates back to the legacy matrix chart.

        Every requested coordinate must exist at every ordered region-pair
        locality.  This is a consumer-facing projection, not a statement that
        the hyperfabric globally has exactly ``len(names)`` fibres.
        """
        requested = tuple(names)
        n = len(self.regions)
        matrices = {name: np.empty((n, n), dtype=float) for name in requested}
        for i, source in enumerate(self.regions):
            for j, target in enumerate(self.regions):
                locality = BaseLocality(source, target)
                by_name = {f.name: f.value for f in self.fibres.get(locality, ())}
                missing = [name for name in requested if name not in by_name]
                if missing:
                    raise KeyError(f"locality {locality} is missing projected fibre(s): {missing}")
                for name in requested:
                    matrices[name][i, j] = by_name[name]
        return StructuralFibreFamily(self.regions, matrices)

    def symmetry_preserves_consumer(
        self,
        action: SymmetryAction,
        consumer: Callable[[BaseLocality, Mapping[str, float]], object],
    ) -> bool:
        """Check invariance before treating a symmetry as quotient authority."""
        for locality, coords in self.fibres.items():
            target = action.base_map.get(locality, locality)
            source_values = {f.name: f.value for f in coords}
            target_values_raw = {f.name: f.value for f in self.fibres[target]}
            target_values = {
                action.fibre_map.get(name, name): value for name, value in target_values_raw.items()
            }
            if consumer(locality, source_values) != consumer(target, target_values):
                return False
        return True


def hyperfabric_from_structural_family(
    family: StructuralFibreFamily,
    *,
    axis: str = "legacy-ndim-chart",
    provenance: str = "current MaleCNS NDim structural chart",
) -> LocalFibreHyperfabric:
    """Lift any matrix-valued structural family into local pair fibres."""
    regions = tuple(family.regions)
    n = len(regions)
    arrays = {name: np.asarray(matrix, dtype=float) for name, matrix in family.fibres.items()}
    for name, matrix in arrays.items():
        if matrix.shape != (n, n):
            raise ValueError(f"fibre {name!r} does not align with region carrier")

    fibres: dict[BaseLocality, tuple[LocalFibreCoordinate, ...]] = {}
    for i, source in enumerate(regions):
        for j, target in enumerate(regions):
            locality = BaseLocality(source, target)
            fibres[locality] = tuple(
                LocalFibreCoordinate(name, float(matrix[i, j]), axis, provenance)
                for name, matrix in arrays.items()
            )

    return LocalFibreHyperfabric(regions=regions, fibres=fibres)


def chart_round_trip_exact(
    family: StructuralFibreFamily,
    names: Sequence[str] | None = None,
) -> bool:
    """Regression helper: lifting/projecting a declared chart changes no values."""
    chart_names = tuple(names) if names is not None else tuple(family.fibres)
    reconstructed = hyperfabric_from_structural_family(family).project_chart(chart_names)
    return all(
        np.array_equal(np.asarray(family.fibres[name]), np.asarray(reconstructed.fibres[name]))
        for name in chart_names
    )
