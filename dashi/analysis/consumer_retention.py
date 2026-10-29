"""Finite consumer-retention and coarse-dynamics audits.

A runnable diagnostic companion to dashi_agda's
DASHI.Biology.BrainITConsumerRelativeExact and generic
CoarseObservableFactorisationBidiExact.

No BOLD, connectome or Brain-IT data is loaded here. A finite exact
factorisation test on supplied states is NOT proof of population-level
generalisation, nor is it a numerical fMRI reconstruction experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Generic, Hashable, Iterable, Mapping, Sequence, TypeVar

Fine = TypeVar("Fine")
Coarse = TypeVar("Coarse", bound=Hashable)
Target = TypeVar("Target")


@dataclass(frozen=True)
class ConsumerCollision(Generic[Fine, Coarse, Target]):
    """Two fine states with the same measured value but different consumer."""

    left: Fine
    right: Fine
    observation: Coarse
    left_target: Target
    right_target: Target


@dataclass(frozen=True)
class RetentionAudit(Generic[Fine, Coarse, Target]):
    """Exact relative to the finite supplied state ensemble only."""

    state_count: int
    fibre_count: int
    ambiguous_fibre_count: int
    factorises_on_sample: bool
    collision: ConsumerCollision[Fine, Coarse, Target] | None


def audit_consumer_retention(
    states: Iterable[Fine],
    observation: Callable[[Fine], Coarse],
    consumer: Callable[[Fine], Target],
) -> RetentionAudit[Fine, Coarse, Target]:
    """Test whether `consumer = decoder(observation)` on *all supplied states*.

    Observations must be hashable. Targets need well-defined equality.
    A collision is a mathematical certificate of no exact decoder on even
    this finite subset; absence of a collision is only finite-sample evidence.
    """
    samples = list(states)
    if not samples:
        raise ValueError("at least one fine state must be supplied")
    representatives: dict[Coarse, tuple[Fine, Target]] = {}
    ambiguous: set[Coarse] = set()
    first_collision: ConsumerCollision[Fine, Coarse, Target] | None = None
    for state in samples:
        q = observation(state)
        y = consumer(state)
        if q not in representatives:
            representatives[q] = (state, y)
            continue
        previous_state, previous_value = representatives[q]
        if previous_value != y:
            ambiguous.add(q)
            if first_collision is None:
                first_collision = ConsumerCollision(
                    previous_state, state, q, previous_value, y
                )
    return RetentionAudit(
        state_count=len(samples),
        fibre_count=len(representatives),
        ambiguous_fibre_count=len(ambiguous),
        factorises_on_sample=not ambiguous,
        collision=first_collision,
    )


@dataclass(frozen=True)
class CommutationAudit(Generic[Fine]):
    """Equality-based `Q(K(x)) == K_coarse(Q(x))` on a finite sample."""

    state_count: int
    mismatches: int
    first_mismatch_state: Fine | None

    @property
    def commutes_on_sample(self) -> bool:
        return self.mismatches == 0


def audit_commutation(
    states: Iterable[Fine],
    fine_step: Callable[[Fine], Fine],
    observation: Callable[[Fine], Coarse],
    coarse_step: Callable[[Coarse], Coarse],
) -> CommutationAudit[Fine]:
    """Compare declared fine and coarse dynamics, without inventing a norm.

    Use only where fine/coarse state operations have the same physical,
    temporal and preprocessing identity. Before running real data, define
    an appropriate numerical discrepancy metric in a *separate* protocol.
    """
    samples = list(states)
    if not samples:
        raise ValueError("at least one fine state must be supplied")
    mismatches = 0
    first = None
    for state in samples:
        if observation(fine_step(state)) != coarse_step(observation(state)):
            mismatches += 1
            if first is None:
                first = state
    return CommutationAudit(len(samples), mismatches, first)


Point = Sequence[float]


def _validated_vertex_points(
    vertices: Sequence[Point],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    """Return finite numeric vertex coordinates with one shared dimension."""
    converted: list[tuple[float, ...]] = []
    dimension: int | None = None
    for index, point in enumerate(vertices):
        coords = tuple(float(value) for value in point)
        if len(coords) < 2:
            raise ValueError(f"{name}[{index}] must have at least two coordinates")
        if not all(math.isfinite(value) for value in coords):
            raise ValueError(f"{name}[{index}] contains a non-finite coordinate")
        if dimension is None:
            dimension = len(coords)
        elif len(coords) != dimension:
            raise ValueError(f"all {name} vertices must use one coordinate dimension")
        converted.append(coords)
    return tuple(converted)


def _map_distance_colour(distance: float, maximum: float) -> str:
    """Encode map distance as a deterministic blue-to-red RGB colour."""
    ratio = 0.0 if maximum <= 0.0 else min(1.0, max(0.0, distance / maximum))
    low = (44, 123, 182)
    high = (215, 25, 28)
    rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(low, high))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def render_vertex_map_orbit_svg(
    source_vertices: Sequence[Point],
    target_vertices: Sequence[Point],
    vertex_map: Mapping[int, int],
    *,
    show_source: bool = True,
    max_correspondence_lines: int | None = 200,
    width: int = 960,
    height: int = 720,
    padding: int = 36,
) -> str:
    """Render a finite vertex correspondence as a self-contained SVG artifact.

    Each mapped target vertex is coloured by the maximum Euclidean displacement
    among source vertices that map to it. The maximum is intentional: a
    many-to-one collision remains visually sensitive to the worst displaced
    preimage instead of being hidden by averaging.

    Source vertices are optional. Correspondence lines are deterministic and
    uniformly subsampled so large maps remain inspectable. The SVG records
    target index, map distance and preimage count as data attributes for
    machine-readable collision/retention debugging.

    This diagnostic is exact only for the supplied finite vertices and map.
    It does not establish geometric, biological or population-level identity.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if padding < 0 or 2 * padding >= min(width, height):
        raise ValueError("padding must leave a positive drawing region")
    if max_correspondence_lines is not None and max_correspondence_lines < 0:
        raise ValueError("max_correspondence_lines must be nonnegative or None")

    source = _validated_vertex_points(source_vertices, name="source_vertices")
    target = _validated_vertex_points(target_vertices, name="target_vertices")
    if not target:
        raise ValueError("at least one target vertex must be supplied")

    pairs = sorted(vertex_map.items())
    target_distances: dict[int, list[float]] = {}
    for source_index, target_index in pairs:
        if source_index < 0 or source_index >= len(source):
            raise ValueError(f"source vertex index out of range: {source_index}")
        if target_index < 0 or target_index >= len(target):
            raise ValueError(f"target vertex index out of range: {target_index}")
        if len(source[source_index]) != len(target[target_index]):
            raise ValueError("mapped source and target vertices must have equal dimension")
        distance = math.dist(source[source_index], target[target_index])
        target_distances.setdefault(target_index, []).append(distance)

    mapped_distance = {
        target_index: max(distances)
        for target_index, distances in target_distances.items()
    }
    maximum_distance = max(mapped_distance.values(), default=0.0)

    visible_points = list(target)
    if show_source:
        visible_points.extend(source)
    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    draw_width = width - 2 * padding
    draw_height = height - 2 * padding
    scale = min(draw_width / span_x, draw_height / span_y)
    used_width = span_x * scale
    used_height = span_y * scale
    offset_x = (width - used_width) / 2.0
    offset_y = (height - used_height) / 2.0

    def project(point: tuple[float, ...]) -> tuple[float, float]:
        x = offset_x + (point[0] - min_x) * scale
        # SVG y increases downward, so invert the second coordinate.
        y = height - (offset_y + (point[1] - min_y) * scale)
        return x, y

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="vertex-map orbit correspondence debug view">',
        "<title>Vertex-map orbit correspondence debug view</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    if show_source and pairs and max_correspondence_lines != 0:
        if max_correspondence_lines is None:
            selected_pairs = pairs
        else:
            stride = max(1, math.ceil(len(pairs) / max_correspondence_lines))
            selected_pairs = pairs[::stride]
        for source_index, target_index in selected_pairs:
            sx, sy = project(source[source_index])
            tx, ty = project(target[target_index])
            lines.append(
                f'<line data-layer="correspondence" '
                f'data-source-index="{source_index}" data-target-index="{target_index}" '
                f'x1="{sx:.3f}" y1="{sy:.3f}" x2="{tx:.3f}" y2="{ty:.3f}" '
                f'stroke="#777777" stroke-opacity="0.32" stroke-width="1"/>'
            )

    if show_source:
        for index, point in enumerate(source):
            x, y = project(point)
            lines.append(
                f'<circle data-layer="source" data-source-index="{index}" '
                f'cx="{x:.3f}" cy="{y:.3f}" r="3" '
                f'fill="#222222" fill-opacity="0.55"/>'
            )

    for index, point in enumerate(target):
        x, y = project(point)
        distances = target_distances.get(index, [])
        if distances:
            distance = mapped_distance[index]
            colour = _map_distance_colour(distance, maximum_distance)
            distance_text = f"{distance:.12g}"
            preimage_count = len(distances)
            lines.append(
                f'<circle data-layer="target" data-target-index="{index}" '
                f'data-map-distance="{distance_text}" '
                f'data-map-preimage-count="{preimage_count}" '
                f'cx="{x:.3f}" cy="{y:.3f}" r="5" fill="{colour}" '
                f'stroke="#111111" stroke-width="0.75"/>'
            )
        else:
            lines.append(
                f'<circle data-layer="target" data-target-index="{index}" '
                f'data-map-distance="unmapped" data-map-preimage-count="0" '
                f'cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#bdbdbd" '
                f'stroke="#111111" stroke-width="0.75"/>'
            )

    lines.extend(
        [
            f'<text x="{padding}" y="{height - 12}" font-family="monospace" '
            f'font-size="12" fill="#222222">map distance: 0 → {maximum_distance:.6g}; '
            f'collision count = target preimage count</text>',
            "</svg>",
            "",
        ]
    )
    return "\n".join(lines)


def write_vertex_map_orbit_svg(
    path: str | Path,
    source_vertices: Sequence[Point],
    target_vertices: Sequence[Point],
    vertex_map: Mapping[int, int],
    **render_options: object,
) -> Path:
    """Write one deterministic vertex-map orbit SVG and return its path."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_vertex_map_orbit_svg(
            source_vertices,
            target_vertices,
            vertex_map,
            **render_options,
        ),
        encoding="utf-8",
    )
    return output
