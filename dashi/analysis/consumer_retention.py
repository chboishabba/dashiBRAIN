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
import re
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


_HEX_LITERAL_RE = re.compile(r"^0[xX][0-9a-fA-F]+$")
_MEMORY_ADDRESS_RE = re.compile(r"^0[xX][0-9a-fA-F]+(?:\(\)|[\[\]+*-].*)$")
_RAW_DOM_RE = re.compile(r"^\s*</?[A-Za-z][^>]*>\s*$")
_RAW_UI_RE = re.compile(
    r"^(?:collapse|expand|button|menu|dropdown|tooltip|modal)"
    r"(?:\s*\(|\s*:|\s*$)",
    re.IGNORECASE,
)


def classify_topic_token(token: str) -> str:
    """Classify one candidate topic label before spectral quotienting.

    The classifier is deliberately conservative. It rejects exact lexical
    forms that are implementation artifacts. It does not infer semantics.
    """
    value = token.strip()
    if not value:
        return "raw_ui_token_noise"
    if _MEMORY_ADDRESS_RE.fullmatch(value):
        return "memory_address_noise"
    if _HEX_LITERAL_RE.fullmatch(value):
        return "hex_literal_noise"
    if _RAW_DOM_RE.fullmatch(value):
        return "raw_dom_token_noise"
    if _RAW_UI_RE.match(value) or "property:" in value.lower():
        return "raw_ui_token_noise"
    return "semantic_token"


def _admitted_topic_labels(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        label
        for label in labels
        if classify_topic_token(label) == "semantic_token"
    )


@dataclass(frozen=True)
class RepresentationHierarchyAudit:
    """Finite threshold hierarchy over admitted representation labels."""

    representations: Mapping[str, tuple[float, ...]]
    admitted_labels: tuple[str, ...]
    primary_topics: tuple[tuple[str, ...], ...]
    subtopics: tuple[tuple[str, ...], ...]
    fibres: tuple[tuple[str, ...], ...]
    primary_threshold: float
    subtopic_threshold: float
    fibre_threshold: float


def _representation_distance(
    representations: Mapping[str, tuple[float, ...]],
    left: str,
    right: str,
) -> float:
    return math.dist(representations[left], representations[right])


def _threshold_components(
    labels: Sequence[str],
    representations: Mapping[str, tuple[float, ...]],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic connected components of a threshold graph."""
    unseen = set(labels)
    components: list[tuple[str, ...]] = []
    for root in sorted(labels):
        if root not in unseen:
            continue
        unseen.remove(root)
        stack = [root]
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = [
                candidate
                for candidate in sorted(unseen)
                if _representation_distance(
                    representations, current, candidate
                ) <= threshold
            ]
            for candidate in neighbours:
                unseen.remove(candidate)
                stack.append(candidate)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _complete_link_fibres(
    labels: Sequence[str],
    representations: Mapping[str, tuple[float, ...]],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    """Build deterministic cliques whose every pair is within threshold."""
    fibres: list[list[str]] = []
    for label in sorted(labels):
        placed = False
        for fibre in fibres:
            if all(
                _representation_distance(representations, label, member)
                <= threshold
                for member in fibre
            ):
                fibre.append(label)
                placed = True
                break
        if not placed:
            fibres.append([label])
    return tuple(tuple(fibre) for fibre in fibres)


def stratify_representations(
    representations: Mapping[str, Sequence[float]],
    *,
    primary_threshold: float,
    subtopic_threshold: float,
    fibre_threshold: float,
) -> RepresentationHierarchyAudit:
    """Construct primary topics, subtopics, and complete-link fibres.

    Primary topics are connected components of the declared macro threshold
    graph. Subtopics are connected components at the tighter local threshold
    inside each primary topic. Fibres use complete-link admission at the
    tightest threshold. Noise labels are removed before every construction.
    """
    thresholds = (
        float(fibre_threshold),
        float(subtopic_threshold),
        float(primary_threshold),
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in thresholds):
        raise ValueError("hierarchy thresholds must be finite and nonnegative")
    if not (thresholds[0] <= thresholds[1] <= thresholds[2]):
        raise ValueError(
            "hierarchy thresholds must satisfy fibre <= subtopic <= primary"
        )

    clean: dict[str, tuple[float, ...]] = {}
    dimension: int | None = None
    for label in sorted(representations):
        if classify_topic_token(label) != "semantic_token":
            continue
        coords = tuple(float(value) for value in representations[label])
        if not coords or not all(math.isfinite(value) for value in coords):
            raise ValueError("representation coordinates must be finite")
        if dimension is None:
            dimension = len(coords)
        elif len(coords) != dimension:
            raise ValueError("all representations must have one dimension")
        clean[label] = coords
    if not clean:
        raise ValueError("at least one semantic representation is required")

    labels = tuple(sorted(clean))
    primary_topics = _threshold_components(
        labels, clean, thresholds[2]
    )

    subtopics_list: list[tuple[str, ...]] = []
    fibres_list: list[tuple[str, ...]] = []
    for primary in primary_topics:
        local_subtopics = _threshold_components(
            primary, clean, thresholds[1]
        )
        subtopics_list.extend(local_subtopics)
        for subtopic in local_subtopics:
            fibres_list.extend(
                _complete_link_fibres(
                    subtopic, clean, thresholds[0]
                )
            )

    return RepresentationHierarchyAudit(
        representations=clean,
        admitted_labels=labels,
        primary_topics=primary_topics,
        subtopics=tuple(subtopics_list),
        fibres=tuple(fibres_list),
        primary_threshold=thresholds[2],
        subtopic_threshold=thresholds[1],
        fibre_threshold=thresholds[0],
    )


@dataclass(frozen=True)
class SpectralStratificationAudit:
    """Finite runtime receipt for the golden five-level AO hierarchy."""

    k: int
    lambda_plus: float
    selected_eigen_indices: tuple[int, ...]
    primary_topics: tuple[str, ...]
    subtopics: tuple[str, ...]
    fibre_count: int
    unblocked_goal_count: int
    levels: tuple[str, ...] = (
        "eigenspace",
        "primary_topic",
        "subtopic",
        "fibre",
        "goal",
    )


def audit_spectral_stratification(
    *,
    eigenvalues: Sequence[float],
    lambda_plus: float,
    k: int,
    primary_labels: Iterable[str],
    subtopic_labels: Iterable[str],
    fibre_count: int,
    unblocked_goal_count: int,
) -> SpectralStratificationAudit:
    """Validate a finite spectral discriminator receipt.

    Numeric eigendecomposition remains upstream. This function checks the
    golden admission contract only: K is 50 or 100, selected indices satisfy
    lambda_i > lambda_plus, and implementation-noise labels are removed before
    topic/subtopic admission.
    """
    if k not in (50, 100):
        raise ValueError("spectral truncation K must be 50 or 100")
    if not math.isfinite(lambda_plus):
        raise ValueError("lambda_plus must be finite")
    if fibre_count < 0 or unblocked_goal_count < 0:
        raise ValueError("fibre and goal counts must be nonnegative")
    clean_eigenvalues = tuple(float(value) for value in eigenvalues)
    if not all(math.isfinite(value) for value in clean_eigenvalues):
        raise ValueError("eigenvalues must be finite")
    selected = tuple(
        index
        for index, value in enumerate(clean_eigenvalues[:k])
        if value > lambda_plus
    )
    return SpectralStratificationAudit(
        k=k,
        lambda_plus=float(lambda_plus),
        selected_eigen_indices=selected,
        primary_topics=_admitted_topic_labels(primary_labels),
        subtopics=_admitted_topic_labels(subtopic_labels),
        fibre_count=fibre_count,
        unblocked_goal_count=unblocked_goal_count,
    )


@dataclass(frozen=True)
class FleetGoal:
    """One typed AO deliverable eligible for independent dispatch."""

    goal_id: str
    pareto_rank: int
    unblocked: bool
    pareto_eligible: bool
    typed_support: frozenset[str]


def _no_typed_meet(left: FleetGoal, right: FleetGoal) -> bool:
    return left.typed_support.isdisjoint(right.typed_support)


def schedule_full_twenty(goals: Iterable[FleetGoal]) -> tuple[FleetGoal, ...]:
    """Select exactly 20 top-ranked pairwise-independent unblocked goals.

    Lower integer pareto_rank is preferred. Ties are resolved by goal_id.
    The function fails closed when it cannot fill all twenty slots.
    """
    candidates = sorted(
        (
            goal
            for goal in goals
            if goal.unblocked and goal.pareto_eligible
        ),
        key=lambda goal: (goal.pareto_rank, goal.goal_id),
    )
    selected: list[FleetGoal] = []
    for goal in candidates:
        if not goal.goal_id:
            raise ValueError("goal_id must be nonempty")
        if any(goal.goal_id == prior.goal_id for prior in selected):
            continue
        if all(_no_typed_meet(goal, prior) for prior in selected):
            selected.append(goal)
            if len(selected) == 20:
                return tuple(selected)
    raise ValueError(
        "full concurrency requires 20 pairwise noTypedMeet "
        "unblocked Pareto-eligible goals"
    )


@dataclass(frozen=True)
class MaxCutAudit:
    """Exact finite unweighted cut receipt. This is not a physics theorem."""

    vertex_count: int
    edge_count: int
    cut_edge_count: int
    boundary_vertices: frozenset[int]
    laplacian_quadratic: int
    sdp_upper_bound: float | None
    cut_within_sdp_bound: bool | None


def audit_max_cut(
    *,
    vertex_count: int,
    edges: Iterable[tuple[int, int]],
    left_vertices: set[int] | frozenset[int],
    sdp_upper_bound: float | None = None,
) -> MaxCutAudit:
    """Audit a literal finite two-way cut and its ±1 Laplacian identity.

    For each undirected unit-weight edge, the signal contribution
    (x_u - x_v)^2 is 4 exactly when the edge crosses the cut.
    """
    if vertex_count <= 0:
        raise ValueError("vertex_count must be positive")
    left = frozenset(left_vertices)
    if any(vertex < 0 or vertex >= vertex_count for vertex in left):
        raise ValueError("left vertex index out of range")

    normalized_edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_u, raw_v in edges:
        u, v = int(raw_u), int(raw_v)
        if u < 0 or u >= vertex_count or v < 0 or v >= vertex_count:
            raise ValueError("edge vertex index out of range")
        if u == v:
            raise ValueError("self edges are not valid cut edges")
        edge = (min(u, v), max(u, v))
        if edge in seen:
            raise ValueError("duplicate undirected edge")
        seen.add(edge)
        normalized_edges.append(edge)

    cut_edges: list[tuple[int, int]] = []
    boundary: set[int] = set()
    quadratic = 0
    for u, v in normalized_edges:
        xu = 1 if u in left else -1
        xv = 1 if v in left else -1
        contribution = (xu - xv) ** 2
        quadratic += contribution
        if contribution:
            cut_edges.append((u, v))
            boundary.add(u)
            boundary.add(v)

    expected = 4 * len(cut_edges)
    if quadratic != expected:
        raise AssertionError("finite cut Laplacian identity failed")

    if sdp_upper_bound is None:
        within = None
    else:
        bound = float(sdp_upper_bound)
        if not math.isfinite(bound):
            raise ValueError("SDP upper bound must be finite")
        if len(cut_edges) > bound:
            raise ValueError("cut weight exceeds supplied SDP upper bound")
        within = True

    return MaxCutAudit(
        vertex_count=vertex_count,
        edge_count=len(normalized_edges),
        cut_edge_count=len(cut_edges),
        boundary_vertices=frozenset(boundary),
        laplacian_quadratic=quadratic,
        sdp_upper_bound=None if sdp_upper_bound is None else float(sdp_upper_bound),
        cut_within_sdp_bound=within,
    )
