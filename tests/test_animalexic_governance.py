from dashi.analysis.animalexic_governance import (
    BehaviourMotif,
    Decision,
    GovernedCandidate,
    Modality,
    MultimodalTrace,
    ProvenanceNode,
    ResidualGate,
    can_promote_multimodal_state,
    independent_by_upstream_closure,
    recurrent_motif_does_not_imply_semantics,
)
from dashi.analysis.embodied import DROSOPHILA_CALCIUM_SOURCE


def test_promoted_candidate_requires_receipt():
    candidate = GovernedCandidate(
        observation="obs",
        candidate="state",
        decision=Decision.PROMOTED,
        receipt=None,
        materialise=lambda x: x,
        admissible_receipt=lambda *_: True,
    )
    try:
        candidate.promoted_state()
    except ValueError as exc:
        assert "receipt" in str(exc).lower()
    else:
        raise AssertionError("receiptless promotion must fail")


def test_abstained_candidate_cannot_materialise_canonical_state():
    candidate = GovernedCandidate(
        observation="obs",
        candidate="state",
        decision=Decision.ABSTAIN,
        receipt="receipt",
        materialise=lambda x: x,
        admissible_receipt=lambda *_: True,
    )
    try:
        candidate.promoted_state()
    except ValueError as exc:
        assert "promoted" in str(exc).lower()
    else:
        raise AssertionError("abstained candidate must not become canonical state")


def test_shared_upstream_root_blocks_independence():
    nodes = {
        "calcium": ProvenanceNode("calcium", ("same_fly",)),
        "motion": ProvenanceNode("motion", ("same_fly",)),
        "same_fly": ProvenanceNode("same_fly"),
    }
    assert independent_by_upstream_closure(nodes, "calcium", "motion") is False


def test_multimodal_promotion_can_require_independent_roots():
    trace = MultimodalTrace(
        frozenset({Modality.OPTICAL_CALCIUM, Modality.MOTION}),
        ("calcium", "motion"),
    )
    nodes = {
        "calcium": ProvenanceNode("calcium", ("same_fly",)),
        "motion": ProvenanceNode("motion", ("same_fly",)),
        "same_fly": ProvenanceNode("same_fly"),
    }
    gate = ResidualGate((0.1, 0.2), lambda x: x < 0.5)
    assert can_promote_multimodal_state(trace, gate, nodes) is True
    assert can_promote_multimodal_state(
        trace, gate, nodes, require_independent_roots=True
    ) is False


def test_single_channel_cannot_satisfy_multimodal_gate():
    trace = MultimodalTrace(frozenset({Modality.OPTICAL_CALCIUM}), ("calcium",))
    gate = ResidualGate((0.1,), lambda x: x < 0.5)
    assert can_promote_multimodal_state(trace, gate, {}) is False


def test_recurrent_motif_does_not_name_itself():
    motif = BehaviourMotif(
        motif="turn-left-cluster-7",
        interval=(10, 20),
        source=DROSOPHILA_CALCIUM_SOURCE,
    )
    assert recurrent_motif_does_not_imply_semantics(motif) is True
