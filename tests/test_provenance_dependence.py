from dashi.analysis.provenance_dependence import (
    DependenceAwareAggregator,
    EvidenceRelation,
    ProvenanceGraph,
    ProvenanceRoot,
    RootKind,
)


def test_same_trial_modalities_are_not_independent_replication():
    trial = ProvenanceRoot("trial-1", RootKind.TRIAL, "trial:1")
    animal = ProvenanceRoot("fly-1", RootKind.ANIMAL, "animal:fly-1")
    graph = ProvenanceGraph(
        {
            "calcium": frozenset({trial, animal}),
            "motion": frozenset({trial, animal}),
        }
    )
    assert graph.relation("calcium", "motion") is EvidenceRelation.SAME_TRIAL_CORROBORATION
    assert not graph.independent_by_upstream_closure("calcium", "motion")


def test_unknown_provenance_cannot_be_promoted_to_independence():
    graph = ProvenanceGraph({"a": frozenset(), "b": frozenset()})
    assert not graph.independent_by_upstream_closure("a", "b")
    assert graph.relation("a", "b") is EvidenceRelation.INDEPENDENCE_UNDETERMINED


def test_shared_registration_marks_pipeline_corroboration():
    registration = ProvenanceRoot("reg-1", RootKind.REGISTRATION, "sha256:registration")
    graph = ProvenanceGraph(
        {
            "calcium": frozenset({registration}),
            "behaviour": frozenset({registration}),
        }
    )
    assert graph.relation("calcium", "behaviour") is EvidenceRelation.SHARED_PIPELINE_CORROBORATION


def test_dependence_aware_aggregation_collapses_shared_root_class_first():
    trial = ProvenanceRoot("trial-1", RootKind.TRIAL, "trial:1")
    external = ProvenanceRoot("dataset-2", RootKind.DATASET, "dataset:2")
    graph = ProvenanceGraph(
        {
            "calcium": frozenset({trial}),
            "motion": frozenset({trial}),
            "replicate": frozenset({external}),
        }
    )
    aggregator = DependenceAwareAggregator(
        graph=graph,
        combine_within_dependence_class=lambda xs: max(xs),
        combine_across_independent_classes=lambda xs: sum(xs) / len(xs),
    )
    # calcium+motion collapse to max(0.2,0.3)=0.3 before combining with replicate=0.1
    assert aggregator.aggregate({"calcium": 0.2, "motion": 0.3, "replicate": 0.1}) == 0.2
