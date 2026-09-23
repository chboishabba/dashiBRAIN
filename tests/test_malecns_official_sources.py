from dashi.io.malecns_official_sources import (
    MALECNS_DATASET_ID,
    MALECNS_NEUPRINT_SERVER,
    OFFICIAL_MALECNS_SOURCES,
    physical_segment_incidence_source,
)


def test_official_dataset_and_neuprint_coordinates_are_pinned():
    assert MALECNS_DATASET_ID == "male-cns:v1.0"
    assert MALECNS_NEUPRINT_SERVER == "https://neuprint.janelia.org"


def test_full_connection_graph_is_the_segment_incidence_source():
    source = physical_segment_incidence_source()
    assert source.key == "full_connection_graph"
    assert source.role == "segment_to_segment_physical_connectivity"
    assert source.dataset_id == MALECNS_DATASET_ID
    assert source.gs_uri.endswith(
        "/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    )
    assert "significant-only" not in source.gs_uri


def test_official_source_registry_keeps_distinct_connectivity_and_transform_roles():
    assert OFFICIAL_MALECNS_SOURCES["synaptic_partners"].role == "synapse_partner_pairs"
    assert OFFICIAL_MALECNS_SOURCES["body_neurotransmitters"].role == "body_neurotransmitter_predictions"
    assert OFFICIAL_MALECNS_SOURCES["fullbrain_roi"].role == "neuropil_segmentation"
    assert OFFICIAL_MALECNS_SOURCES["jrc2018_unisex_skeletons"].role == "spatial_transform_product"


def test_jrc2018_skeleton_product_does_not_claim_connectivity_authority():
    transformed = OFFICIAL_MALECNS_SOURCES["jrc2018_unisex_skeletons"]
    assert transformed.can_pay_physical_connectivity is False
    assert transformed.can_pay_spatial_template_transform is True
