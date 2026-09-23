"""Official MaleCNS v1.0 source coordinates and authority boundaries.

This registry mirrors the canonical MaleCNS download page without folding large
bulk artifacts into the repo's recommended working-set manifest.  It separates
connectivity authority from spatial-template products and from API access.
"""

from __future__ import annotations

from dataclasses import dataclass


MALECNS_DATASET_ID = "male-cns:v1.0"
MALECNS_NEUPRINT_SERVER = "https://neuprint.janelia.org"
MALECNS_BULK_ROOT = "gs://flyem-male-cns/v1.0/connectome-data/flat-connectome"


@dataclass(frozen=True)
class MaleCNSOfficialSource:
    key: str
    role: str
    dataset_id: str
    gs_uri: str
    https_url: str | None = None
    can_pay_physical_connectivity: bool = False
    can_pay_spatial_template_transform: bool = False
    note: str = ""


OFFICIAL_MALECNS_SOURCES: dict[str, MaleCNSOfficialSource] = {
    "full_connection_graph": MaleCNSOfficialSource(
        key="full_connection_graph",
        role="segment_to_segment_physical_connectivity",
        dataset_id=MALECNS_DATASET_ID,
        gs_uri=(
            f"{MALECNS_BULK_ROOT}/"
            "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        ),
        https_url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
            "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        ),
        can_pay_physical_connectivity=True,
        note="Official full segment-to-segment connection graph.",
    ),
    "synaptic_partners": MaleCNSOfficialSource(
        key="synaptic_partners",
        role="synapse_partner_pairs",
        dataset_id=MALECNS_DATASET_ID,
        gs_uri=f"{MALECNS_BULK_ROOT}/syn-partners-male-cns-v1.0-minconf-0.5.feather",
        https_url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
            "flat-connectome/syn-partners-male-cns-v1.0-minconf-0.5.feather"
        ),
        can_pay_physical_connectivity=True,
        note="Synaptic partner pairs with body IDs and primary neuropil.",
    ),
    "body_neurotransmitters": MaleCNSOfficialSource(
        key="body_neurotransmitters",
        role="body_neurotransmitter_predictions",
        dataset_id=MALECNS_DATASET_ID,
        gs_uri=f"{MALECNS_BULK_ROOT}/body-neurotransmitters-male-cns-v1.0.feather",
        https_url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
            "flat-connectome/body-neurotransmitters-male-cns-v1.0.feather"
        ),
        note="Aggregate neurotransmitter predictions per neuron body.",
    ),
    "fullbrain_roi": MaleCNSOfficialSource(
        key="fullbrain_roi",
        role="neuropil_segmentation",
        dataset_id=MALECNS_DATASET_ID,
        gs_uri="gs://flyem-male-cns/rois/fullbrain-roi-v4",
        note="Brain neuropil compartment segmentation transferred from JRC2018M and manually refined.",
    ),
    "jrc2018_unisex_skeletons": MaleCNSOfficialSource(
        key="jrc2018_unisex_skeletons",
        role="spatial_transform_product",
        dataset_id=MALECNS_DATASET_ID,
        gs_uri=(
            "gs://flyem-male-cns/v1.0/segmentation/"
            "skeletons-unisex-template/"
        ),
        can_pay_spatial_template_transform=True,
        note=(
            "MaleCNS skeletons transformed to JRC2018 unisex template space; "
            "transform is available via navis-flybrains."
        ),
    ),
}


def physical_segment_incidence_source() -> MaleCNSOfficialSource:
    """Return the canonical source for segment-level physical connectivity."""
    return OFFICIAL_MALECNS_SOURCES["full_connection_graph"]
