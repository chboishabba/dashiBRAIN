from __future__ import annotations

import pytest

from scripts.fetch_vfb_jrc2018_painted_domains import (
    EXPECTED_DOMAIN_COUNT,
    TEMPLATE_ID,
    TEMPLATE_NAME,
    _vfb_image_url,
)


def test_vfb_jrc2018_template_identity_is_explicit():
    assert TEMPLATE_ID == "VFB_00101567"
    assert TEMPLATE_NAME == "JRC2018Unisex"
    assert EXPECTED_DOMAIN_COUNT == 46


def test_vfb_domain_id_maps_to_template_specific_volume_url():
    assert _vfb_image_url("VFB_00102273") == (
        "https://www.virtualflybrain.org/data/VFB/i/0010/2273/"
        "VFB_00101567/volume.nrrd"
    )
    assert _vfb_image_url("VFB_00102154") == (
        "https://www.virtualflybrain.org/data/VFB/i/0010/2154/"
        "VFB_00101567/volume.nrrd"
    )


def test_vfb_domain_url_rejects_non_vfb_identifier():
    with pytest.raises(ValueError, match="unexpected VFB identifier"):
        _vfb_image_url("AMMC")
