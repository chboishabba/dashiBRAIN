from __future__ import annotations

from scripts.fetch_bifrost_fda import _file_resource


def test_dryad_file_resource_uses_links_when_top_level_id_is_absent():
    meta = {
        "path": "nifti1_compliant_FDA.nii",
        "size": 2430898528,
        "_links": {
            "self": {"href": "/api/v2/files/3172387"},
            "stash:download": {"href": "/api/v2/files/3172387/download"},
        },
    }

    file_id, download_url = _file_resource(meta)

    assert file_id == 3172387
    assert download_url == "https://datadryad.org/api/v2/files/3172387/download"


def test_dryad_file_resource_accepts_download_link_without_numeric_id():
    meta = {
        "_links": {
            "stash:download": {"href": "https://datadryad.org/api/v2/files/example/download"}
        }
    }

    file_id, download_url = _file_resource(meta)

    assert file_id is None
    assert download_url == "https://datadryad.org/api/v2/files/example/download"
