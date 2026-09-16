from __future__ import annotations

from dashi.io.princeton_data_commons import (
    PDCMirrorRecord,
    parse_globus_file_manager_url,
    parse_pdc_document_export,
)


def test_parse_pdc_document_export_retains_files_and_globus_coordinates():
    payload = {
        "id": "doi-10-34770-s5hx-1x75",
        "title": "High-speed whole-brain imaging in Drosophila",
        "doi_value": "10.34770/s5hx-1x75",
        "globus_url": (
            "https://app.globus.org/file-manager?"
            "origin_id=dc43f461-0ca7-4203-848c-33a9fc00a464&"
            "origin_path=%2F10.34770%2Fs5hx-1x75%2F1%2F"
        ),
        "files": [
            {"filename": "10.34770/s5hx-1x75/1/Data.zip", "size": 123},
            {"filename": "10.34770/s5hx-1x75/1/README.txt", "size": 45},
        ],
    }

    record = parse_pdc_document_export(payload)
    assert isinstance(record, PDCMirrorRecord)
    assert record.doi == "10.34770/s5hx-1x75"
    assert record.record_id == "doi-10-34770-s5hx-1x75"
    assert record.globus_origin_id == "dc43f461-0ca7-4203-848c-33a9fc00a464"
    assert record.globus_origin_path == "/10.34770/s5hx-1x75/1/"
    assert record.file_names == (
        "10.34770/s5hx-1x75/1/Data.zip",
        "10.34770/s5hx-1x75/1/README.txt",
    )


def test_parse_globus_file_manager_url_requires_origin_id_and_path():
    origin_id, origin_path = parse_globus_file_manager_url(
        "https://app.globus.org/file-manager?origin_id=abc&origin_path=%2Fa%2Fb%2F"
    )
    assert origin_id == "abc"
    assert origin_path == "/a/b/"


def test_document_export_rejects_missing_globus_url_for_large_mirror():
    payload = {
        "id": "doi-10-34770-s5hx-1x75",
        "doi_value": "10.34770/s5hx-1x75",
        "files": [],
        "globus_url": None,
    }
    try:
        parse_pdc_document_export(payload)
    except ValueError as exc:
        assert "globus_url" in str(exc)
    else:
        raise AssertionError("missing Globus coordinates must not be accepted")
