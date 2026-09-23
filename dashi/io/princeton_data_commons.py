"""Resolve source-bound Princeton Data Commons mirror metadata.

Gauthey et al. explicitly cite Princeton Data Commons DOI 10.34770/s5hx-1x75
as a second repository for the raw and preprocessed data. Princeton Data
Commons exports record metadata as JSON, including a Globus file-manager URL
for large datasets. This module parses that metadata without assuming a Globus
path from the DOI alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class PDCMirrorFile:
    filename: str
    size: int | None
    url: str | None


@dataclass(frozen=True)
class PDCMirrorRecord:
    record_id: str
    doi: str
    title: str | None
    globus_url: str
    globus_origin_id: str
    globus_origin_path: str
    files: tuple[PDCMirrorFile, ...]

    @property
    def file_names(self) -> tuple[str, ...]:
        return tuple(file.filename for file in self.files)


def parse_globus_file_manager_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "app.globus.org":
        raise ValueError(f"not a Globus file-manager URL: {url!r}")
    query = parse_qs(parsed.query)
    origin_ids = query.get("origin_id", ())
    origin_paths = query.get("origin_path", ())
    if len(origin_ids) != 1 or not origin_ids[0]:
        raise ValueError("Globus URL lacks exactly one origin_id")
    if len(origin_paths) != 1 or not origin_paths[0]:
        raise ValueError("Globus URL lacks exactly one origin_path")
    return str(origin_ids[0]), str(origin_paths[0])


def _parse_file(raw: object) -> PDCMirrorFile:
    if not isinstance(raw, Mapping):
        raise ValueError("PDC files entry must be an object")
    filename = raw.get("filename")
    if filename is None:
        # DocumentExport may serialize DatasetFile attributes rather than the
        # original pdc_describe hash, in which case ``full_path`` is retained.
        filename = raw.get("full_path") or raw.get("name")
    if not isinstance(filename, str) or not filename:
        raise ValueError("PDC file entry lacks filename/full_path/name")
    size_raw = raw.get("size")
    size = None if size_raw is None else int(size_raw)
    url_raw = raw.get("url")
    if url_raw is None:
        url_raw = raw.get("download_url")
    url = None if url_raw is None else str(url_raw)
    return PDCMirrorFile(filename=filename, size=size, url=url)


def parse_pdc_document_export(payload: Mapping[str, object]) -> PDCMirrorRecord:
    record_id = payload.get("id")
    doi = payload.get("doi_value")
    globus_url = payload.get("globus_url")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("PDC document export lacks id")
    if not isinstance(doi, str) or not doi:
        raise ValueError("PDC document export lacks doi_value")
    if not isinstance(globus_url, str) or not globus_url:
        raise ValueError("PDC document export lacks globus_url")
    origin_id, origin_path = parse_globus_file_manager_url(globus_url)

    files_raw = payload.get("files", ())
    if not isinstance(files_raw, Sequence) or isinstance(files_raw, (str, bytes, bytearray)):
        raise ValueError("PDC document export files must be a sequence")
    files = tuple(_parse_file(item) for item in files_raw)

    title_raw = payload.get("title")
    title = None if title_raw is None else str(title_raw)
    return PDCMirrorRecord(
        record_id=record_id,
        doi=doi,
        title=title,
        globus_url=globus_url,
        globus_origin_id=origin_id,
        globus_origin_path=origin_path,
        files=files,
    )
