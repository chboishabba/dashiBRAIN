#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-data/jrc2018/JRC2018_UNISEX_38um_iso_16bit.nrrd}
URL="https://raw.githubusercontent.com/natverse/nat.flybrains/98a3931bdad8580c7db21fccc79ae5ee4225f9c9/data-raw/JRC2018_UNISEX_38um_iso_16bit.nrrd"

mkdir -p "$(dirname "$OUT")"
# curl -C - resumes an interrupted download; retry-all-errors covers transient CDN failures.
curl -fL --retry 8 --retry-delay 2 --retry-all-errors -C - -o "$OUT" "$URL"

python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
with p.open('rb') as f:
    header = f.read(4096).split(b'\n\n', 1)[0].decode('ascii', 'replace')
required = [
    'NRRD0004',
    'type: uint16',
    'sizes: 1652 773 456',
    'space directions: (0.38,0,0) (0,0.38,0) (0,0,0.38)',
]
missing = [item for item in required if item not in header]
if missing:
    raise SystemExit(f'JRC2018Unisex NRRD header mismatch: missing {missing!r}')
print(f'JRC2018Unisex template ready: {p} ({p.stat().st_size} bytes)')
PY
