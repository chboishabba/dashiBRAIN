#!/usr/bin/env bash
set -euo pipefail

# VFB's JRC2018Unisex viewer template, on the exact grid used by the 46 painted
# domain NRRDs.  This is intentionally preferred over the detached nat.flybrains
# 0.38 um header because the downstream painted domains are rasterized on this
# VFB-native 1210 x 566 x 174 grid.
OUT=${1:-data/jrc2018/VFB_JRC2018Unisex_volume.nrrd}
URL="https://www.virtualflybrain.org/data/VFB/i/0010/1567/VFB_00101567/volume.nrrd"

mkdir -p "$(dirname "$OUT")"
curl -fL --retry 8 --retry-delay 2 --retry-all-errors -C - -o "$OUT" "$URL"

python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
with p.open('rb') as f:
    header = f.read(4096).split(b'\n\n', 1)[0].decode('ascii', 'replace')
required = [
    'NRRD',
    'sizes: 1210 566 174',
    'space directions:',
]
missing = [item for item in required if item not in header]
if missing:
    raise SystemExit(f'VFB JRC2018Unisex NRRD header mismatch: missing {missing!r}')
if p.stat().st_size <= 4096:
    raise SystemExit('VFB JRC2018Unisex template contains no raster payload')
print(f'VFB JRC2018Unisex template ready: {p} ({p.stat().st_size} bytes)')
print(header)
PY
