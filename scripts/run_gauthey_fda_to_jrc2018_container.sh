#!/usr/bin/env bash
set -euo pipefail

BIFROST_COMMIT="${BIFROST_COMMIT:-01d387975d5d0aebdf49b7eb31efcec1a2fc081f}"
IMAGE="${BIFROST_IMAGE:-dashi-bifrost:${BIFROST_COMMIT:0:12}}"
ENGINE="${CONTAINER_ENGINE:-}"

if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then
    ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then
    ENGINE=podman
  else
    echo "Neither docker nor podman is available." >&2
    exit 2
  fi
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if ! "$ENGINE" image inspect "$IMAGE" >/dev/null 2>&1; then
  "$ENGINE" build --tag "$IMAGE" "https://github.com/ClandininLab/bifrost.git#${BIFROST_COMMIT}"
fi

exec "$ENGINE" run --rm \
  --net=host \
  --volume "$ROOT:/work" \
  --workdir /work \
  --env PYTHONPATH=/work \
  "$IMAGE" \
  python scripts/register_gauthey_fda_labels_to_jrc2018.py \
    --bifrost-bin bifrost \
    "$@"
