#!/usr/bin/env bash
set -euo pipefail

# Run the Gauthey a2_r5 -> FDA registration in BIFROST's published container
# environment, without installing BIFROST/ANTs/TensorFlow into the host venv.
#
# BIFROST source: ClandininLab/bifrost, commit
# 01d387975d5d0aebdf49b7eb31efcec1a2fc081f
# Paper: Brezovec et al., PNAS 2024, DOI 10.1073/pnas.2322687121

BIFROST_COMMIT="${BIFROST_COMMIT:-01d387975d5d0aebdf49b7eb31efcec1a2fc081f}"
IMAGE="${BIFROST_IMAGE:-dashi-bifrost:${BIFROST_COMMIT:0:12}}"
ENGINE="${CONTAINER_ENGINE:-}"

if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then
    ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then
    ENGINE=podman
  else
    echo "Neither docker nor podman is available; cannot run BIFROST container." >&2
    exit 2
  fi
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if ! "$ENGINE" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "+ building $IMAGE from ClandininLab/bifrost@$BIFROST_COMMIT" >&2
  "$ENGINE" build \
    --tag "$IMAGE" \
    "https://github.com/ClandininLab/bifrost.git#${BIFROST_COMMIT}"
fi

# The BIFROST Dockerfile installs the package and its ANTs/TensorFlow dependencies.
# Mount the repository at a stable path so all relative data paths resolve inside
# the container exactly as they do on the host.
exec "$ENGINE" run --rm \
  --volume "$ROOT:/work" \
  --workdir /work \
  --env PYTHONPATH=/work \
  "$IMAGE" \
  python scripts/register_gauthey_lbm_a2_r5_to_fda.py \
    --bifrost-bin bifrost \
    "$@"
