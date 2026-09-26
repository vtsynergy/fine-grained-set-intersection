#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="${ROOT_DIR}/sorted_exhaustive_32_8_prefix"
INPUT_DIR="${ROOT_DIR}/dataset_mtx"
OUTPUT_DIR="${ROOT_DIR}/dataset_csr"

unset JACCARD_FORCE_VERTEX_CENTRIC
export JACCARD_FORCE_EDGE_CENTRIC=1

if [[ ! -x "${EX_DIR}/fileConvert" ]]; then
  echo "Missing ${EX_DIR}/fileConvert; run make in ${EX_DIR} first." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if (( $# > 0 )); then
  GRAPHS=("$@")
else
  GRAPHS=(roadNet-CA)
  # GRAPHS=(mawi_201512020330 roadNet-CA circuit5M)
fi

for file in "${GRAPHS[@]}"; do
  input="${INPUT_DIR}/${file}.mtx"
  output="${OUTPUT_DIR}/${file}.csr"
  if [[ ! -f "${input}" ]]; then
    echo "Missing input: ${input}" >&2
    exit 1
  fi

  echo "Converting ${file}..."
  CONVERT_FORCE_DROP_WEIGHTS=1 \
  CONVERT_FORCE_REVERSE=1 \
    "${EX_DIR}/fileConvert" "${input}" "${output}" 1
done
