#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EX_DIR="${ROOT_DIR}/sorted_exhaustive_32_8_prefix"
INPUT_DIR="${ROOT_DIR}/dataset_csr"
OUTPUT_CSV="${ROOT_DIR}/fine_grained_results.csv"

unset JACCARD_FORCE_VERTEX_CENTRIC
export JACCARD_FORCE_EDGE_CENTRIC=1

if [[ ! -x "${EX_DIR}/jaccardCUDA" ]]; then
  echo "Missing ${EX_DIR}/jaccardCUDA; run make in ${EX_DIR} first." >&2
  exit 1
fi

if (( $# > 0 )); then
  GRAPHS=("$@")
else
  GRAPHS=(roadNet-CA)
  # GRAPHS=(mawi_201512020330 roadNet-CA circuit5M)
fi

TMP_OUTPUT="$(mktemp)"
trap 'rm -f "${TMP_OUTPUT}"' EXIT
header_written=0
: > "${OUTPUT_CSV}"

for graph in "${GRAPHS[@]}"; do
  input="${INPUT_DIR}/${graph}.csr"
  if [[ ! -f "${input}" ]]; then
    echo "Missing input: ${input}" >&2
    exit 1
  fi

  echo "Running ${graph}..." >&2
  "${EX_DIR}/jaccardCUDA" "${input}" "${EX_DIR}/out.csr" > "${TMP_OUTPUT}"

  if (( header_written == 0 )); then
    printf 'graph,%s\n' "$(head -n 1 "${TMP_OUTPUT}" | tr -d '\r')" >> "${OUTPUT_CSV}"
    header_written=1
  fi

  tail -n +2 "${TMP_OUTPUT}" | sed "/^[[:space:]]*$/d; s/^/${graph},/" >> "${OUTPUT_CSV}"
done

cat "${OUTPUT_CSV}"
