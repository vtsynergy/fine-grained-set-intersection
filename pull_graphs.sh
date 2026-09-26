#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${ROOT_DIR}/dataset_mtx"
mkdir -p "${OUTPUT_DIR}"

declare -A URLS=(
  [roadNet-CA]="https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-CA.tar.gz"
  # [mawi_201512020330]="https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz"
  # [circuit5M]="https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz"
)

if (( $# > 0 )); then
  GRAPHS=("$@")
else
  GRAPHS=(roadNet-CA)
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

for graph in "${GRAPHS[@]}"; do
  if [[ -z "${URLS[$graph]+x}" ]]; then
    echo "Unknown graph: ${graph}" >&2
    exit 1
  fi

  archive="${TMP_DIR}/${graph}.tar.gz"
  extract_dir="${TMP_DIR}/${graph}"
  mkdir -p "${extract_dir}"

  echo "Downloading ${graph}..."
  wget -q --show-progress -O "${archive}" "${URLS[$graph]}"
  tar -xzf "${archive}" -C "${extract_dir}"

  mtx_file="$(find "${extract_dir}" -type f -name "${graph}.mtx" -print -quit)"
  if [[ -z "${mtx_file}" ]]; then
    echo "Could not find ${graph}.mtx in ${archive}" >&2
    exit 1
  fi

  cp "${mtx_file}" "${OUTPUT_DIR}/${graph}.mtx"
  echo "Stored ${OUTPUT_DIR}/${graph}.mtx"
done
