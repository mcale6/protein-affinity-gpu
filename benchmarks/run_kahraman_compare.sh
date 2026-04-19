#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="${ROOT_DIR}/.venv/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
MANIFEST="${MANIFEST:-${ROOT_DIR}/benchmarks/datasets/kahraman_2013_t3.tsv}"
STRUCTURES_DIR="${STRUCTURES_DIR:-${ROOT_DIR}/benchmarks/downloads/kahraman_2013_t3}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmarks/output/kahraman_2013_t3}"
REPEATS="${REPEATS:-3}"

download_file() {
  local url="$1"
  local destination="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${destination}"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -qO "${destination}" "${url}"
    return
  fi

  echo "Neither curl nor wget is available for downloads." >&2
  exit 1
}

download_structure() {
  local pdb_id="$1"
  local tmp_file
  tmp_file="$(mktemp)"

  if download_file "https://files.rcsb.org/download/${pdb_id}.pdb" "${tmp_file}"; then
    mv "${tmp_file}" "${STRUCTURES_DIR}/${pdb_id}.pdb"
    return
  fi

  if download_file "https://files.rcsb.org/download/${pdb_id}.cif" "${tmp_file}"; then
    mv "${tmp_file}" "${STRUCTURES_DIR}/${pdb_id}.cif"
    return
  fi

  rm -f "${tmp_file}"
  echo "Failed to download ${pdb_id} from RCSB in PDB or mmCIF format." >&2
  exit 1
}

mkdir -p "${STRUCTURES_DIR}" "${OUTPUT_DIR}"

while IFS=$'\t' read -r pdb_id _rest; do
  if [[ -z "${pdb_id}" || "${pdb_id}" == "pdb_id" ]]; then
    continue
  fi

  pdb_id="$(printf '%s' "${pdb_id}" | tr '[:lower:]' '[:upper:]')"
  if [[ -f "${STRUCTURES_DIR}/${pdb_id}.pdb" || -f "${STRUCTURES_DIR}/${pdb_id}.cif" ]]; then
    continue
  fi

  echo "Fetching ${pdb_id}"
  download_structure "${pdb_id}"
done < "${MANIFEST}"

cmd=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/benchmarks/compare.py"
  --manifest "${MANIFEST}"
  --structures-dir "${STRUCTURES_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --repeats "${REPEATS}"
)

if [[ "${REQUIRE_GPU:-0}" == "1" ]]; then
  cmd+=(--require-gpu)
fi

cmd+=("$@")
"${cmd[@]}"
