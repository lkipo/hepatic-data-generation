#!/usr/bin/env bash
# run_pipeline.sh
#
# Full local pipeline: liver mask -> C++-faithful Python vessel generation -> raw intensity
# Produces N samples compatible with DeepVesselNet training format.
#
# Prerequisites:
#   Python env:
#        python3 -m pip install -r datagen_pipeline/requirements.txt
#
# Usage:
#   bash datagen_pipeline/run_pipeline.sh [N_SAMPLES] [OUT_DIR]
#   MAX_JOBS=4 bash datagen_pipeline/run_pipeline.sh [N_SAMPLES] [OUT_DIR]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_SAMPLES=${1:-10}
OUT_DIR=${2:-"hepatic_dvn_dataset"}
OUT_DIR="$(mkdir -p "${OUT_DIR}" && cd "${OUT_DIR}" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/liver_inputs"

PYTHON=${PYTHON:-python3}
SHAPE_Z=${SHAPE_Z:-120}
SHAPE_Y=${SHAPE_Y:-100}
SHAPE_X=${SHAPE_X:-150}
VOXEL_SIZE=${VOXEL_SIZE:-1.0}
N_ENDPOINTS=${N_ENDPOINTS:-2000}
PHANTOM_RES=${PHANTOM_RES:-${VOXEL_SIZE}}
DESIRED_RES=${DESIRED_RES:-${VOXEL_SIZE}}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-0}
MAX_JOBS=${MAX_JOBS:-1}

if ! [[ "${MAX_JOBS}" =~ ^[0-9]+$ ]] || [ "${MAX_JOBS}" -lt 1 ]; then
    echo "MAX_JOBS must be a positive integer." >&2
    exit 1
fi

mkdir -p "${INPUT_DIR}" "${OUT_DIR}"

if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import nibabel
import numpy
import scipy
PY
then
    echo "Missing Python dependencies for ${PYTHON}." >&2
    echo "Install them with: ${PYTHON} -m pip install -r ${SCRIPT_DIR}/requirements.txt" >&2
    echo "Or choose another environment with: PYTHON=/path/to/python bash ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# Step 1: Generate liver mask once (reuse across samples or regenerate)
echo "=== Generating liver mask ==="
STEP_START=${SECONDS}
"${PYTHON}" "${SCRIPT_DIR}/generate_liver_mask.py" \
    --shape ${SHAPE_Z} ${SHAPE_Y} ${SHAPE_X} \
    --voxel_size ${VOXEL_SIZE} \
    --out "${INPUT_DIR}"
echo "    mask complete in $((SECONDS - STEP_START))s"

STEP_START=${SECONDS}
"${PYTHON}" "${SCRIPT_DIR}/make_phantom_dat.py" \
    --liver_mask "${INPUT_DIR}/liver_mask.nii.gz" \
    --segment_map "${INPUT_DIR}/couinaud_segments.nii.gz" \
    --blood_demand "${INPUT_DIR}/blood_demand.nii.gz" \
    --out "${INPUT_DIR}/Phantom.dat"
echo "    Phantom.dat complete in $((SECONDS - STEP_START))s"

echo "=== Generating ${N_SAMPLES} hepatic vessel samples ==="
echo "    parallel sample jobs: ${MAX_JOBS}"

run_sample() {
    local i="$1"
    SAMPLE_DIR="${OUT_DIR}/sample_${i}"
    mkdir -p "${SAMPLE_DIR}"

    echo "--- Sample ${i}/${N_SAMPLES} ---"
    SAMPLE_START=${SECONDS}

    # Step 2: Run the local C++-faithful generator. It writes Tree1.txt and
    # Radii1.txt, matching Whitehead's original text output format.
    SEED=$("${PYTHON}" -c "import random; print(random.randrange(1, 2**31))")

    "${PYTHON}" "${SCRIPT_DIR}/hepatic_vessel_generation.py" \
        --phantom "${INPUT_DIR}/Phantom.dat" \
        --Nx ${SHAPE_X} \
        --Ny ${SHAPE_Y} \
        --Nz ${SHAPE_Z} \
        --endpoints ${N_ENDPOINTS} \
        --trees 1 \
        --phantom_res ${PHANTOM_RES} \
        --desired_res ${DESIRED_RES} \
        --out "${SAMPLE_DIR}" \
        --seed ${SEED} \
        --progress_interval ${PROGRESS_INTERVAL}
    echo "    vessel tree complete in $((SECONDS - SAMPLE_START))s"

    STEP_START=${SECONDS}
    "${PYTHON}" "${SCRIPT_DIR}/tree_txt_to_dvn_volumes.py" \
        --tree "${SAMPLE_DIR}/Tree1.txt" \
        --radii "${SAMPLE_DIR}/Radii1.txt" \
        --shape ${SHAPE_Z} ${SHAPE_Y} ${SHAPE_X} \
        --phantom_res ${PHANTOM_RES} \
        --voxel_size ${VOXEL_SIZE} \
        --out "${SAMPLE_DIR}"
    echo "    rasterization complete in $((SECONDS - STEP_START))s"

    # Step 3: Simulate the raw channel from the rasterized sparse centreline radii.
    STEP_START=${SECONDS}
    "${PYTHON}" "${SCRIPT_DIR}/simulate_raw_volume_python.py" \
        --input_dir "${SAMPLE_DIR}" \
        --out "${SAMPLE_DIR}/raw.nii.gz" \
        --seed ${SEED}
    echo "    raw simulation complete in $((SECONDS - STEP_START))s"

    echo "    -> ${SAMPLE_DIR}/ complete in $((SECONDS - SAMPLE_START))s"
}

pids=""
for i in $(seq -w 1 ${N_SAMPLES}); do
    while [ "$(jobs -pr | wc -l | tr -d ' ')" -ge "${MAX_JOBS}" ]; do
        sleep 1
    done
    run_sample "${i}" &
    pids="${pids} $!"
done

status=0
for pid in ${pids}; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [ "${status}" -ne 0 ]; then
    echo "One or more sample generation jobs failed." >&2
    exit "${status}"
fi

echo ""
echo "=== Done. Dataset at: ${OUT_DIR}/ ==="
echo "Each sample contains:"
echo "  raw.nii.gz, seg.nii.gz, centerline.nii.gz,"
echo "  points.nii.gz, bifurcation.nii.gz, radius.nii.gz, Tree1.txt, Radii1.txt"
