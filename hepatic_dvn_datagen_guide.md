# Hepatic Synthetic Data Generation for DeepVesselNet

## Overview

This pipeline replaces DeepVesselNet's brain-derived synthetic data with hepatic arterial trees
from Whitehead et al. (2023), producing the same 7-channel format DVN expects:

| Channel | Description |
|---|---|
| `raw` | 3D intensity volume (voxelized vessels + background) |
| `seg` | Binary vessel segmentation |
| `centerline` | Binary centerline mask |
| `points` | Edge-count per node (≥3 = bifurcation) |
| `bifurcation` | Block mask around bifurcation points |
| `radius` | Radius along centerlines |
| `TreeN.txt`, `RadiiN.txt` | Whitehead centerline tree and branch radii |

---

## Step 1 — Generate a Whitehead-compatible phantom

`generate_liver_mask.py` creates NIfTI liver inputs. `make_phantom_dat.py` then
converts those inputs into the `Phantom.dat` layout used by Whitehead's C++
generator: `(Nx, Ny, Nz, 10)` uint8, column-major order.

```bash
python datagen_pipeline/generate_liver_mask.py --out datagen_pipeline/liver_inputs
python datagen_pipeline/make_phantom_dat.py \
  --liver_mask datagen_pipeline/liver_inputs/liver_mask.nii.gz \
  --segment_map datagen_pipeline/liver_inputs/couinaud_segments.nii.gz \
  --blood_demand datagen_pipeline/liver_inputs/blood_demand.nii.gz \
  --out datagen_pipeline/liver_inputs/Phantom.dat
```

---

## Step 2 — Generate vessels with the C++-faithful Python port

Use `hepatic_vessel_generation.py`. It mirrors
`HepaticVesselGeneration/Vessel_sim_CPU/main.cpp` and writes the same text format:
`TreeN.txt` and `RadiiN.txt`.

Progress is printed during tree growth. By default the generator reports about
every 1% of connected endpoints; set `PROGRESS_INTERVAL=25` when running
`run_pipeline.sh` to report every 25 connected endpoints.

Multiple samples can run in parallel with `MAX_JOBS`, for example:

```bash
MAX_JOBS=4 bash datagen_pipeline/run_pipeline.sh 12
```

---

## Step 3 — Convert tree text → DVN volumetric channels

Use `tree_txt_to_dvn_volumes.py` to rasterize `TreeN.txt`/`RadiiN.txt` into
`seg`, `centerline`, `points`, `bifurcation`, and `radius` volumes.

---

## Step 4 — Simulate raw intensity volume

Use `simulate_raw_volume_python.py` to render vessels into a 3D intensity volume
with Gaussian vessel profiles + noise, mimicking CT/MRA appearance.

---

## File layout

```
hepatic_dvn_data/
  generate_liver_mask.py        # Step 1
  make_phantom_dat.py           # Step 1
  hepatic_vessel_generation.py  # Step 2
  tree_txt_to_dvn_volumes.py    # Step 3
  simulate_raw_volume_python.py # Step 4
  run_pipeline.sh               # Orchestrates all steps
  requirements.txt
```

---

## Citation

```
Whitehead JF, et al. In silico simulation of Hepatic arteries.
Med Phys. 2023. https://doi.org/10.1002/mp.16379

Tetteh G, et al. DeepVesselNet. Front Neurosci. 2020.
https://doi.org/10.3389/fnins.2020.592352
```
