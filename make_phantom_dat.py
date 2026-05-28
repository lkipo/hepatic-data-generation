"""
Build a Whitehead-compatible Phantom.dat from local NIfTI liver inputs.

The original C++ generator reads a flat uint8 array with dimensions
(Nx, Ny, Nz, 10) in column-major order. Layers 0-7 are Couinaud segment
blood-demand maps, layer 8 is used by the C++ initial-root sampling, and
layer 9 is the whole-liver mask.
"""

from __future__ import annotations

import argparse
import os

import nibabel as nib
import numpy as np


def load_zyx(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata())


def write_phantom_dat(liver_mask_path: str, segment_map_path: str,
                      blood_demand_path: str, out_path: str) -> tuple[int, int, int]:
    liver = load_zyx(liver_mask_path) > 0
    segment = load_zyx(segment_map_path).astype(np.int16)
    demand = load_zyx(blood_demand_path).astype(np.float32)

    if liver.shape != segment.shape or liver.shape != demand.shape:
        raise ValueError("liver_mask, segment_map, and blood_demand must have the same shape")

    if demand.max() > 0:
        demand = demand / float(demand.max())
    demand = np.clip(demand, 0.0, 1.0)

    z_size, y_size, x_size = liver.shape
    phantom = np.zeros((x_size, y_size, z_size, 10), dtype=np.uint8, order="F")
    demand_u8 = np.rint(demand * 255.0).astype(np.uint8)

    liver_xyz = np.transpose(liver, (2, 1, 0))
    segment_xyz = np.transpose(segment, (2, 1, 0))
    demand_xyz = np.transpose(demand_u8, (2, 1, 0))

    for seg_id in range(1, 9):
        phantom[..., seg_id - 1] = np.where(
            liver_xyz & (segment_xyz == seg_id),
            demand_xyz,
            0,
        ).astype(np.uint8)
    phantom[..., 8] = np.where(liver_xyz, demand_xyz, 0).astype(np.uint8)
    phantom[..., 9] = np.where(liver_xyz, 255, 0).astype(np.uint8)

    missing_layers = [
        str(layer)
        for layer in range(10)
        if np.count_nonzero(phantom[..., layer]) == 0
    ]
    if missing_layers:
        raise ValueError(
            "Cannot build usable Phantom.dat; empty layer(s): "
            + ", ".join(missing_layers)
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    phantom.ravel(order="F").tofile(out_path)
    return x_size, y_size, z_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--liver_mask", required=True)
    parser.add_argument("--segment_map", required=True)
    parser.add_argument("--blood_demand", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    nx, ny, nz = write_phantom_dat(
        args.liver_mask,
        args.segment_map,
        args.blood_demand,
        args.out,
    )
    print(f"Wrote {args.out}")
    print(f"Dimensions for generator: --Nx {nx} --Ny {ny} --Nz {nz}")


if __name__ == "__main__":
    main()
