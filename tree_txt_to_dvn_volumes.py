"""
Rasterize Whitehead TreeN.txt/RadiiN.txt outputs into DVN NIfTI volumes.

Tree coordinates are x y z voxel coordinates in the Phantom.dat grid. Radii
are stored in phantom voxels by the original generator.
"""

from __future__ import annotations

import argparse
import math
import os

import nibabel as nib
import numpy as np


def parse_tree(tree_path: str) -> list[np.ndarray]:
    branches: list[np.ndarray] = []
    with open(tree_path, "r") as handle:
        for line in handle:
            values = [float(v) for v in line.split()]
            if not values:
                branches.append(np.zeros((0, 3), dtype=np.float32))
                continue
            if len(values) % 3:
                raise ValueError(f"Line in {tree_path} does not contain x y z triplets")
            branches.append(np.asarray(values, dtype=np.float32).reshape(-1, 3))
    return branches


def parse_radii(radii_path: str) -> np.ndarray:
    with open(radii_path, "r") as handle:
        return np.asarray([float(line.strip()) for line in handle if line.strip()], dtype=np.float32)


def draw_sphere(seg, center_zyx, radius_vox):
    radius = int(math.ceil(radius_vox))
    z0, y0, x0 = center_zyx
    z_size, y_size, x_size = seg.shape
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz * dz + dy * dy + dx * dx <= radius_vox * radius_vox:
                    z, y, x = z0 + dz, y0 + dy, x0 + dx
                    if 0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size:
                        seg[z, y, x] = 1


def rasterize_tree(tree_path: str, radii_path: str, shape_zyx: tuple[int, int, int],
                   phantom_res: float, voxel_size: float, out_dir: str):
    branches = parse_tree(tree_path)
    radii = parse_radii(radii_path)
    if len(radii) < len(branches):
        raise ValueError("Radii file has fewer entries than the tree file")

    os.makedirs(out_dir, exist_ok=True)
    seg = np.zeros(shape_zyx, dtype=np.uint8)
    centerline = np.zeros(shape_zyx, dtype=np.uint8)
    points = np.zeros(shape_zyx, dtype=np.uint8)
    bifurcation = np.zeros(shape_zyx, dtype=np.uint8)
    radius_vol = np.zeros(shape_zyx, dtype=np.float32)

    scale = phantom_res / voxel_size

    endpoint_counts: dict[tuple[int, int, int], int] = {}
    for branch, radius_phantom in zip(branches, radii):
        if len(branch) == 0:
            continue
        radius_vox = max(float(radius_phantom) * scale, 0.5)
        radius_mm = float(radius_phantom) * phantom_res
        pts_zyx = np.rint(branch[:, [2, 1, 0]] * scale).astype(int)

        for pt in pts_zyx:
            z, y, x = pt
            if 0 <= z < shape_zyx[0] and 0 <= y < shape_zyx[1] and 0 <= x < shape_zyx[2]:
                centerline[z, y, x] = 1
                radius_vol[z, y, x] = max(radius_vol[z, y, x], radius_mm)
                draw_sphere(seg, (z, y, x), radius_vox)

        for pt in (pts_zyx[0], pts_zyx[-1]):
            z, y, x = pt
            if 0 <= z < shape_zyx[0] and 0 <= y < shape_zyx[1] and 0 <= x < shape_zyx[2]:
                key = (int(z), int(y), int(x))
                endpoint_counts[key] = endpoint_counts.get(key, 0) + 1

    for (z, y, x), count in endpoint_counts.items():
        points[z, y, x] = max(points[z, y, x], min(count, 255))
        if count >= 3:
            z0, z1 = max(z - 3, 0), min(z + 4, shape_zyx[0])
            y0, y1 = max(y - 3, 0), min(y + 4, shape_zyx[1])
            x0, x1 = max(x - 3, 0), min(x + 4, shape_zyx[2])
            bifurcation[z0:z1, y0:y1, x0:x1] = 1

    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    nib.save(nib.Nifti1Image(seg, affine), os.path.join(out_dir, "seg.nii.gz"))
    nib.save(nib.Nifti1Image(centerline, affine), os.path.join(out_dir, "centerline.nii.gz"))
    nib.save(nib.Nifti1Image(points, affine), os.path.join(out_dir, "points.nii.gz"))
    nib.save(nib.Nifti1Image(bifurcation, affine), os.path.join(out_dir, "bifurcation.nii.gz"))
    nib.save(nib.Nifti1Image(radius_vol, affine), os.path.join(out_dir, "radius.nii.gz"))
    print(f"Saved DVN volumes to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--radii", required=True)
    parser.add_argument("--shape", nargs=3, type=int, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--phantom_res", type=float, default=1.0)
    parser.add_argument("--voxel_size", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rasterize_tree(
        args.tree,
        args.radii,
        tuple(args.shape),
        args.phantom_res,
        args.voxel_size,
        args.out,
    )


if __name__ == "__main__":
    main()
