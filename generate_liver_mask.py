"""
generate_liver_mask.py

Creates a synthetic liver blood-demand map and Couinaud segment label map
as NIfTI files. Use these as inputs to Whitehead's vessel generator when
real liver segmentations (e.g. XCAT) are not available.

Couinaud segments (simplified spatial layout):
  Segment 1  - caudate lobe (posterior center)
  Segments 2,3 - left lateral (left side)
  Segment 4  - left medial
  Segments 5,8 - right anterior
  Segments 6,7 - right posterior

Output files:
  liver_mask.nii.gz        - binary liver mask
  blood_demand.nii.gz      - uniform blood demand map (0-1, zero outside liver)
  couinaud_segments.nii.gz - integer label map (0=background, 1-8=segments)
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def make_ellipsoid(shape, center, radii):
    """Return boolean ellipsoid mask."""
    Z, Y, X = np.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (
        ((Z - center[0]) / radii[0]) ** 2 +
        ((Y - center[1]) / radii[1]) ** 2 +
        ((X - center[2]) / radii[2]) ** 2
    ) <= 1.0
    return mask


def generate_liver_mask(
    shape=(200, 180, 250),   # (Z, Y, X) voxels
    voxel_size_mm=1.0,       # isotropic
    output_dir="."
):
    Z, Y, X = shape
    cz, cy, cx = Z // 2, Y // 2, X // 2

    # --- Main liver ellipsoid ---
    liver = make_ellipsoid(shape, (cz, cy, cx), (Z * 0.42, Y * 0.44, X * 0.46))

    # --- Couinaud segments (8 approximate sub-regions) ---
    seg_map = np.zeros(shape, dtype=np.uint8)

    # Divide liver roughly into left (x < cx) and right (x >= cx) lobes
    # and anterior (y < cy) / posterior (y >= cy) quadrants
    left  = liver & (np.arange(X)[None, None, :] < cx)
    right = liver & (np.arange(X)[None, None, :] >= cx)
    ant   = np.arange(Y)[None, :, None] < cy
    post  = ~ant
    sup   = np.arange(Z)[:, None, None] >= cz
    inf   = ~sup

    # Segment 1: caudate (posterior, central, superior)
    seg_map[liver & post & sup &
            (np.abs(np.arange(X)[None, None, :] - cx) < X * 0.12)] = 1

    # Left lobe
    seg_map[left & sup & post] = 2   # Segment 2
    seg_map[left & inf & post] = 3   # Segment 3
    seg_map[left & ant]        = 4   # Segment 4

    # Right lobe (anterior)
    seg_map[right & ant & inf] = 5   # Segment 5
    seg_map[right & ant & sup] = 8   # Segment 8

    # Right lobe (posterior)
    seg_map[right & post & inf] = 6  # Segment 6
    seg_map[right & post & sup] = 7  # Segment 7

    # Fill any unlabeled liver voxels with nearest-neighbor (simple fallback)
    unlabeled = liver & (seg_map == 0)
    if unlabeled.any():
        seg_map[unlabeled] = 5  # assign to segment 5 as fallback

    # --- Blood demand: uniform inside liver with slight Gaussian variation ---
    demand = liver.astype(np.float32)
    noise = gaussian_filter(np.random.rand(*shape).astype(np.float32), sigma=8)
    demand = demand * (0.7 + 0.3 * (noise / noise.max()))
    demand[~liver] = 0.0

    # --- Save NIfTI ---
    affine = np.diag([voxel_size_mm] * 3 + [1.0])

    nib.save(nib.Nifti1Image(liver.astype(np.uint8), affine),
             f"{output_dir}/liver_mask.nii.gz")
    nib.save(nib.Nifti1Image(demand, affine),
             f"{output_dir}/blood_demand.nii.gz")
    nib.save(nib.Nifti1Image(seg_map, affine),
             f"{output_dir}/couinaud_segments.nii.gz")

    print(f"Liver voxels: {liver.sum()}")
    for s in range(1, 9):
        print(f"  Segment {s}: {(seg_map == s).sum()} voxels")

    return liver, demand, seg_map


if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=3, type=int, default=[200, 180, 250],
                   metavar=("Z", "Y", "X"))
    p.add_argument("--voxel_size", type=float, default=1.0)
    p.add_argument("--out", default="liver_inputs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    generate_liver_mask(tuple(args.shape), args.voxel_size, args.out)
    print(f"Saved to {args.out}/")
