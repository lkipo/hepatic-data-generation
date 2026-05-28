"""
simulate_raw_volume.py

Given the binary vessel segmentation (seg.nii.gz) and radius volume
(radius.nii.gz), produces a raw.nii.gz intensity volume that mimics
a CT/MRA scan — the 'raw' channel expected by DeepVesselNet.

Model:
  - Background: Gaussian noise around a liver-tissue HU baseline
  - Vessels:    Bright tubular structures with Gaussian cross-section
                profiles (vessel center = peak_intensity, falls off
                with distance)
  - Optional:   Additive Poisson-like noise to simulate acquisition

Usage:
  python simulate_raw_volume.py \\
      --seg sample_001/seg.nii.gz \\
      --radius sample_001/radius.nii.gz \\
      --out sample_001/raw.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, gaussian_filter


def simulate_raw(
    seg_path: str,
    radius_path: str,
    out_path: str,
    # Intensity parameters (HU-like for CT; adjust for MRA)
    background_mean: float = 60.0,
    background_std: float = 15.0,
    vessel_peak: float = 280.0,   # iodine-enhanced vessel HU
    vessel_sigma_factor: float = 0.5,  # Gaussian σ = factor × radius
    noise_std: float = 10.0,
    blur_sigma: float = 0.5,      # final PSF blur
    dtype=np.float32,
):
    """
    Parameters
    ----------
    seg_path    : path to binary vessel segmentation NIfTI
    radius_path : path to radius-along-centerline NIfTI (mm)
    out_path    : where to write raw intensity NIfTI
    """
    seg_img = nib.load(seg_path)
    rad_img = nib.load(radius_path)
    affine  = seg_img.affine
    voxel_size = float(np.abs(affine[0, 0]))  # assume isotropic

    seg = seg_img.get_fdata().astype(bool)
    rad = rad_img.get_fdata().astype(np.float32)   # mm

    # --- Background ---
    raw = np.random.normal(background_mean, background_std, seg.shape).astype(dtype)

    # --- Distance from vessel surface (inside vessels = negative EDT convention) ---
    # EDT from the outside of the vessel mask gives distance-to-vessel for bg voxels
    # EDT inside the vessel mask gives distance-to-surface for vessel voxels
    dist_outside = distance_transform_edt(~seg) * voxel_size   # mm, bg voxels
    dist_inside  = distance_transform_edt(seg)  * voxel_size   # mm, vessel voxels

    # Smooth radius field so every vessel voxel has a local radius estimate
    rad_smooth = np.where(seg, rad, 0.0)
    # Propagate radius inward (max of nearby centerline radii) — simple dilation approx
    from scipy.ndimage import maximum_filter
    rad_vessel = maximum_filter(rad_smooth, size=7)
    rad_vessel = np.where(seg, rad_vessel, 1.0)  # avoid div by zero

    # Gaussian vessel profile: I = peak * exp(-d² / (2σ²))
    sigma_mm = vessel_sigma_factor * rad_vessel
    sigma_mm = np.clip(sigma_mm, 0.1, None)

    vessel_intensity = vessel_peak * np.exp(
        -(dist_inside ** 2) / (2.0 * sigma_mm ** 2)
    )

    # Blend vessel signal over background
    raw = np.where(seg, raw + vessel_intensity, raw)

    # Soft halo just outside vessels (partial volume effect)
    halo_range_mm = voxel_size * 1.5
    halo_mask = (dist_outside > 0) & (dist_outside < halo_range_mm)
    halo_intensity = (vessel_peak * 0.3) * np.exp(
        -(dist_outside[halo_mask] ** 2) / (2.0 * (halo_range_mm * 0.5) ** 2)
    )
    raw[halo_mask] += halo_intensity

    # --- Acquisition noise ---
    raw += np.random.normal(0, noise_std, raw.shape).astype(dtype)

    # --- PSF blur ---
    if blur_sigma > 0:
        raw = gaussian_filter(raw, sigma=blur_sigma / voxel_size)

    # --- Clip to realistic HU range ---
    raw = np.clip(raw, -200, 600).astype(dtype)

    # --- Save ---
    nib.save(nib.Nifti1Image(raw, affine), out_path)
    print(f"Saved raw volume → {out_path}")
    print(f"  shape: {raw.shape}, range: [{raw.min():.1f}, {raw.max():.1f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seg",    required=True, help="seg.nii.gz")
    p.add_argument("--radius", required=True, help="radius.nii.gz")
    p.add_argument("--out",    required=True, help="raw.nii.gz output path")
    p.add_argument("--bg_mean",  type=float, default=60.0)
    p.add_argument("--bg_std",   type=float, default=15.0)
    p.add_argument("--peak",     type=float, default=280.0)
    p.add_argument("--noise",    type=float, default=10.0)
    p.add_argument("--blur",     type=float, default=0.5)
    args = p.parse_args()

    simulate_raw(
        seg_path=args.seg,
        radius_path=args.radius,
        out_path=args.out,
        background_mean=args.bg_mean,
        background_std=args.bg_std,
        vessel_peak=args.peak,
        noise_std=args.noise,
        blur_sigma=args.blur,
    )
