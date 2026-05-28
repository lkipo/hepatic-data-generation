"""
simulate_raw_volume_python.py

Raw-volume simulator for outputs produced by hepatic_vessel_generation.py.

The Python vessel generator writes:
  seg.nii.gz         binary vessel mask
  centerline.nii.gz  binary vessel centreline
  radius.nii.gz      radius values on centreline voxels

Unlike simulate_raw_volume.py, this version treats radius.nii.gz as a sparse
centreline field and propagates each centreline radius to nearby vessel voxels
using nearest-centreline indices. This gives a more coherent tubular intensity
profile for the pure-Python generator output.

Usage:
  python3 simulate_raw_volume_python.py \
      --input_dir datagen_pipeline/hepatic_10k \
      --out datagen_pipeline/hepatic_10k/raw_python.nii.gz

or:
  python3 simulate_raw_volume_python.py \
      --seg datagen_pipeline/hepatic_10k/seg.nii.gz \
      --centerline datagen_pipeline/hepatic_10k/centerline.nii.gz \
      --radius datagen_pipeline/hepatic_10k/radius.nii.gz \
      --out datagen_pipeline/hepatic_10k/raw_python.nii.gz
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _voxel_spacing_mm(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    zooms = img.header.get_zooms()[:3]
    return tuple(float(abs(z)) if z else 1.0 for z in zooms)


def _load_bool(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    return img.get_fdata().astype(bool), img


def _resolve_paths(
    input_dir: Optional[str],
    seg: Optional[str],
    centerline: Optional[str],
    radius: Optional[str],
) -> Tuple[str, str, str]:
    if input_dir:
        seg = seg or os.path.join(input_dir, "seg.nii.gz")
        centerline = centerline or os.path.join(input_dir, "centerline.nii.gz")
        radius = radius or os.path.join(input_dir, "radius.nii.gz")

    missing = [
        name for name, path in (
            ("--seg", seg),
            ("--centerline", centerline),
            ("--radius", radius),
        )
        if not path
    ]
    if missing:
        raise SystemExit(
            "Missing required input(s): "
            + ", ".join(missing)
            + ". Pass --input_dir or explicit file paths."
        )

    for path in (seg, centerline, radius):
        if not os.path.exists(path):
            raise SystemExit(f"Input file does not exist: {path}")

    return seg, centerline, radius


def _nearest_centerline_radius(
    seg: np.ndarray,
    centerline: np.ndarray,
    radius: np.ndarray,
    spacing: Sequence[float],
    default_radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    centerline = centerline & seg
    centerline = centerline | (radius > 0)

    if not centerline.any():
        raise ValueError(
            "No centreline voxels found. Expected centerline.nii.gz or positive "
            "values in radius.nii.gz from hepatic_vessel_generation.py."
        )

    # For every voxel, get distance to and indices of the nearest centreline
    # voxel. Radius is sparse, so nearest-neighbour propagation is preferable to
    # a fixed-size maximum filter for large generated vessels.
    dist_to_centerline, nearest = distance_transform_edt(
        ~centerline,
        sampling=spacing,
        return_indices=True,
    )
    nearest_radius = radius[tuple(nearest)]
    nearest_radius = np.where(nearest_radius > 0, nearest_radius, default_radius_mm)
    nearest_radius = np.where(seg, nearest_radius, 0.0).astype(np.float32)
    return dist_to_centerline.astype(np.float32), nearest_radius


def simulate_raw_for_python_generator(
    seg_path: str,
    centerline_path: str,
    radius_path: str,
    out_path: str,
    background_mean: float = 60.0,
    background_std: float = 18.0,
    vessel_peak: float = 330.0,
    radius_sigma_factor: float = 0.45,
    edge_softening_mm: float = 0.35,
    halo_fraction: float = 0.18,
    halo_sigma_mm: float = 0.75,
    noise_std: float = 12.0,
    blur_sigma_mm: float = 0.45,
    seed: Optional[int] = None,
) -> np.ndarray:
    seg, seg_img = _load_bool(seg_path)
    centerline, _ = _load_bool(centerline_path)
    rad_img = nib.load(radius_path)
    radius = rad_img.get_fdata().astype(np.float32)

    if centerline.shape != seg.shape or radius.shape != seg.shape:
        raise ValueError("seg, centerline, and radius volumes must have the same shape")

    spacing = _voxel_spacing_mm(seg_img)
    min_spacing = min(spacing)
    rng = np.random.default_rng(seed)

    positive_radii = radius[radius > 0]
    default_radius_mm = (
        float(np.median(positive_radii)) if positive_radii.size else min_spacing
    )

    dist_center_mm, local_radius_mm = _nearest_centerline_radius(
        seg=seg,
        centerline=centerline,
        radius=radius,
        spacing=spacing,
        default_radius_mm=default_radius_mm,
    )

    raw = rng.normal(background_mean, background_std, seg.shape).astype(np.float32)

    sigma_mm = np.clip(local_radius_mm * radius_sigma_factor, min_spacing * 0.35, None)
    core_profile = np.exp(-(dist_center_mm**2) / (2.0 * sigma_mm**2))

    # Light edge lift prevents large generated vessels from looking hollow after
    # blur when the sparse centreline radius is imperfect.
    dist_to_surface_mm = distance_transform_edt(seg, sampling=spacing).astype(np.float32)
    edge_profile = np.exp(
        -(np.maximum(dist_to_surface_mm - edge_softening_mm, 0.0) ** 2)
        / (2.0 * max(edge_softening_mm, min_spacing * 0.25) ** 2)
    )
    vessel_signal = vessel_peak * np.maximum(core_profile, 0.28 * edge_profile)
    raw = np.where(seg, raw + vessel_signal, raw)

    if halo_fraction > 0:
        dist_outside_mm = distance_transform_edt(~seg, sampling=spacing).astype(np.float32)
        halo_mask = (~seg) & (dist_outside_mm <= halo_sigma_mm * 3.0)
        halo = vessel_peak * halo_fraction * np.exp(
            -(dist_outside_mm[halo_mask] ** 2) / (2.0 * halo_sigma_mm**2)
        )
        raw[halo_mask] += halo

    if noise_std > 0:
        raw += rng.normal(0.0, noise_std, raw.shape).astype(np.float32)

    if blur_sigma_mm > 0:
        sigma_vox = tuple(blur_sigma_mm / s for s in spacing)
        raw = gaussian_filter(raw, sigma=sigma_vox).astype(np.float32)

    raw = np.clip(raw, -200, 700).astype(np.float32)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(raw, seg_img.affine, seg_img.header), out_path)

    print(f"Saved raw volume -> {out_path}")
    print(f"  shape: {raw.shape}")
    print(f"  spacing mm: {spacing}")
    print(f"  vessel voxels: {int(seg.sum())}")
    print(f"  centerline voxels: {int(centerline.sum())}")
    print(f"  range: [{raw.min():.1f}, {raw.max():.1f}]")
    return raw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate raw.nii.gz from pure-Python hepatic vessel outputs."
    )
    p.add_argument("--input_dir", help="Directory containing seg/centerline/radius NIfTI files")
    p.add_argument("--seg", help="Binary vessel mask, usually seg.nii.gz")
    p.add_argument("--centerline", help="Centreline mask, usually centerline.nii.gz")
    p.add_argument("--radius", help="Sparse centreline radius volume, usually radius.nii.gz")
    p.add_argument("--out", required=True, help="Output raw NIfTI path")
    p.add_argument("--bg_mean", type=float, default=60.0)
    p.add_argument("--bg_std", type=float, default=18.0)
    p.add_argument("--peak", type=float, default=330.0)
    p.add_argument("--radius_sigma_factor", type=float, default=0.45)
    p.add_argument("--edge_softening", type=float, default=0.35)
    p.add_argument("--halo_fraction", type=float, default=0.18)
    p.add_argument("--halo_sigma", type=float, default=0.75)
    p.add_argument("--noise", type=float, default=12.0)
    p.add_argument("--blur", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    seg_path, centerline_path, radius_path = _resolve_paths(
        args.input_dir,
        args.seg,
        args.centerline,
        args.radius,
    )
    simulate_raw_for_python_generator(
        seg_path=seg_path,
        centerline_path=centerline_path,
        radius_path=radius_path,
        out_path=args.out,
        background_mean=args.bg_mean,
        background_std=args.bg_std,
        vessel_peak=args.peak,
        radius_sigma_factor=args.radius_sigma_factor,
        edge_softening_mm=args.edge_softening,
        halo_fraction=args.halo_fraction,
        halo_sigma_mm=args.halo_sigma,
        noise_std=args.noise,
        blur_sigma_mm=args.blur,
        seed=args.seed,
    )
