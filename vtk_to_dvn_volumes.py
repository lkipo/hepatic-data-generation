"""
vtk_to_dvn_volumes.py

Reads the VTK polydata graph produced by Whitehead's HepaticVesselGeneration
and rasterizes it into the 6 volumetric channels expected by DeepVesselNet:

  seg.nii.gz          - binary vessel segmentation
  centerline.nii.gz   - binary centerline mask
  points.nii.gz       - edge-count per node  (≥3 = bifurcation)
  bifurcation.nii.gz  - block mask around bifurcation points
  radius.nii.gz       - radius value along centerlines

Usage:
  python vtk_to_dvn_volumes.py \\
      --vtk vessel_tree.vtk \\
      --shape 325 304 600 \\
      --voxel_size 0.385 \\
      --out sample_001/

The VTK graph is assumed to contain:
  - Points:  3D positions
  - Lines:   vessel segments (cell connectivity)
  - PointData array "Radius" (or similar name, see --radius_array)
"""

import argparse
import os
import numpy as np
import nibabel as nib
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def world_to_voxel(pts_world, origin, voxel_size):
    """Convert world-space coords to voxel indices (Z,Y,X)."""
    pts_vox = (pts_world - origin) / voxel_size
    return pts_vox[:, [2, 1, 0]]  # XYZ -> ZYX


def bresenham_3d(p0, p1):
    """
    3D Bresenham line between two integer voxel coordinates.
    Returns array of shape (N,3) in ZYX order.
    """
    p0, p1 = np.array(p0, dtype=int), np.array(p1, dtype=int)
    d = np.abs(p1 - p0)
    s = np.sign(p1 - p0)
    pts = [p0.copy()]
    dom = np.argmax(d)
    if d[dom] == 0:
        return np.array(pts)
    err = d - d[dom] // 2
    cur = p0.copy()
    for _ in range(d[dom]):
        for ax in range(3):
            if ax == dom:
                continue
            err[ax] += d[ax]
            if err[ax] >= d[dom]:
                cur[ax] += s[ax]
                err[ax] -= d[dom]
        cur[dom] += s[dom]
        pts.append(cur.copy())
    return np.array(pts)


def draw_sphere(vol, center, radius_vox):
    """Paint a sphere (or cylinder cross-section) into a volume."""
    r = int(np.ceil(radius_vox))
    z0, y0, x0 = center
    Z, Y, X = vol.shape
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dz**2 + dy**2 + dx**2 <= radius_vox**2:
                    zz, yy, xx = z0 + dz, y0 + dy, x0 + dx
                    if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X:
                        vol[zz, yy, xx] = 1


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def vtk_graph_to_dvn(vtk_path, shape, voxel_size, out_dir,
                     radius_array="Radius", bif_block_radius=3):
    """
    Parameters
    ----------
    vtk_path       : path to VTK polydata file from Whitehead's generator
    shape          : (Z, Y, X) of output volumes
    voxel_size     : isotropic mm per voxel
    out_dir        : directory for output NIfTI files
    radius_array   : name of the PointData array holding vessel radii (mm)
    bif_block_radius : half-size of block placed around bifurcation nodes
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Read VTK ---
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    pd = reader.GetOutput()

    pts_world = vtk_to_numpy(pd.GetPoints().GetData())   # (N,3) XYZ mm
    n_pts = pts_world.shape[0]

    # Radii
    r_data = pd.GetPointData().GetArray(radius_array)
    if r_data is None:
        # Try the first available scalar array
        r_data = pd.GetPointData().GetArray(0)
    radii_mm = vtk_to_numpy(r_data) if r_data is not None else np.ones(n_pts)

    # Lines / edges
    lines = pd.GetLines()
    lines.InitTraversal()
    edges = []
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        n = id_list.GetNumberOfIds()
        for i in range(n - 1):
            edges.append((id_list.GetId(i), id_list.GetId(i + 1)))

    # --- Map world → voxel ---
    origin = pts_world.min(axis=0) - voxel_size  # small margin
    pts_vox = world_to_voxel(pts_world, origin, voxel_size).astype(int)

    # Clip to volume
    Z, Y, X = shape
    pts_vox[:, 0] = pts_vox[:, 0].clip(0, Z - 1)
    pts_vox[:, 1] = pts_vox[:, 1].clip(0, Y - 1)
    pts_vox[:, 2] = pts_vox[:, 2].clip(0, X - 1)
    radii_vox = radii_mm / voxel_size

    # --- Allocate volumes ---
    seg   = np.zeros(shape, dtype=np.uint8)
    cl    = np.zeros(shape, dtype=np.uint8)
    rad   = np.zeros(shape, dtype=np.float32)
    node_edges = np.zeros(n_pts, dtype=int)  # edge count per node

    # Count edges per node
    for a, b in edges:
        node_edges[a] += 1
        node_edges[b] += 1

    # --- Rasterize edges ---
    for a, b in edges:
        line_pts = bresenham_3d(pts_vox[a], pts_vox[b])
        # Interpolate radius along segment
        n_line = len(line_pts)
        r_a, r_b = radii_vox[a], radii_vox[b]
        for k, (z, y, x) in enumerate(line_pts):
            if not (0 <= z < Z and 0 <= y < Y and 0 <= x < X):
                continue
            t = k / max(n_line - 1, 1)
            r = r_a * (1 - t) + r_b * t
            cl[z, y, x] = 1
            rad[z, y, x] = max(rad[z, y, x], r * voxel_size)  # back to mm
            # Draw vessel cross-section
            draw_sphere(seg, (z, y, x), r)

    # --- Points volume (edge count per node voxel) ---
    pts_vol = np.zeros(shape, dtype=np.uint8)
    for i, (z, y, x) in enumerate(pts_vox):
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            pts_vol[z, y, x] = max(pts_vol[z, y, x], node_edges[i])

    # --- Bifurcation volume (block mask around nodes with ≥3 edges) ---
    bif_pts = pts_vox[node_edges >= 3]
    bif_vol = np.zeros(shape, dtype=np.uint8)
    struct = np.ones((2 * bif_block_radius + 1,) * 3, dtype=bool)
    for z, y, x in bif_pts:
        z0 = max(z - bif_block_radius, 0)
        z1 = min(z + bif_block_radius + 1, Z)
        y0 = max(y - bif_block_radius, 0)
        y1 = min(y + bif_block_radius + 1, Y)
        x0 = max(x - bif_block_radius, 0)
        x1 = min(x + bif_block_radius + 1, X)
        bif_vol[z0:z1, y0:y1, x0:x1] = 1

    # --- Save NIfTI ---
    affine = np.diag([voxel_size] * 3 + [1.0])
    nib.save(nib.Nifti1Image(seg,      affine), f"{out_dir}/seg.nii.gz")
    nib.save(nib.Nifti1Image(cl,       affine), f"{out_dir}/centerline.nii.gz")
    nib.save(nib.Nifti1Image(pts_vol,  affine), f"{out_dir}/points.nii.gz")
    nib.save(nib.Nifti1Image(bif_vol,  affine), f"{out_dir}/bifurcation.nii.gz")
    nib.save(nib.Nifti1Image(rad,      affine), f"{out_dir}/radius.nii.gz")

    stats = {
        "vessel_voxels": int(seg.sum()),
        "centerline_voxels": int(cl.sum()),
        "bifurcation_nodes": int((node_edges >= 3).sum()),
        "total_nodes": n_pts,
        "total_edges": len(edges),
    }
    print(f"Saved to {out_dir}/")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vtk", required=True, help="VTK graph from Whitehead generator")
    p.add_argument("--shape", nargs=3, type=int, default=[325, 304, 600],
                   metavar=("Z", "Y", "X"))
    p.add_argument("--voxel_size", type=float, default=0.385,
                   help="Isotropic voxel size in mm (DVN default: 0.385)")
    p.add_argument("--out", required=True, help="Output directory for this sample")
    p.add_argument("--radius_array", default="Radius",
                   help="Name of VTK PointData array holding radii")
    p.add_argument("--bif_block", type=int, default=3,
                   help="Half-size of bifurcation block mask in voxels")
    args = p.parse_args()

    vtk_graph_to_dvn(
        vtk_path=args.vtk,
        shape=tuple(args.shape),
        voxel_size=args.voxel_size,
        out_dir=args.out,
        radius_array=args.radius_array,
        bif_block_radius=args.bif_block,
    )
