"""
hepatic_vessel_generation.py
=============================
Pure-Python implementation of the hepatic arterial tree generation algorithm
described in:

  Whitehead JF, Laeseke PF, Periyasamy S, Speidel MA, Wagner MG.
  "In silico simulation of hepatic arteries: An open-source algorithm for
  efficient synthetic data generation."
  Medical Physics, 2023. https://doi.org/10.1002/mp.16379

Algorithm overview
------------------
1. Initialise a skeleton vessel tree (proper → left/right → Couinaud segments).
2. For each random endpoint p_k (sampled via Monte-Carlo from blood-demand map):
   a. Find the branch c_i with minimum metabolic cost to connect p_k.
   b. Replace c_i with three new branches creating a bifurcation at weighted centroid b.
   c. Reject connections that cause vessel intersections or leave the liver mask.
3. After all endpoints are connected:
   a. Assign radii via Murray's law (root radius → all children).
   b. Optimise bifurcation angles using cubic-Hermite curves.
4. Export tree as a Python dataclass / VTK polydata / NIfTI volumes.

References for equations:
  Eq.1  – bifurcation centroid b
  Eq.2-5 – metabolic cost function
  Eq.6  – Murray's law for radii
  Eq.7  – intersection displacement
  Eq.8  – Murray's law for bifurcation angles
  Eq.9  – cubic-Hermite vessel centreline
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, label, gaussian_filter
from tqdm import tqdm


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Branch:
    """A single vessel branch c_i = (s, e) with associated metadata."""
    idx: int                    # unique ID
    s: np.ndarray               # start point (3,)  world coords (mm or voxel)
    e: np.ndarray               # end   point (3,)
    parent_idx: Optional[int]   # parent branch index (None for root)
    children: List[int] = field(default_factory=list)
    Q: int = 1                  # number of terminal endpoints in subtree
    radius: float = 0.0         # assigned after full tree is built
    segment: int = 0            # Couinaud segment (1-8) or 0 = any

    # Curved centreline (set after bifurcation-angle optimisation)
    ks: Optional[np.ndarray] = None   # tangent at start
    ke: Optional[np.ndarray] = None   # tangent at end


@dataclass
class VesselTree:
    """Container for the full hepatic arterial tree."""
    branches: Dict[int, Branch] = field(default_factory=dict)
    _next_idx: int = 0

    def add_branch(self, s, e, parent_idx=None, segment=0) -> int:
        idx = self._next_idx
        self._next_idx += 1
        Q = 1  # terminal by default
        self.branches[idx] = Branch(idx=idx, s=s.copy(), e=e.copy(),
                                    parent_idx=parent_idx,
                                    segment=segment, Q=Q)
        if parent_idx is not None and parent_idx in self.branches:
            self.branches[parent_idx].children.append(idx)
        return idx

    def remove_branch(self, idx: int):
        b = self.branches.pop(idx, None)
        if b is None:
            return
        if b.parent_idx is not None and b.parent_idx in self.branches:
            p = self.branches[b.parent_idx]
            if idx in p.children:
                p.children.remove(idx)

    def ancestors(self, idx: int) -> List[int]:
        """Return list of ancestor branch indices (excluding idx itself)."""
        result = []
        b = self.branches.get(idx)
        while b is not None and b.parent_idx is not None:
            result.append(b.parent_idx)
            b = self.branches.get(b.parent_idx)
        return result

    def is_terminal(self, idx: int) -> bool:
        return len(self.branches[idx].children) == 0

    def update_Q(self, from_idx: int):
        """Recompute Q (terminal descendant count) for from_idx and all ancestors."""
        def count_terminals(idx):
            b = self.branches[idx]
            if not b.children:
                b.Q = 1
            else:
                b.Q = sum(count_terminals(c) for c in b.children)
            return b.Q
        count_terminals(from_idx)
        # Propagate upward
        b = self.branches[from_idx]
        while b.parent_idx is not None:
            p = self.branches[b.parent_idx]
            p.Q = sum(self.branches[c].Q for c in p.children)
            b = p


# ============================================================================
# Geometry helpers
# ============================================================================

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def normalise(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    return v / n if n > 1e-12 else v


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return norm(a - b)


def segment_to_segment_distance(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Minimum distance between two line segments [p1,p2] and [p3,p4].
    Returns (distance, closest_pt_on_seg1, closest_pt_on_seg2).
    Based on Dan Sunday's algorithm.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    r  = p1 - p3
    a  = np.dot(d1, d1)
    e  = np.dot(d2, d2)
    f  = np.dot(d2, r)

    if a < 1e-10 and e < 1e-10:
        return norm(r), p1, p3

    if a < 1e-10:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = np.dot(d1, r)
        if e < 1e-10:
            s, t = np.clip(-c / a, 0, 1), 0.0
        else:
            b_   = np.dot(d1, d2)
            denom = a * e - b_ * b_
            if abs(denom) > 1e-10:
                s = np.clip((b_ * f - c * e) / denom, 0, 1)
            else:
                s = 0.0
            t = (b_ * s + f) / e
            if t < 0:
                t = 0.0; s = np.clip(-c / a, 0, 1)
            elif t > 1:
                t = 1.0; s = np.clip((b_ - c) / a, 0, 1)

    cp1 = p1 + s * d1
    cp2 = p3 + t * d2
    return norm(cp1 - cp2), cp1, cp2


def cubic_hermite(s_prime: np.ndarray, e: np.ndarray,
                  ks: np.ndarray, ke: np.ndarray,
                  n_pts: int = 20) -> np.ndarray:
    """
    Eq. 9: cubic-Hermite curve from s' to e with tangents ks, ke.
    Returns (n_pts, 3) array of points.
    """
    t = np.linspace(0, 1, n_pts)
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =    t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =    t**3 -   t**2
    pts = (h00[:, None] * s_prime +
           h01[:, None] * e +
           h10[:, None] * ks +
           h11[:, None] * ke)
    return pts


# ============================================================================
# Metabolic cost function  (Eqs. 2-5)
# ============================================================================

def bifurcation_centroid(s_i: np.ndarray, e_i: np.ndarray,
                         p_k: np.ndarray, Qi: int, gamma: float) -> np.ndarray:
    """Eq. 1: weighted centroid b of the new bifurcation."""
    Qi_g  = Qi ** gamma
    Qi1_g = (Qi + 1) ** gamma
    denom = 1.0 + Qi1_g + Qi_g
    b = (p_k + Qi1_g * s_i + Qi_g * e_i) / denom
    return b


def cost(branch: Branch, p_k: np.ndarray,
         tree: VesselTree, gamma: float) -> float:
    """
    Eqs. 2-5: change in total vascular volume when connecting p_k to branch c_i.
    Uses Q^gamma as a surrogate for cross-sectional area.
    """
    s_i, e_i = branch.s, branch.e
    Qi = branch.Q

    b = bifurcation_centroid(s_i, e_i, p_k, Qi, gamma)

    # δ1: replace (s_i, e_i) → (b, e_i)
    Qi_g = Qi ** gamma
    delta1 = Qi_g * (dist(e_i, b) - dist(e_i, s_i))

    # δ2: new branches (s_i, b) and (b, p_k)
    Qi1_g = (Qi + 1) ** gamma
    delta2 = Qi1_g * dist(b, s_i) + dist(p_k, b)

    # δ3: volume increase in all parent branches due to diameter growth
    delta3 = 0.0
    for anc_idx in tree.ancestors(branch.idx):
        anc = tree.branches[anc_idx]
        Qk_g  = anc.Q ** gamma
        Qk1_g = (anc.Q + 1) ** gamma
        delta3 += (Qk1_g - Qk_g) * dist(anc.e, anc.s)

    return delta1 + delta2 + delta3


# ============================================================================
# Intersection check  (Section 2.1.2)
# ============================================================================

def branches_intersect(
    s_a: np.ndarray, e_a: np.ndarray, r_a: float,
    s_b: np.ndarray, e_b: np.ndarray, r_b: float
) -> bool:
    """True if two branches with given radii overlap."""
    d, _, _ = segment_to_segment_distance(s_a, e_a, s_b, e_b)
    return d < (r_a + r_b)


def new_branches_intersect(
    b_pt: np.ndarray,
    s_i: np.ndarray, e_i: np.ndarray,
    p_k: np.ndarray,
    r_new: float,          # approximate radius (use parent's for check)
    tree: VesselTree,
    candidate_idx: int,    # c_i index — skip ancestors/self
    min_radius: float = 0.1
) -> bool:
    """
    Check if any of the three new branches (s_i,b), (b,e_i), (b,p_k)
    intersect existing branches (excluding ancestors of c_i).
    Uses a small radius approximation during tree-building.
    """
    ancestors_set = set(tree.ancestors(candidate_idx)) | {candidate_idx}
    new_segs = [(s_i, b_pt), (b_pt, e_i), (b_pt, p_k)]
    r_check = max(r_new, min_radius)

    for bk_idx, bk in tree.branches.items():
        if bk_idx in ancestors_set:
            continue
        for ns, ne in new_segs:
            d, _, _ = segment_to_segment_distance(ns, ne, bk.s, bk.e)
            if d < r_check * 2:   # conservative: 2× during generation
                return True
    return False


# ============================================================================
# Post-process: displace smaller branch when overlap found (Eq. 7)
# ============================================================================

def resolve_intersections_post(tree: VesselTree):
    """
    After tree generation: if any two branches overlap, displace the
    smaller one outward (Eq. 7).
    """
    branch_list = list(tree.branches.values())
    n = len(branch_list)
    for i in range(n):
        for j in range(i + 1, n):
            ca = branch_list[i]
            cb = branch_list[j]
            ra, rb = ca.radius, cb.radius
            if ra < 1e-6 or rb < 1e-6:
                continue
            d, va, vb = segment_to_segment_distance(ca.s, ca.e, cb.s, cb.e)
            if d < ra + rb:
                # Displace the smaller branch
                if ra >= rb:
                    # displace cb
                    v_dir = normalise(vb - va)
                    vc = va + v_dir * (ra + rb)
                    # replace branch cb with two branches split at vc
                    # (simplified: just move the closest endpoint)
                    if norm(vc - cb.s) < norm(vc - cb.e):
                        tree.branches[cb.idx].s = vc
                    else:
                        tree.branches[cb.idx].e = vc
                else:
                    v_dir = normalise(va - vb)
                    vc = vb + v_dir * (ra + rb)
                    if norm(vc - ca.s) < norm(vc - ca.e):
                        tree.branches[ca.idx].s = vc
                    else:
                        tree.branches[ca.idx].e = vc


# ============================================================================
# Murray's law: radii  (Eq. 6)
# ============================================================================

def assign_radii_murrays_law(
    tree: VesselTree,
    root_idx: int,
    root_radius: float,
    gamma: float
):
    """
    Traverse tree from root and assign radii using Murray's law (Eq. 6):
        R_parent^(2/gamma) = R_child1^(2/gamma) + R_child2^(2/gamma)
    Root radius is given; children radii are derived from parent + sibling.
    """
    exp = 2.0 / gamma

    def _assign(idx: int, r_parent: float):
        b = tree.branches[idx]
        b.radius = r_parent
        children = b.children
        if len(children) == 0:
            return
        if len(children) == 1:
            _assign(children[0], r_parent)
            return
        if len(children) == 2:
            c0, c1 = children
            Q0 = tree.branches[c0].Q
            Q1 = tree.branches[c1].Q
            # Distribute proportionally to Q^(gamma) (flow proxy)
            total = Q0 ** gamma + Q1 ** gamma
            # From Murray: r0^(2/gamma) + r1^(2/gamma) = r_parent^(2/gamma)
            # and r0/r1 = (Q0/Q1)^(gamma/2) (equal shear stress assumption)
            frac0 = (Q0 ** gamma) / total
            frac1 = (Q1 ** gamma) / total
            r0 = r_parent * (frac0 ** (gamma / 2.0))
            r1 = r_parent * (frac1 ** (gamma / 2.0))
            _assign(c0, r0)
            _assign(c1, r1)
        else:
            # >2 children — split evenly
            for c in children:
                _assign(c, r_parent / len(children) ** (gamma / 2.0))

    _assign(root_idx, root_radius)


# ============================================================================
# Murray's law: bifurcation angles  (Eq. 8)
# ============================================================================

def murray_bifurcation_angle(R0: float, R1: float, gamma: float) -> float:
    """
    Eq. 8: theoretical bifurcation angle θ_m between child branch
    with radius R1 and the parent continuation, given parent radius R0.
    Returns angle in radians.
    """
    if R0 < 1e-6 or R1 < 1e-6:
        return 0.0
    exp = 2.0 / gamma
    try:
        R2_sibling = ((R0 ** exp) - (R1 ** exp))
        if R2_sibling <= 0:
            return 0.0
        # cos(θ) = (R0^4 + R1^4 - (R0^(2/γ) - R1^(2/γ))^(2γ)) / (2 R0² R1²)
        inner = (R0 ** exp - R1 ** exp) ** (2.0 * gamma)
        numerator = R0**4 + R1**4 - inner
        denominator = 2.0 * R0**2 * R1**2
        cos_theta = np.clip(numerator / denominator, -1.0, 1.0)
        return float(np.arccos(cos_theta))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return 0.0


# ============================================================================
# Bifurcation angle optimisation with cubic-Hermite curves  (Sec. 2.1.3)
# ============================================================================

def optimise_bifurcation_angles(tree: VesselTree, root_idx: int, gamma: float):
    """
    For each branch, compute the tangent vectors k_s and k_e using the
    theoretically ideal bifurcation angle from Murray's law, then store
    the Hermite parameters so the centreline can be sampled later.
    """
    def _process(idx: int, parent_dir: Optional[np.ndarray]):
        b = tree.branches[idx]
        d = b.e - b.s            # nominal direction vector
        d_norm = normalise(d)

        # --- k_e: project (e - s) into plane of two children ---
        if len(b.children) == 2:
            c0, c1 = b.children
            d0 = normalise(tree.branches[c0].e - tree.branches[c0].s)
            d1 = normalise(tree.branches[c1].e - tree.branches[c1].s)
            # Plane normal of the two children
            plane_n = normalise(np.cross(d0, d1))
            if norm(plane_n) < 1e-8:
                ke = d_norm.copy()
            else:
                ke = d_norm - np.dot(d_norm, plane_n) * plane_n
                ke = normalise(ke)
        else:
            ke = d_norm.copy()

        # --- k_s: rotate parent direction by Murray bifurcation angle ---
        if parent_dir is None or b.radius < 1e-6:
            ks = d_norm.copy()
        else:
            if b.parent_idx is not None:
                pr = tree.branches[b.parent_idx].radius
            else:
                pr = b.radius
            theta = murray_bifurcation_angle(pr, b.radius, gamma)
            # Rotation axis: perpendicular to parent_dir in the plane
            # containing parent_dir and d_norm
            rot_axis = np.cross(parent_dir, d_norm)
            if norm(rot_axis) < 1e-8:
                ks = d_norm.copy()
            else:
                rot_axis = normalise(rot_axis)
                # Rotate parent_dir toward d_norm by theta
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                ks = (cos_t * parent_dir +
                      sin_t * np.cross(rot_axis, parent_dir) +
                      (1 - cos_t) * np.dot(rot_axis, parent_dir) * rot_axis)
                ks = normalise(ks)

        b.ks = ks * norm(d)    # scaled tangent (Hermite convention)
        b.ke = ke * norm(d)

        for child_idx in b.children:
            _process(child_idx, d_norm)

    _process(root_idx, None)


# ============================================================================
# Couinaud segment initialisation  (Section 2.1.4)
# ============================================================================

def build_initial_skeleton(
    liver_mask: np.ndarray,
    segment_map: np.ndarray,
    rng: np.random.Generator
) -> Tuple[VesselTree, int, np.ndarray]:
    """
    Create the initial vessel tree skeleton:
      root (hilum) → proper hepatic artery → left/right hepatic → 8 Couinaud segments

    Parameters
    ----------
    liver_mask   : (Z,Y,X) bool
    segment_map  : (Z,Y,X) int  (values 1-8)
    rng          : numpy random generator

    Returns
    -------
    tree         : VesselTree with skeleton branches
    root_idx     : index of the proper hepatic artery branch
    hilum_pt     : 3D coord of root start point
    """
    tree = VesselTree()

    def random_point_in_segment(seg_id: int) -> np.ndarray:
        coords = np.argwhere((segment_map == seg_id) & liver_mask)
        if len(coords) == 0:
            coords = np.argwhere(liver_mask)
        return coords[rng.integers(len(coords))].astype(float)

    # Step 1: proper hepatic artery — hilum → segment 4 endpoint
    # Hilum approximated as the centroid of the inferior liver border
    liver_z = np.argwhere(liver_mask)
    z_min = liver_z[:, 0].min()
    hilum_candidates = np.argwhere(
        liver_mask & (np.arange(liver_mask.shape[0])[:, None, None] <= z_min + 5)
    )
    if len(hilum_candidates) == 0:
        hilum_candidates = np.argwhere(liver_mask)
    hilum = hilum_candidates[rng.integers(len(hilum_candidates))].astype(float)

    seg4_pt = random_point_in_segment(4)
    root_idx = tree.add_branch(hilum, seg4_pt, parent_idx=None, segment=0)

    # Step 2: connect segment 8 to form left/right hepatic split
    seg8_pt = random_point_in_segment(8)
    # Bifurcation near the midpoint of proper hepatic
    mid_proper = (hilum + seg4_pt) / 2.0
    lha_idx = tree.add_branch(mid_proper, seg8_pt, parent_idx=root_idx, segment=0)
    rha_idx = tree.add_branch(mid_proper, seg4_pt.copy(),
                               parent_idx=root_idx, segment=0)
    # Adjust root to end at bifurcation
    tree.branches[root_idx].e = mid_proper.copy()
    # Fix children list of root
    tree.branches[root_idx].children = [lha_idx, rha_idx]
    tree.branches[lha_idx].parent_idx = root_idx
    tree.branches[rha_idx].parent_idx = root_idx

    # Left lobe: segments 2, 3, 4  →  connect to left hepatic artery
    # Right lobe: segments 5, 6, 7, 8 → connect to right hepatic artery
    # Segment 1 → either (connect to nearest lobe)
    lha_end = tree.branches[lha_idx].e
    rha_end = tree.branches[rha_idx].e

    # Order from paper: 2, 5, 3, 7, 6, 1
    seg_assignments = {
        2: lha_idx, 3: lha_idx,
        5: rha_idx, 6: rha_idx, 7: rha_idx,
        1: lha_idx,  # segment 1 — caudate; connect to left
    }

    for seg_id, parent in seg_assignments.items():
        pt = random_point_in_segment(seg_id)
        parent_branch = tree.branches[parent]
        tree.add_branch(parent_branch.e.copy(), pt,
                        parent_idx=parent, segment=seg_id)

    # Initialise Q counts bottom-up
    _recount_Q(tree, root_idx)

    return tree, root_idx, hilum


def _recount_Q(tree: VesselTree, idx: int) -> int:
    b = tree.branches[idx]
    if not b.children:
        b.Q = 1
        return 1
    b.Q = sum(_recount_Q(tree, c) for c in b.children)
    return b.Q


# ============================================================================
# Endpoint sampling
# ============================================================================

def sample_endpoints(
    blood_demand: np.ndarray,
    liver_mask: np.ndarray,
    segment_map: np.ndarray,
    n_endpoints: int,
    rng: np.random.Generator
) -> List[Tuple[np.ndarray, int]]:
    """
    Monte-Carlo sampling of endpoints p_i from the blood demand map.
    Each endpoint is tagged with its Couinaud segment.
    Returns list of (point, segment_id).
    """
    flat_demand = blood_demand.flatten()
    flat_demand = flat_demand / flat_demand.sum()

    flat_mask = liver_mask.flatten()
    flat_demand[~flat_mask] = 0.0
    flat_demand /= flat_demand.sum()

    shape = blood_demand.shape
    indices = rng.choice(len(flat_demand), size=n_endpoints * 2,
                          p=flat_demand, replace=True)
    pts = []
    for idx_flat in indices:
        if len(pts) >= n_endpoints:
            break
        z, y, x = np.unravel_index(idx_flat, shape)
        # Add small random jitter within voxel
        jitter = rng.uniform(-0.5, 0.5, 3)
        pt = np.array([z, y, x], dtype=float) + jitter
        seg = int(segment_map[z, y, x])
        if seg < 1 or seg > 8:
            seg = 1
        pts.append((pt, seg))

    # If we didn't collect enough, pad with uniform samples
    while len(pts) < n_endpoints:
        coords = np.argwhere(liver_mask)
        c = coords[rng.integers(len(coords))]
        seg = int(segment_map[c[0], c[1], c[2]])
        if seg < 1:
            seg = 1
        pts.append((c.astype(float), seg))

    return pts[:n_endpoints]


# ============================================================================
# Main vessel tree generation loop  (Table 1 pseudocode)
# ============================================================================

def grow_vessel_tree(
    tree: VesselTree,
    root_idx: int,
    endpoints: List[Tuple[np.ndarray, int]],
    liver_mask: np.ndarray,
    gamma: float,
    check_intersections: bool = True,
    max_retries: int = 2,
    approx_radius: float = 0.5,   # mm radius used during intersection check
    verbose: bool = True,
) -> VesselTree:
    """
    Main loop: attach each endpoint to the tree using the minimum-cost branch.

    Parameters
    ----------
    tree               : initialised skeleton tree
    root_idx           : root branch index
    endpoints          : list of (point, segment_id) from sample_endpoints()
    liver_mask         : (Z,Y,X) bool
    gamma              : Murray's law exponent (2.8-3.1)
    check_intersections: whether to reject intersecting connections
    max_retries        : max times an endpoint can be re-queued
    approx_radius      : radius (mm) used for intersection check before
                         radii are finalised
    """
    shape = liver_mask.shape
    retry_count: Dict[int, int] = {}  # endpoint index → retry count
    queue = list(enumerate(endpoints))  # (ep_idx, (pt, seg))

    iterator = tqdm(total=len(queue), desc="Growing vessel tree") if verbose else None

    processed = 0
    while queue:
        ep_idx, (p_k, seg_k) = queue.pop(0)
        if iterator:
            iterator.update(1)
            processed += 1

        # Build priority queue of branches in same segment, sorted by cost
        candidate_costs = []
        for b_idx, branch in tree.branches.items():
            if branch.segment != seg_k and branch.segment != 0:
                c = float('inf')
            else:
                try:
                    c = cost(branch, p_k, tree, gamma)
                except Exception:
                    c = float('inf')
            heapq.heappush(candidate_costs, (c, b_idx))

        connected = False
        tried = []

        while candidate_costs and not connected:
            c_val, ci_idx = heapq.heappop(candidate_costs)
            if c_val == float('inf'):
                break

            branch_i = tree.branches[ci_idx]
            b_pt = bifurcation_centroid(
                branch_i.s, branch_i.e, p_k, branch_i.Q, gamma)

            # Check all three new branches are inside liver mask
            def pt_in_mask(pt):
                iz, iy, ix = int(round(pt[0])), int(round(pt[1])), int(round(pt[2]))
                if not (0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]):
                    return False
                return bool(liver_mask[iz, iy, ix])

            if not (pt_in_mask(b_pt) and pt_in_mask(p_k)):
                continue

            # Intersection check
            if check_intersections:
                if new_branches_intersect(
                    b_pt, branch_i.s, branch_i.e, p_k,
                    approx_radius, tree, ci_idx
                ):
                    tried.append(ci_idx)
                    continue

            # --- Accept: replace branch_i with three new branches ---
            old_parent  = branch_i.parent_idx
            old_segment = branch_i.segment
            old_Q       = branch_i.Q

            # Remove old branch from parent's children list
            if old_parent is not None and old_parent in tree.branches:
                p_branch = tree.branches[old_parent]
                if ci_idx in p_branch.children:
                    p_branch.children.remove(ci_idx)

            # New branch (b → e_i): inherits old subtree
            new_ei_idx = tree._next_idx
            tree._next_idx += 1
            new_ei = Branch(
                idx=new_ei_idx,
                s=b_pt.copy(),
                e=branch_i.e.copy(),
                parent_idx=None,   # will be set below
                children=branch_i.children.copy(),
                Q=old_Q,
                segment=old_segment,
            )
            tree.branches[new_ei_idx] = new_ei
            # Reparent old children
            for child_idx in new_ei.children:
                tree.branches[child_idx].parent_idx = new_ei_idx

            # New branch (b → p_k): new terminal
            new_pk_idx = tree.add_branch(b_pt, p_k,
                                          parent_idx=None, segment=seg_k)
            tree.branches[new_pk_idx].Q = 1

            # New branch (s_i → b): replaces old branch
            # Re-use the old index for the (s_i, b) branch
            branch_i.e = b_pt.copy()
            branch_i.children = [new_ei_idx, new_pk_idx]
            branch_i.Q = old_Q + 1

            # Wire parents
            new_ei.parent_idx = ci_idx
            tree.branches[new_pk_idx].parent_idx = ci_idx

            # Restore old branch to its parent
            if old_parent is not None and old_parent in tree.branches:
                tree.branches[old_parent].children.append(ci_idx)

            # Propagate Q upward
            tree.update_Q(ci_idx)

            connected = True

        if not connected:
            rc = retry_count.get(ep_idx, 0)
            if rc < max_retries:
                retry_count[ep_idx] = rc + 1
                queue.append((ep_idx, (p_k, seg_k)))
            # else: discard endpoint

    if iterator:
        iterator.close()

    return tree


# ============================================================================
# Voxelisation helpers
# ============================================================================

def rasterise_tree(
    tree: VesselTree,
    shape: Tuple[int, int, int],
    voxel_size: float = 1.0,
    n_pts_per_branch: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelise the vessel tree into 5 DVN-compatible volumes.

    Returns
    -------
    seg          : uint8  binary vessel mask
    centerline   : uint8  binary centreline mask
    points_vol   : uint8  edge-count per node voxel
    bifurcation  : uint8  block mask around bifurcation nodes
    radius_vol   : float32 radius (mm) along centrelines
    """
    Z, Y, X = shape
    seg        = np.zeros(shape, np.uint8)
    centreline = np.zeros(shape, np.uint8)
    pts_vol    = np.zeros(shape, np.uint8)
    bif_vol    = np.zeros(shape, np.uint8)
    rad_vol    = np.zeros(shape, np.float32)

    def vox(pt):
        z = int(round(pt[0])); y = int(round(pt[1])); x = int(round(pt[2]))
        return (np.clip(z, 0, Z-1), np.clip(y, 0, Y-1), np.clip(x, 0, X-1))

    def in_bounds(z, y, x):
        return 0 <= z < Z and 0 <= y < Y and 0 <= x < X

    for b in tree.branches.values():
        r_vox = b.radius / voxel_size   # radius in voxels
        n_edges = len(b.children) + (1 if b.parent_idx is not None else 0)

        # Sample centreline
        if b.ks is not None and b.ke is not None:
            # curved (Hermite)
            s_prime = b.s  # simplified: use s directly
            cl_pts = cubic_hermite(s_prime, b.e, b.ks, b.ke, n_pts_per_branch)
        else:
            t = np.linspace(0, 1, n_pts_per_branch)
            cl_pts = b.s[None, :] + t[:, None] * (b.e - b.s)[None, :]

        for pt in cl_pts:
            z, y, x = vox(pt)
            centreline[z, y, x] = 1
            rad_vol[z, y, x] = max(rad_vol[z, y, x], b.radius)

            # Draw sphere of radius r_vox
            r_int = max(1, int(math.ceil(r_vox)))
            for dz in range(-r_int, r_int + 1):
                for dy in range(-r_int, r_int + 1):
                    for dx in range(-r_int, r_int + 1):
                        if dz**2 + dy**2 + dx**2 <= r_vox**2:
                            zz, yy, xx = z+dz, y+dy, x+dx
                            if in_bounds(zz, yy, xx):
                                seg[zz, yy, xx] = 1

        # Node point (start of branch = bifurcation point)
        z, y, x = vox(b.s)
        pts_vol[z, y, x] = max(pts_vol[z, y, x], n_edges)

        # Bifurcation block (≥3 edges at node)
        if n_edges >= 3:
            bk = 3  # block half-size in voxels
            z0,y0,x0 = max(z-bk,0), max(y-bk,0), max(x-bk,0)
            z1,y1,x1 = min(z+bk+1,Z), min(y+bk+1,Y), min(x+bk+1,X)
            bif_vol[z0:z1, y0:y1, x0:x1] = 1

    return seg, centreline, pts_vol, bif_vol, rad_vol


# ============================================================================
# Public API
# ============================================================================

def generate_hepatic_tree(
    liver_mask: np.ndarray,
    segment_map: np.ndarray,
    blood_demand: Optional[np.ndarray] = None,
    n_endpoints: int = 5000,
    gamma: float = 2.9,
    root_radius_mm: float = 2.5,
    voxel_size: float = 1.0,
    check_intersections: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[VesselTree, int]:
    """
    Full Whitehead et al. hepatic vessel generation pipeline.

    Parameters
    ----------
    liver_mask      : (Z,Y,X) bool — binary liver volume
    segment_map     : (Z,Y,X) int  — Couinaud segment labels 1-8 (0=background)
    blood_demand    : (Z,Y,X) float in [0,1], uniform if None
    n_endpoints     : number of terminal vessel endpoints to generate
    gamma           : Murray's law exponent (paper: 2.8–3.1)
    root_radius_mm  : radius of the proper hepatic artery (paper: 2.0–3.5 mm)
    voxel_size      : mm per voxel (isotropic)
    check_intersections : enable vessel intersection rejection
    seed            : random seed
    verbose         : show progress bar

    Returns
    -------
    tree            : VesselTree with all branches, radii, and Hermite params
    root_idx        : index of root (proper hepatic artery) branch
    """
    rng = np.random.default_rng(seed)

    if blood_demand is None:
        blood_demand = liver_mask.astype(np.float32)

    t0 = time.time()

    # 1. Skeleton initialisation
    if verbose:
        print("Building initial skeleton...")
    tree, root_idx, hilum = build_initial_skeleton(liver_mask, segment_map, rng)

    # 2. Sample endpoints
    if verbose:
        print(f"Sampling {n_endpoints} endpoints...")
    endpoints = sample_endpoints(blood_demand, liver_mask, segment_map,
                                  n_endpoints, rng)

    # 3. Grow tree
    if verbose:
        print("Growing tree (this is the slow step)...")
    tree = grow_vessel_tree(
        tree, root_idx, endpoints, liver_mask,
        gamma=gamma,
        check_intersections=check_intersections,
        verbose=verbose,
    )

    # 4. Assign radii via Murray's law
    if verbose:
        print("Assigning radii (Murray's law)...")
    assign_radii_murrays_law(tree, root_idx, root_radius_mm, gamma)

    # 5. Optimise bifurcation angles
    if verbose:
        print("Optimising bifurcation angles...")
    optimise_bifurcation_angles(tree, root_idx, gamma)

    # 6. Post-process intersections
    if verbose:
        print("Post-processing intersections...")
    resolve_intersections_post(tree)

    elapsed = time.time() - t0
    if verbose:
        n_branches = len(tree.branches)
        n_bifurcations = sum(1 for b in tree.branches.values() if len(b.children) >= 2)
        print(f"\nDone in {elapsed:.1f}s  |  branches: {n_branches}  "
              f"|  bifurcations: {n_bifurcations}")

    return tree, root_idx


def generate_volumes(
    tree: VesselTree,
    liver_mask: np.ndarray,
    voxel_size: float = 1.0,
    add_noise: bool = True,
    background_hu: float = 60.0,
    vessel_hu: float = 280.0,
) -> Dict[str, np.ndarray]:
    """
    Convert a VesselTree to the 6 DVN training volumes + a raw intensity image.

    Returns dict with keys:
        raw, seg, centerline, points, bifurcation, radius
    """
    shape = liver_mask.shape
    seg, cl, pts, bif, rad = rasterise_tree(tree, shape, voxel_size)

    # Raw intensity (Gaussian vessel profile + noise)
    raw = np.random.normal(background_hu, 15.0, shape).astype(np.float32)
    dist_in = distance_transform_edt(seg) * voxel_size
    raw += seg * vessel_hu * np.exp(-(dist_in**2) / (2.0 * (0.5)**2))
    if add_noise:
        raw += np.random.normal(0, 10.0, shape).astype(np.float32)
        raw = gaussian_filter(raw, sigma=0.5)
    raw = np.clip(raw, -200, 600).astype(np.float32)

    return {
        "raw":          raw,
        "seg":          seg,
        "centerline":   cl,
        "points":       pts,
        "bifurcation":  bif,
        "radius":       rad,
    }


# ============================================================================
# Export helpers
# ============================================================================

def save_volumes_nifti(volumes: Dict[str, np.ndarray],
                        out_dir: str,
                        voxel_size: float = 1.0):
    """Save all volumes as NIfTI files."""
    import nibabel as nib
    import os
    os.makedirs(out_dir, exist_ok=True)
    affine = np.diag([voxel_size] * 3 + [1.0])
    for name, vol in volumes.items():
        nib.save(nib.Nifti1Image(vol, affine),
                 os.path.join(out_dir, f"{name}.nii.gz"))
    print(f"Saved {len(volumes)} volumes to {out_dir}/")


def save_vtk_polydata(tree: VesselTree, out_path: str,
                       n_pts_per_branch: int = 20):
    """
    Export tree as a VTK polydata file (lines + radius scalar).
    Requires the `vtk` package; gracefully skips if unavailable.
    """
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError:
        print("vtk not installed — skipping VTK export. "
              "Install with: pip install vtk")
        return

    pts_vtk = vtk.vtkPoints()
    lines   = vtk.vtkCellArray()
    radii   = vtk.vtkFloatArray()
    radii.SetName("Radius")

    pt_idx = 0
    for b in tree.branches.values():
        if b.ks is not None and b.ke is not None:
            cl_pts = cubic_hermite(b.s, b.e, b.ks, b.ke, n_pts_per_branch)
        else:
            t = np.linspace(0, 1, n_pts_per_branch)
            cl_pts = b.s[None, :] + t[:, None] * (b.e - b.s)[None, :]

        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(cl_pts))
        for k, pt in enumerate(cl_pts):
            pts_vtk.InsertNextPoint(float(pt[2]), float(pt[1]), float(pt[0]))  # ZYX→XYZ
            radii.InsertNextValue(float(b.radius))
            line.GetPointIds().SetId(k, pt_idx)
            pt_idx += 1
        lines.InsertNextCell(line)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts_vtk)
    pd.SetLines(lines)
    pd.GetPointData().AddArray(radii)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(out_path)
    writer.SetInputData(pd)
    writer.Write()
    print(f"Saved VTK → {out_path}")


# ============================================================================
# CLI / demo
# ============================================================================

if __name__ == "__main__":
    import argparse, os

    p = argparse.ArgumentParser(
        description="Whitehead et al. hepatic vessel generation (Python)"
    )
    p.add_argument("--liver_mask",    default=None, help="NIfTI liver mask")
    p.add_argument("--segment_map",   default=None, help="NIfTI Couinaud segment map")
    p.add_argument("--blood_demand",  default=None, help="NIfTI blood demand map")
    p.add_argument("--n_endpoints",   type=int, default=2000)
    p.add_argument("--gamma",         type=float, default=2.9)
    p.add_argument("--root_radius",   type=float, default=2.5)
    p.add_argument("--voxel_size",    type=float, default=1.0)
    p.add_argument("--no_intersect",  action="store_true",
                   help="Disable intersection check (faster)")
    p.add_argument("--out",           default="hepatic_output",
                   help="Output directory")
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    # Load or generate liver mask
    if args.liver_mask:
        import nibabel as nib
        liver_mask   = nib.load(args.liver_mask).get_fdata().astype(bool)
        segment_map  = nib.load(args.segment_map).get_fdata().astype(int)
        blood_demand = (nib.load(args.blood_demand).get_fdata().astype(np.float32)
                        if args.blood_demand else None)
        voxel_size   = float(np.abs(nib.load(args.liver_mask).header.get_zooms()[0]))
    else:
        print("No liver mask provided — generating synthetic ellipsoidal liver...")
        # Import the mask generator from the same package
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from generate_liver_mask import generate_liver_mask as gen_mask
        shape = (120, 100, 150)
        liver_mask, blood_demand, segment_map = gen_mask(
            shape=shape, voxel_size_mm=args.voxel_size, output_dir="/tmp"
        )
        voxel_size = args.voxel_size

    # Generate tree
    tree, root_idx = generate_hepatic_tree(
        liver_mask=liver_mask,
        segment_map=segment_map,
        blood_demand=blood_demand,
        n_endpoints=args.n_endpoints,
        gamma=args.gamma,
        root_radius_mm=args.root_radius,
        voxel_size=voxel_size,
        check_intersections=not args.no_intersect,
        seed=args.seed,
        verbose=True,
    )

    # Generate volumes
    print("Rasterising volumes...")
    volumes = generate_volumes(tree, liver_mask, voxel_size=voxel_size)

    # Save
    save_volumes_nifti(volumes, args.out, voxel_size)
    save_vtk_polydata(tree, os.path.join(args.out, "vessel_tree.vtk"))
    print("Complete.")
