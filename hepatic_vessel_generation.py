"""
hepatic_vessel_generation.py
=============================
Faithful Python port of Whitehead et al. C++ hepatic vessel generation
(main.cpp / phantom.h from jewhiteh/HepaticVesselGeneration).

Key differences from the paper-only previous version:
  - Data layout: flat parallel arrays (xstart, xstop, ystart, ystop,
    zstart, zstop, root, children, Qs, section, r) exactly as in C++.
  - Blood-demand map: 4-D uint8 array of shape (Nx, Ny, Nz, 10),
    column-major (Fortran) order. Layers 0-7 = Couinaud segments,
    layer 8 = overall liver mask (index dims[3]-1 = 9).
  - Endpoint sampling: rejection sampling via uniform random + blood-demand
    threshold, exactly as in build_tree / create_tree.
  - Section assignment: probabilistic via get_section, exactly as in C++.
  - Cost function: C++ formula (not Eq.2-5 rearrangement; the C++ uses
    Euclidean distances, not squared distances, for the metabolic cost).
  - V[] array: cumulative volumetric cost propagated up the tree.
  - Intersection check: radius-based segment-to-segment distance, with the
    1.05× best-cost window, exactly as in the C++ search loop.
  - push_branches: post-processing displacement of smaller branch, including
    the split-branch-in-middle case and the recheck queue.
  - optimizeAngles: Rodrigues rotation, linearPoint computation, Hermite
    polynomial interpolation (the Newton forward-difference form used in C++).
  - Output: Tree<N>.txt (one branch per line, x y z triplets) and Radii<N>.txt,
    matching the C++ output format.

Usage
-----
  python hepatic_vessel_generation.py \
      --phantom   BloodDemandMap/Phantom.dat \
      --Nx 512 --Ny 512 --Nz 512 \
      --endpoints 5000 \
      --trees 1 \
      --phantom_res 0.77 \
      --desired_res 0.3 \
      --out CLines/
"""

from __future__ import annotations
import argparse
import copy
import math
import os
import struct
import sys
import time
from collections import deque
from typing import List, Tuple

import numpy as np

SMALL_NUM = 1e-7
INT_MAX = 2**31 - 1
MT19937_MAX = 2**32 - 1


class CppMT19937:
    """Small wrapper matching C++ std::mt19937 draws used by the original."""

    _N = 624
    _M = 397
    _MATRIX_A = 0x9908B0DF
    _UPPER_MASK = 0x80000000
    _LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int | None = None, state=None, index=None):
        if state is not None:
            self._mt = list(state)
            self._index = int(index)
            return
        self._mt = [0] * self._N
        self._index = self._N
        self.seed(0 if seed is None else int(seed))

    def seed(self, seed: int):
        self._mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self._N):
            self._mt[i] = (
                1812433253 * (self._mt[i - 1] ^ (self._mt[i - 1] >> 30)) + i
            ) & 0xFFFFFFFF
        self._index = self._N

    def clone(self) -> "CppMT19937":
        return CppMT19937(state=copy.deepcopy(self._mt), index=self._index)

    def _twist(self):
        mag01 = [0, self._MATRIX_A]
        for kk in range(self._N - self._M):
            y = (self._mt[kk] & self._UPPER_MASK) | (self._mt[kk + 1] & self._LOWER_MASK)
            self._mt[kk] = self._mt[kk + self._M] ^ (y >> 1) ^ mag01[y & 1]
        for kk in range(self._N - self._M, self._N - 1):
            y = (self._mt[kk] & self._UPPER_MASK) | (self._mt[kk + 1] & self._LOWER_MASK)
            self._mt[kk] = self._mt[kk + (self._M - self._N)] ^ (y >> 1) ^ mag01[y & 1]
        y = (self._mt[self._N - 1] & self._UPPER_MASK) | (self._mt[0] & self._LOWER_MASK)
        self._mt[self._N - 1] = self._mt[self._M - 1] ^ (y >> 1) ^ mag01[y & 1]
        self._index = 0

    def random_uint(self) -> int:
        if self._index >= self._N:
            self._twist()
        y = self._mt[self._index]
        self._index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def random(self) -> float:
        return self.random_uint() / float(MT19937_MAX)


# ============================================================================
# Linear index (column-major, matches C++ getLinearInd)
# dims = [Nx, Ny, Nz, 10]
# ============================================================================
def linear_ind(y: int, x: int, z: int, l: int, dims) -> int:
    """
    C++: *index = y + dims[0]*x + dims[1]*dims[0]*z + dims[2]*dims[1]*dims[0]*l
    Note: C++ parameter order is (y, x, z, l) — first arg is the x-voxel,
    named 'y' in the signature (confusing but literal).
    """
    return int(y) + dims[0]*int(x) + dims[1]*dims[0]*int(z) + dims[2]*dims[1]*dims[0]*int(l)


# ============================================================================
# Segment-to-segment distance (exact C++ port)
# ============================================================================
def dist3d_seg_to_seg(s1x, s1y, s1z, e1x, e1y, e1z,
                      s2x, s2y, s2z, e2x, e2y, e2z):
    """Returns (distance, P1[3], P2[3]) — closest points on each segment."""
    ux, uy, uz = e1x-s1x, e1y-s1y, e1z-s1z
    vx, vy, vz = e2x-s2x, e2y-s2y, e2z-s2z
    wx, wy, wz = s1x-s2x, s1y-s2y, s1z-s2z
    a = ux*ux + uy*uy + uz*uz
    b = ux*vx + uy*vy + uz*vz
    c = vx*vx + vy*vy + vz*vz
    d = ux*wx + uy*wy + uz*wz
    e = vx*wx + vy*wy + vz*wz
    D = a*c - b*b
    sD = D
    tD = D
    if D < SMALL_NUM:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b*e - c*d
        tN = a*e - b*d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e+b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d+b) < 0.0:
            sN = 0.0
        elif (-d+b) > a:
            sN = sD
        else:
            sN = (-d+b)
            sD = a
    sc = 0.0 if abs(sN) < SMALL_NUM else sN/sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN/tD
    dpx = wx + sc*ux - tc*vx
    dpy = wy + sc*uy - tc*vy
    dpz = wz + sc*uz - tc*vz
    P1 = [s1x+sc*ux, s1y+sc*uy, s1z+sc*uz]
    P2 = [s2x+tc*vx, s2y+tc*vy, s2z+tc*vz]
    dist = math.sqrt(dpx*dpx + dpy*dpy + dpz*dpz)
    return dist, P1, P2


def dist3d_seg_to_seg_binary(s1x, s1y, s1z, e1x, e1y, e1z,
                             s2x, s2y, s2z, e2x, e2y, e2z) -> float:
    """Fast version — distance only, no closest points."""
    d, _, _ = dist3d_seg_to_seg(s1x, s1y, s1z, e1x, e1y, e1z,
                                s2x, s2y, s2z, e2x, e2y, e2z)
    return d


# ============================================================================
# isequal (2-decimal place comparison, as in C++)
# ============================================================================
def isequal(sx, sy, sz, ex, ey, ez) -> bool:
    return (round(sx*100) == round(ex*100) and
            round(sy*100) == round(ey*100) and
            round(sz*100) == round(ez*100))


# ============================================================================
# isOutMask
# ============================================================================
def is_out_mask(x1, y1, z1, x2, y2, z2, perf, dims) -> bool:
    """Returns True if any point along the segment is outside liver mask."""
    layer = dims[3]-1  # = 9

    def chk(x, y, z):
        xi, yi, zi = int(x), int(y), int(z)
        if xi < 0 or yi < 0 or zi < 0 or xi >= dims[0] or yi >= dims[1] or zi >= dims[2]:
            return True
        return perf[linear_ind(xi, yi, zi, layer, dims)] == 0
    if chk(x1, y1, z1):
        return True
    if chk(x2, y2, z2):
        return True
    xv, yv, zv = x2-x1, y2-y1, z2-z1
    length = math.sqrt(xv*xv + yv*yv + zv*zv)
    maxIt = math.ceil(length)
    for i in range(maxIt):
        t = i/maxIt
        if chk(x1+xv*t, y1+yv*t, z1+zv*t):
            return True
    return False


# ============================================================================
# get_section
# ============================================================================
def get_section(x, y, z, perf, mt_rng, dims) -> int:
    """Probabilistic assignment to Couinaud segment (1-8), returns 0 if none."""
    # C++ takes mt19937 by value here, so the probability draw must not advance
    # the caller's generator state.
    local_rng = mt_rng.clone()
    regions = [0]*8
    probability = [0.0]*8
    count = 7
    for i in range(8):
        idx = linear_ind(int(x), int(y), int(z), i, dims)
        if perf[idx] != 0:
            regions[count] = i+1
            probability[count] = float(perf[idx])
            count -= 1
    # build cumulative distribution
    for i in range(7):
        probability[i+1] += probability[i]
    if probability[6] == 0.0:
        return regions[7]
    total = probability[7]
    if total == 0.0:
        return 0
    if total != 1.0:
        for i in range(8):
            probability[i] /= total
    prob = local_rng.random()
    if prob == 1.0 or prob == 0.0:
        return regions[7]
    for i in range(7):
        if probability[i] <= prob < probability[i+1]:
            return regions[i+1]
    return regions[7]


# ============================================================================
# get_best_cost_idx (linear scan as in C++)
# ============================================================================
def get_best_cost_idx(all_costs, count) -> int:
    best_idx = -1
    best_cost = math.inf
    for pos in range(count):
        if all_costs[pos] < best_cost:
            best_idx = pos
            best_cost = all_costs[pos]
    return best_idx


# ============================================================================
# create_tree
# ============================================================================
def create_tree(xstart, xstop, ystart, ystop, zstart, zstop,
                root_arr, section, length, Qs, V,
                # list of 8 deques (index 0=region1…7=region8)
                regions,
                gamma, terminal_pts, children,
                perf, mt_rng, dims, arrayAlloc, scale,
                progress_interval=0):
    """
    Direct port of Liver::create_tree.
    All array arguments are Python lists (pre-allocated to arrayAlloc size).
    Returns count (number of branches).
    """
    qPower = 2.0 / gamma
    max_tries = 2
    offset = 0
    removed = 0
    count = 1
    region = 1   # starts at 1 (first point already selected in seg 4)

    # order to pull from regions (0-indexed into regions list = seg-1)
    order = [4, 8, 2, 5, 3, 7, 6, 1]  # 1-indexed segment numbers

    # --- initial segment inside mask check ---
    out = is_out_mask(xstart[0], ystart[0], zstart[0],
                      xstop[0], ystop[0], zstop[0], perf, dims)
    while out:
        # mark old points as visited
        for px, py, pz in [(xstart[0], ystart[0], zstart[0]),
                           (xstop[0], ystop[0], zstop[0])]:
            idx = linear_ind(int(px), int(py), int(pz), dims[3]-1, dims)
            perf[idx] = 1
        init_secs = [9, 3]  # 0-indexed layers (segment indices)
        for pos2 in range(2):
            found = False
            while not found:
                k = mt_rng.random()*dims[2] - 0.5
                j = mt_rng.random()*dims[1] - 0.5
                i = mt_rng.random()*dims[0] - 0.5
                val = 255.0 * mt_rng.random()
                idx = linear_ind(int(i), int(j), int(k), init_secs[pos2], dims)
                if val < perf[idx]:
                    if pos2 == 0:
                        xstart[0] = i
                        ystart[0] = j
                        zstart[0] = k
                    else:
                        xstop[0] = i
                        ystop[0] = j
                        zstop[0] = k
                    perf[idx] = 255  # mark as -1 (uint8 wraps)
                    found = True
        out = is_out_mask(xstart[0], ystart[0], zstart[0],
                          xstop[0], ystop[0], zstop[0], perf, dims)

    # initial branch metadata
    children[0] = -1
    children[1] = -1
    root_arr[0] = -1
    section[0] = 4
    Qs[0] = 1
    dx = xstart[0]-xstop[0]
    dy = ystart[0]-ystop[0]
    dz = zstart[0]-zstop[0]
    length[0] = math.sqrt(dx*dx + dy*dy + dz*dz)
    V[0] = 0.0

    Qscale = 1.0 / (Qs[0] ** (1.0/gamma))
    all_costs = [0.0] * arrayAlloc
    if progress_interval <= 0:
        progress_interval = max(1, min(1000, terminal_pts // 100 or 1))
    start_time = time.time()
    last_report_connected = -1

    n = 0
    while n < terminal_pts + offset + removed:

        # --- resample if all regions empty ---
        all_empty = all(len(regions[i]) == 0 for i in range(8))
        if all_empty:
            cur_pts = (count - 1) // 2
            print(f"Resampling {terminal_pts - cur_pts} endpoints.")
            counter = cur_pts
            mxInt = 2**32 - 1  # mt19937 max
            pos2 = cur_pts
            while pos2 < terminal_pts:
                found = False
                advance_pos = True
                while not found:
                    k = mt_rng.random()*dims[2] - 0.5
                    j = mt_rng.random()*dims[1] - 0.5
                    i = mt_rng.random()*dims[0] - 0.5
                    val = 255.0 * mt_rng.random()
                    idx = linear_ind(int(i), int(j), int(k), dims[3]-1, dims)
                    if val < perf[idx]:
                        point = [i, j, k, 0.0]
                        sec = get_section(int(i), int(
                            j), int(k), perf, mt_rng, dims)
                        if sec != 0:
                            regions[sec-1].append(point)
                        else:
                            advance_pos = False
                        counter += 1
                        perf[idx] = 255
                        found = True
                if advance_pos:
                    pos2 += 1

        connected_pts = (count - 1) // 2
        should_report = (
            connected_pts != last_report_connected
            and (
                connected_pts == 0
                or connected_pts >= terminal_pts
                or connected_pts % progress_interval == 0
            )
        )
        if should_report:
            elapsed = max(time.time() - start_time, 1e-9)
            rate = connected_pts / elapsed
            pct = 100.0 * connected_pts / max(terminal_pts, 1)
            print(
                f"[grow] {connected_pts}/{terminal_pts} endpoints "
                f"({pct:5.1f}%) | branches={count} | retries={offset} "
                f"| removed={removed} | {rate:.2f} endpoints/s",
                flush=True,
            )
            last_report_connected = connected_pts

        # --- get next point from regions in round-robin order ---
        found_pt = False
        point = None
        while not found_pt:
            seg_idx = order[(region) % 8] - 1  # 0-indexed
            if len(regions[seg_idx]) > 0:
                point = regions[seg_idx].popleft()
                found_pt = True
            region += 1
        sctn = order[(region - 1) % 8]  # 1-indexed Couinaud segment

        # let initial tree form before counting attempts
        if count > 50:
            point[3] += 1

        # ---- compute cost for each branch ----
        for pos in range(count):
            # segment filter
            if count < 15:
                if ((sctn >= 5 and 1 < section[pos] < 5) or
                        (sctn < 5 and section[pos] >= 5 and sctn > 1)):
                    if count > 3:
                        all_costs[pos] = INT_MAX
                        continue
            else:
                if sctn != section[pos]:
                    all_costs[pos] = INT_MAX
                    continue

            wTrunk = (Qs[pos]+1.0)**qPower
            wBranch = Qs[pos]**qPower
            norm_ = 1.0 / (1.0 + wTrunk + wBranch)
            xc = norm_*(point[0] + wTrunk*xstart[pos] + wBranch*xstop[pos])
            yc = norm_*(point[1] + wTrunk*ystart[pos] + wBranch*ystop[pos])
            zc = norm_*(point[2] + wTrunk*zstart[pos] + wBranch*zstop[pos])

            old_len = math.sqrt(
                (xstart[pos]-xstop[pos])**2+(ystart[pos]-ystop[pos])**2+(zstart[pos]-zstop[pos])**2)
            old_br_l = math.sqrt(
                (xc-xstop[pos])**2+(yc-ystop[pos])**2+(zc-zstop[pos])**2)
            new_br_l = math.sqrt(
                (point[0]-xc)**2+(point[1]-yc)**2+(point[2]-zc)**2)
            old_rt_l = math.sqrt(
                (xstart[pos]-xc)**2+(ystart[pos]-yc)**2+(zstart[pos]-zc)**2)

            dist_cost = -(old_len * (Qs[pos]**qPower))
            dist_cost += old_br_l*wBranch + new_br_l + old_rt_l*wTrunk

            loc = root_arr[pos]
            vol_change = 0.0
            while loc != -1:
                vol_change += V[loc]
                loc = root_arr[loc]
            all_costs[pos] = dist_cost + vol_change

        # ---- find best viable connection (up to 50 candidates, ≤1.05×best) ----
        Qscale = 1.0 / (Qs[0]**(1.0/gamma))
        r_con = scale * Qscale * (1.0**(1.0/gamma))
        max_cost = math.inf
        best_idx = -1
        best_cost = math.inf
        tmp_costs = all_costs[:count][:]  # local copy for iteration

        for trial in range(50):
            pos = get_best_cost_idx(tmp_costs, count)
            if trial == 0:
                max_cost = tmp_costs[pos] * 1.05
            cur_best = tmp_costs[pos]
            tmp_costs[pos] = math.inf
            if cur_best > max_cost or pos == -1:
                break

            wTrunk = (Qs[pos]+1.0)**qPower
            wBranch = Qs[pos]**qPower
            norm_ = 1.0 / (1.0 + wTrunk + wBranch)
            xc = norm_*(point[0]+wTrunk*xstart[pos]+wBranch*xstop[pos])
            yc = norm_*(point[1]+wTrunk*ystart[pos]+wBranch*ystop[pos])
            zc = norm_*(point[2]+wTrunk*zstart[pos]+wBranch*zstop[pos])

            if is_out_mask(point[0], point[1], point[2], xc, yc, zc, perf, dims):
                cur_best = math.inf
                continue
            if is_out_mask(xstart[pos], ystart[pos], zstart[pos], xc, yc, zc, perf, dims):
                cur_best = math.inf
                continue
            if is_out_mask(xstop[pos], ystop[pos], zstop[pos],  xc, yc, zc, perf, dims):
                cur_best = math.inf
                continue

            # Intersection check
            r_pos1 = scale * Qscale * (Qs[pos]**(1.0/gamma))
            r_pos2 = scale * Qscale * ((Qs[pos]+1)**(1.0/gamma))
            sister = -1
            if root_arr[pos] != -1:
                sister = children[2*root_arr[pos]]
                if sister == pos:
                    sister = children[2*root_arr[pos]+1]

            intersected = False
            for ic in range(count):
                if intersected or pos == ic:
                    continue
                r_check = scale * Qscale * (Qs[ic]**(1.0/gamma))

                d_sc = math.inf
                if root_arr[pos] != ic and sister != ic:
                    d_sc = dist3d_seg_to_seg_binary(
                        xstart[pos], ystart[pos], zstart[pos], xc, yc, zc,
                        xstart[ic], ystart[ic], zstart[ic], xstop[ic], ystop[ic], zstop[ic])
                d_cs = math.inf
                if children[pos*2] != ic and children[pos*2+1] != ic:
                    d_cs = dist3d_seg_to_seg_binary(
                        xc, yc, zc, xstop[pos], ystop[pos], zstop[pos],
                        xstart[ic], ystart[ic], zstart[ic], xstop[ic], ystop[ic], zstop[ic])
                d_cx = dist3d_seg_to_seg_binary(
                    xc, yc, zc, point[0], point[1], point[2],
                    xstart[ic], ystart[ic], zstart[ic], xstop[ic], ystop[ic], zstop[ic])

                if d_sc < r_pos2+r_check or d_cs < r_pos1+r_check or d_cx < r_con+r_check:
                    cur_best = math.inf
                    intersected = True

            if not intersected and cur_best < INT_MAX:
                best_idx = pos
                best_cost = cur_best
                break

        # ---- update tree ----
        if best_cost < INT_MAX:
            pos = best_idx
            wTrunk = (Qs[pos]+1.0)**qPower
            wBranch = Qs[pos]**qPower
            norm_ = 1.0 / (1.0+wTrunk+wBranch)
            xc = norm_*(point[0]+wTrunk*xstart[pos]+wBranch*xstop[pos])
            yc = norm_*(point[1]+wTrunk*ystart[pos]+wBranch*ystop[pos])
            zc = norm_*(point[2]+wTrunk*zstart[pos]+wBranch*zstop[pos])

            old_br_l = math.sqrt(
                (xc-xstop[pos])**2+(yc-ystop[pos])**2+(zc-zstop[pos])**2)
            new_br_l = math.sqrt(
                (point[0]-xc)**2+(point[1]-yc)**2+(point[2]-zc)**2)
            old_rt_l = math.sqrt(
                (xstart[pos]-xc)**2+(ystart[pos]-yc)**2+(zstart[pos]-zc)**2)
            length[count] = old_br_l
            length[count+1] = new_br_l
            length[pos] = old_rt_l

            # branch: centroid → old endpoint
            xstart[count] = xc
            ystart[count] = yc
            zstart[count] = zc
            xstop[count] = xstop[pos]
            ystop[count] = ystop[pos]
            zstop[count] = zstop[pos]
            # branch: centroid → new point
            xstart[count+1] = xc
            ystart[count+1] = yc
            zstart[count+1] = zc
            xstop[count+1] = point[0]
            ystop[count+1] = point[1]
            zstop[count+1] = point[2]
            # branch: old start → centroid
            xstop[pos] = xc
            ystop[pos] = yc
            zstop[pos] = zc

            Qs[count] = Qs[pos]
            root_arr[count] = pos
            children[count*2] = children[pos*2]
            children[count*2+1] = children[pos*2+1]
            section[count] = section[pos]

            if children[2*pos] != -1:
                root_arr[children[2*pos]] = count
                root_arr[children[2*pos+1]] = count

            Qs[count+1] = 1
            root_arr[count+1] = pos
            children[2*(count+1)] = -1
            children[2*(count+1)+1] = -1
            section[count+1] = sctn

            children[2*pos] = count
            children[2*pos+1] = count+1

            # propagate Q and V up
            p = pos
            while p != -1:
                Qs[p] += 1
                V[p] = length[p] * ((Qs[p]+1)**qPower - Qs[p]**qPower)
                p = root_arr[p]
            V[count] = length[count] * \
                ((Qs[count]+1)**qPower - Qs[count]**qPower)
            V[count+1] = length[count+1] * \
                ((Qs[count+1]+1)**qPower - Qs[count+1]**qPower)

            count += 2

        elif point[3] < max_tries:
            regions[sctn-1].appendleft(point)
            if count < 15:
                region -= 1
            offset += 1
        else:
            removed += 1

        if count == (terminal_pts*2 + 1):
            elapsed = max(time.time() - start_time, 1e-9)
            print(
                f"[grow] {terminal_pts}/{terminal_pts} endpoints (100.0%) "
                f"completed in {elapsed:.1f}s",
                flush=True,
            )
            return count
        n += 1

    return count


# ============================================================================
# push_branches (post-process intersection correction)
# ============================================================================
def push_branches(xstart, xstop, ystart, ystop, zstart, zstop,
                  root_arr, Qs, section, count_ref,
                  Qscale, gamma, scale, terminal_pts, children, r):
    """
    Post-process: displace smaller branch when two branches overlap.
    Returns 0 on success, -1 if stuck in recheck loop.
    Modifies arrays in-place; count_ref is a list [count] so we can mutate.
    """
    print("Fixing any intersection that may have occurred due to growth...")
    recheck = deque()
    max_indx = count_ref[0]
    safety = 0.001

    def process_pair(j, k, max_indx_local):
        nonlocal count_ref
        if j == k:
            return
        r_j = r[j]
        r_k = r[k]
        # skip parent, children, sister
        sister = -1
        if root_arr[j] != -1:
            sister = children[2*root_arr[j]]
            if sister == j:
                sister = children[2*root_arr[j]+1]
        if (children[2*j] == k or children[2*j+1] == k or
                root_arr[j] == k or sister == k):
            return

        d, c1, c2 = dist3d_seg_to_seg(
            xstart[j], ystart[j], zstart[j], xstop[j], ystop[j], zstop[j],
            xstart[k], ystart[k], zstart[k], xstop[k], ystop[k], zstop[k])
        if d == 0:
            return
        if d >= r_j+r_k:
            return

        # check not same ancestry
        p2 = root_arr[k]
        while p2 != j and p2 != -1:
            p2 = root_arr[p2]
        if p2 != -1:
            return
        p1 = root_arr[j]
        while p1 != k and p1 != -1:
            p1 = root_arr[p1]
        if p1 != -1:
            return

        # choose which to move
        if r_j < r_k:
            closest = c1[:]
            mov_vec = [(c1[i]-c2[i])*(safety+r_j+r_k)/d for i in range(3)]
            indx = j
        else:
            closest = c2[:]
            mov_vec = [(c2[i]-c1[i])*(safety+r_j+r_k)/d for i in range(3)]
            indx = k

        parent = root_arr[indx]

        if isequal(closest[0], closest[1], closest[2], xstart[indx], ystart[indx], zstart[indx]):
            xstop[parent] = closest[0]+mov_vec[0]
            ystop[parent] = closest[1]+mov_vec[1]
            zstop[parent] = closest[2]+mov_vec[2]
            xstart[indx] = closest[0]+mov_vec[0]
            ystart[indx] = closest[1]+mov_vec[1]
            zstart[indx] = closest[2]+mov_vec[2]
            sis = -1
            if children[2*parent] != indx:
                sis = children[2*parent]
            elif children[2*parent+1] != indx:
                sis = children[2*parent+1]
            if sis != -1:
                xstart[sis] = closest[0]+mov_vec[0]
                ystart[sis] = closest[1]+mov_vec[1]
                zstart[sis] = closest[2]+mov_vec[2]
                recheck.append(sis)
            recheck.append(parent)
            recheck.append(indx)

        elif isequal(closest[0], closest[1], closest[2], xstop[indx], ystop[indx], zstop[indx]):
            ch1 = children[2*indx]
            ch2 = children[2*indx+1]
            if ch1 != -1:
                xstart[ch1] = closest[0]+mov_vec[0]
                ystart[ch1] = closest[1]+mov_vec[1]
                zstart[ch1] = closest[2]+mov_vec[2]
                xstart[ch2] = closest[0]+mov_vec[0]
                ystart[ch2] = closest[1]+mov_vec[1]
                zstart[ch2] = closest[2]+mov_vec[2]
                recheck.append(ch1)
                recheck.append(ch2)
            xstop[indx] = closest[0]+mov_vec[0]
            ystop[indx] = closest[1]+mov_vec[1]
            zstop[indx] = closest[2]+mov_vec[2]
            recheck.append(indx)
        else:
            # split branch
            nc = count_ref[0]
            xstart[nc] = closest[0]+mov_vec[0]
            ystart[nc] = closest[1]+mov_vec[1]
            zstart[nc] = closest[2]+mov_vec[2]
            xstop[nc] = xstop[indx]
            ystop[nc] = ystop[indx]
            zstop[nc] = zstop[indx]
            r[nc] = r[indx]
            children[2*nc] = children[2*indx]
            children[2*nc+1] = children[2*indx+1]
            root_arr[nc] = indx
            Qs[nc] = Qs[indx]
            section[nc] = section[indx]
            recheck.append(nc)
            recheck.append(indx)
            xstop[indx] = closest[0]+mov_vec[0]
            ystop[indx] = closest[1]+mov_vec[1]
            zstop[indx] = closest[2]+mov_vec[2]
            if children[2*nc] != -1:
                root_arr[children[2*nc]] = nc
                root_arr[children[2*nc+1]] = nc
            children[2*indx] = nc
            children[2*indx+1] = nc
            count_ref[0] += 1
        return

    for j in range(max_indx):
        if j % 5000 == 0:
            print(f"Checking branch: {j} of {max_indx}")
        for k in range(j+1, max_indx):
            process_pair(j, k, max_indx)

    max_check = len(recheck)*10
    while recheck:
        if len(recheck) > max_check:
            return -1
        j = recheck.popleft()
        for k in range(count_ref[0]):
            if j != k:
                process_pair(j, k, count_ref[0])
    return 0


# ============================================================================
# optimizeAngles + polyInterp + polyInterpWithControlPoints
# ============================================================================
def poly_interp(xstart, ystart, zstart, xstop, ystop, zstop,
                ssx, ssy, ssz, esx, esy, esz,
                lpx, lpy, lpz,
                branch_pts, k, out_res_scale):
    """
    Port of Liver::polyInterp.
    Appends interpolated (x,y,z) tuples to branch_pts[k].
    """
    branch_pts[k].append((xstart, ystart, zstart))

    if k != 0:
        len_lin = math.ceil(out_res_scale * 1.5 * math.sqrt(
            (lpx-xstart)**2+(lpy-ystart)**2+(lpz-zstart)**2))
        if len_lin < 5:
            len_lin = 5
        for i in range(1, len_lin):
            x = xstart + i*(lpx-xstart)/len_lin
            y = ystart + i*(lpy-ystart)/len_lin
            z = zstart + i*(lpz-zstart)/len_lin
            branch_pts[k].append((x, y, z))
        branch_pts[k].append((lpx, lpy, lpz))
    else:
        lpx = xstart
        lpy = ystart
        lpz = zstart

    len_poly = math.ceil(out_res_scale * 1.5 * math.sqrt(
        (xstop-lpx)**2+(ystop-lpy)**2+(zstop-lpz)**2))
    if len_poly < 5:
        len_poly = 5

    # Newton forward difference coefficients (exactly as in C++)
    stPts = [ssx, ssy, ssz]
    linPts = [lpx, lpy, lpz]
    divdif = [xstop-lpx, ystop-lpy, zstop-lpz]
    dzzdx = [divdif[j]-stPts[j] for j in range(3)]
    dzdxdx = [esx-divdif[0], esy-divdif[1], esz-divdif[2]]
    dif1 = [2*dzzdx[j]-dzdxdx[j] for j in range(3)]
    dif2 = [dzdxdx[j]-dzzdx[j] for j in range(3)]

    for i in range(1, len_poly):
        l = i / len_poly
        pt = [0.0]*3
        for j in range(3):
            pt[j] = l*dif2[j] + dif1[j]
            pt[j] = l*pt[j] + stPts[j]
            pt[j] = l*pt[j] + linPts[j]
        branch_pts[k].append((pt[0], pt[1], pt[2]))
    branch_pts[k].append((xstop, ystop, zstop))


def poly_interp_control_pts(ctrl_pts, branch_pts, kk, out_res_scale):
    """
    Port of Liver::polyInterpWithControlPoints.
    Cubic spline through control_pts (list of (x,y,z)).
    """
    n = len(ctrl_pts)
    if n < 2:
        return

    yi = np.array([[p[0], p[1], p[2]] for p in ctrl_pts], dtype=np.float64)
    div = yi[1:] - yi[:-1]       # (n-1, 3)

    # build tridiagonal system
    C = np.zeros((n, n))
    b = np.ones((n, 3))
    for i in range(1, n-1):
        C[i, i] = 4
        C[i, i-1] = 1
        C[i, i+1] = 1
        b[i] = div[i-1]+div[i]
    C[0, 0] = 1
    C[n-1, n-1] = 1
    b[0] = 1.0
    b[n-1] = 1.0

    s = np.linalg.solve(C, b)    # (n,3)
    c4 = s[:-1]+s[1:] - 2*div    # (n-1,3)
    c3 = (div - s[:-1]) - c4     # (n-1,3)

    # c1 layout: [c4.T, c3.T, s[:n-1].T, yi[:n-1].T] reshaped
    c1_T = np.hstack([c4.T, c3.T, s[:n-1].T, yi[:n-1].T])  # (3, 4*(n-1))
    c1 = c1_T.reshape((n-1)*3, 4)

    branch_pts[kk].append(ctrl_pts[0])
    for i in range(n-1):
        seg_len = math.sqrt((yi[i+1, 0]-yi[i, 0])**2 +
                            (yi[i+1, 1]-yi[i, 1])**2+(yi[i+1, 2]-yi[i, 2])**2)
        l = math.ceil(out_res_scale * seg_len)
        if l < 5:
            l = 5
        idx = 3*i
        for jj in range(1, l+1):
            q = jj/l
            x = float(c1[idx,  0])
            y = float(c1[idx+1, 0])
            z = float(c1[idx+2, 0])
            for order in range(1, 4):
                x = q*x + float(c1[idx,  order])
                y = q*y + float(c1[idx+1, order])
                z = q*z + float(c1[idx+2, order])
            branch_pts[kk].append((x, y, z))


def optimize_angles(xstart, xstop, ystart, ystop, zstart, zstop,
                    root_arr, Qs, count, gamma, scale, children, r,
                    pre_push_count, out_res_scale):
    """
    Port of Liver::optimizeAngles.
    Returns branch_pts: list of lists of (x,y,z).
    """
    angles = [0.0]*count
    lpX = [0.0]*count
    lpY = [0.0]*count
    lpZ = [0.0]*count
    ssX = [0.0]*count
    ssY = [0.0]*count
    ssZ = [0.0]*count
    esX = [0.0]*count
    esY = [0.0]*count
    esZ = [0.0]*count
    ctrl_pts = [[] for _ in range(count)]

    # compute Murray angles
    for i in range(1, count):
        rP = r[root_arr[i]]
        r1 = r[i]
        if rP < 1e-9 or r1 < 1e-9:
            continue
        try:
            val = (rP**4 + r1**4 - (rP**gamma - r1**gamma)
                   ** (4.0/gamma)) / (2*rP**2*r1**2)
            val = max(-1.0, min(1.0, val))
            angles[i] = math.acos(val)
        except:
            pass

    # root
    ssX[0] = ssY[0] = ssZ[0] = 0.0
    lpX[0] = -1.0

    for i in range(1, count):
        parent = root_arr[i]
        sister = children[2*parent]
        if sister == i:
            sister = children[2*parent+1]
        if sister == i:
            continue  # pushed branch

        if children[2*i] == -1:  # endpoint
            esX[i] = esY[i] = esZ[i] = 0.0

        pVx = xstop[parent]-xstart[parent]
        pVy = ystop[parent]-ystart[parent]
        pVz = zstop[parent]-zstart[parent]
        cVx = xstop[i]-xstart[i]
        cVy = ystop[i]-ystart[i]
        cVz = zstop[i]-zstart[i]
        sVx = xstop[sister]-xstart[sister]
        sVy = ystop[sister]-ystart[sister]
        sVz = zstop[sister]-zstart[sister]

        # normal = sister × child
        nVx = sVy*cVz - sVz*cVy
        nVy = sVz*cVx - sVx*cVz
        nVz = sVx*cVy - sVy*cVx
        nL = math.sqrt(nVx*nVx + nVy*nVy + nVz*nVz)
        if nL < 1e-9:
            continue
        nVx /= nL
        nVy /= nL
        nVz /= nL

        # project parent into plane
        dot_pn = pVx*nVx + pVy*nVy + pVz*nVz
        ppVx = pVx - nVx*dot_pn
        ppVy = pVy - nVy*dot_pn
        ppVz = pVz - nVz*dot_pn

        # Rodrigues rotation of parent by Murray angle about normal
        crossNPx = nVy*pVz - nVz*pVy
        crossNPy = nVz*pVx - nVx*pVz
        crossNPz = nVx*pVy - nVy*pVx
        cos_a = math.cos(angles[i])
        sin_a = math.sin(angles[i])
        rotVx = ppVx*cos_a + crossNPx*sin_a + nVx*dot_pn*(1-cos_a)
        rotVy = ppVy*cos_a + crossNPy*sin_a + nVy*dot_pn*(1-cos_a)
        rotVz = ppVz*cos_a + crossNPz*sin_a + nVz*dot_pn*(1-cos_a)
        rLen = math.sqrt(rotVx*rotVx + rotVy*rotVy + rotVz*rotVz)
        if rLen < 1e-9:
            continue

        angle_sis = angles[sister] if sister < len(angles) else 0.0
        sep_dist = (r[i]+r[sister])*1.01
        denom = math.sqrt(max(0.0, 2.0 - 2.0*math.cos(angles[i]+angle_sis)))
        dist_val = sep_dist/denom if denom > 1e-9 else 0.0
        cLen = math.sqrt(cVx*cVx + cVy*cVy + cVz*cVz)
        factor = 1.0
        if dist_val > cLen/2.0:
            dist_val = cLen/2.0
            factor = 2.0

        lpX[i] = xstop[parent] + (rotVx/rLen)*dist_val
        lpY[i] = ystop[parent] + (rotVy/rLen)*dist_val
        lpZ[i] = zstop[parent] + (rotVz/rLen)*dist_val

        ppLen = math.sqrt(ppVx*ppVx + ppVy*ppVy + ppVz*ppVz)
        pLen = math.sqrt(pVx*pVx + pVy*pVy + pVz*pVz)

        ssX[i] = (rotVx/rLen)*cLen*factor
        ssY[i] = (rotVy/rLen)*cLen*factor
        ssZ[i] = (rotVz/rLen)*cLen*factor
        if ppLen > 1e-9:
            esX[parent] = (ppVx/ppLen)*pLen
            esY[parent] = (ppVy/ppLen)*pLen
            esZ[parent] = (ppVz/ppLen)*pLen

    # build control points for pushed branches
    for i in range(pre_push_count-5 if pre_push_count > 5 else 0, count):
        master = root_arr[i]
        if master != -1 and children[2*master] != 0 and children[2*master] == children[2*master+1]:
            lpX[master] = -1.0
            ctrl_pts[master] = [(xstart[master], ystart[master], zstart[master]),
                                (xstart[i],    ystart[i],    zstart[i])]
            xstop[master] = xstop[i]
            ystop[master] = ystop[i]
            zstop[master] = zstop[i]
            esX[master] = esX[i]
            esY[master] = esY[i]
            esZ[master] = esZ[i]
            ctrl_pts[master].append((xstop[i], ystop[i], zstop[i]))

    # polynomial interpolation
    branch_pts = [[] for _ in range(count)]
    for i in range(pre_push_count):
        if lpX[i] != -1 or i == 0:
            poly_interp(xstart[i], ystart[i], zstart[i],
                        xstop[i], ystop[i], zstop[i],
                        ssX[i], ssY[i], ssZ[i],
                        esX[i], esY[i], esZ[i],
                        lpX[i], lpY[i], lpZ[i],
                        branch_pts, i, out_res_scale)
        else:
            poly_interp_control_pts(ctrl_pts[i], branch_pts, i, out_res_scale)

    return branch_pts


# ============================================================================
# build_tree — main entry point (port of Liver::build_tree)
# ============================================================================
def build_tree(Nx, Ny, Nz, terminal_pts, tree_number,
               scale, seed, out_res_scale, perf_path, out_dir,
               progress_interval=0):
    """
    Full port of Liver::build_tree.
    Reads Phantom.dat, runs create_tree, push_branches, optimizeAngles,
    writes Tree<N>.txt and Radii<N>.txt.
    Returns 0 on success, -1 if stuck.
    """
    mt_rng = CppMT19937(seed)

    dims = [Nx, Ny, Nz, 10]
    gamma = 2.8 + 0.3 * mt_rng.random()

    # load blood demand map
    n_voxels = 10 * Nx * Ny * Nz
    perf = np.fromfile(perf_path, dtype=np.uint8, count=n_voxels)
    if len(perf) != n_voxels:
        raise IOError(f"Expected {n_voxels} bytes, got {len(perf)}")

    factor = 1.2
    arrayAlloc = math.ceil(terminal_pts * 2 * factor) + 10
    xstart = [0.0]*arrayAlloc
    xstop = [0.0]*arrayAlloc
    ystart = [0.0]*arrayAlloc
    ystop = [0.0]*arrayAlloc
    zstart = [0.0]*arrayAlloc
    zstop = [0.0]*arrayAlloc
    children = [-1]*(2*arrayAlloc)
    root_arr = [-1]*arrayAlloc
    section = [0]*arrayAlloc
    length = [0.0]*arrayAlloc
    Qs = [1]*arrayAlloc
    r_arr = [0.0]*arrayAlloc
    V = [0.0]*arrayAlloc
    regions = [deque() for _ in range(8)]  # one per Couinaud segment

    seg_number = 7  # 8 segments, C++ counts from 0 so uses 7

    # --- initial branch: segment 8 start → segment 3 stop ---
    init_secs = [8, 3]  # 0-indexed segment layers
    for pos in range(2):
        found = False
        while not found:
            k = mt_rng.random()*Nz - 0.5
            j = mt_rng.random()*Ny - 0.5
            i = mt_rng.random()*Nx - 0.5
            val = 254.5 * mt_rng.random()
            idx = linear_ind(int(i), int(j), int(k), init_secs[pos], dims)
            if val < perf[idx]:
                if pos == 0:
                    xstart[0] = i
                    ystart[0] = j
                    zstart[0] = k
                else:
                    xstop[0] = i
                    ystop[0] = j
                    zstop[0] = k
                perf[idx] = 255  # mark visited
                found = True

    # --- one endpoint per segment ---
    for pos in range(min(seg_number, terminal_pts)):
        found = False
        while not found:
            k = mt_rng.random()*Nz - 0.5
            j = mt_rng.random()*Ny - 0.5
            i = mt_rng.random()*Nx - 0.5
            val = 254.5 * mt_rng.random()
            idx = linear_ind(int(i), int(j), int(k), pos, dims)
            if val < perf[idx]:
                point = [i, j, k, 0.0]
                regions[pos].append(point)
                found = True

    # --- remaining endpoints sampled uniformly + section-tagged ---
    pos = seg_number
    while pos < terminal_pts:
        found = False
        advance_pos = True
        while not found:
            k = mt_rng.random()*Nz - 0.5
            j = mt_rng.random()*Ny - 0.5
            i = mt_rng.random()*Nx - 0.5
            val = 254.5 * mt_rng.random()
            idx = linear_ind(int(i), int(j), int(k), dims[3]-1, dims)
            if val < perf[idx]:
                point = [i, j, k, 0.0]
                sec = get_section(int(i), int(j), int(k), perf, mt_rng, dims)
                if sec != 0:
                    regions[sec-1].append(point)
                else:
                    advance_pos = False
                found = True
        if advance_pos:
            pos += 1

    # --- grow tree ---
    print(f"Building Tree {tree_number} with {terminal_pts} points", flush=True)
    count = create_tree(
        xstart, xstop, ystart, ystop, zstart, zstop,
        root_arr, section, length, Qs, V,
        regions, gamma, terminal_pts, children,
        perf, mt_rng, dims, arrayAlloc, scale,
        progress_interval=progress_interval)

    # --- assign radii ---
    Qscale = 1.0 / (Qs[0]**(1.0/gamma))
    for t in range(count):
        r_arr[t] = Qscale * (Qs[t]**(1.0/gamma)) * scale

    pre_push_count = count
    count_ref = [count]

    # --- push intersecting branches ---
    ret = push_branches(xstart, xstop, ystart, ystop, zstart, zstop,
                        root_arr, Qs, section, count_ref,
                        Qscale, gamma, scale, terminal_pts, children, r_arr)
    if ret != 0:
        return -1
    count = count_ref[0]

    # --- bifurcation angle optimisation + interpolation ---
    branch_pts = optimize_angles(
        xstart, xstop, ystart, ystop, zstart, zstop,
        root_arr, Qs, count, gamma, scale, children, r_arr,
        pre_push_count, out_res_scale)

    # --- write output ---
    os.makedirs(out_dir, exist_ok=True)
    tree_path = os.path.join(out_dir, f"Tree{tree_number}.txt")
    radii_path = os.path.join(out_dir, f"Radii{tree_number}.txt")

    with open(tree_path, 'w') as fh:
        for t in range(pre_push_count):
            pts = branch_pts[t]
            fh.write(' '.join(f"{p[0]} {p[1]} {p[2]}" for p in pts))
            fh.write('\n')

    with open(radii_path, 'w') as fh:
        for t in range(pre_push_count):
            fh.write(f"{r_arr[t]}\n")

    print(f"Wrote {tree_path} and {radii_path}", flush=True)
    return 0


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Hepatic vessel generation (Python port of Whitehead C++)")
    p.add_argument("--phantom",      required=True,
                   help="Path to Phantom.dat (blood demand map)")
    p.add_argument("--Nx",           type=int, required=True)
    p.add_argument("--Ny",           type=int, required=True)
    p.add_argument("--Nz",           type=int, required=True)
    p.add_argument("--endpoints",    type=int, default=5000)
    p.add_argument("--trees",        type=int, default=1)
    p.add_argument("--phantom_res",  type=float, default=0.77)
    p.add_argument("--desired_res",  type=float, default=0.3)
    p.add_argument("--out",          default="CLines")
    p.add_argument("--seed",         type=int, default=None)
    p.add_argument("--progress_interval", type=int, default=0,
                   help="Report growth progress every N connected endpoints; 0 chooses about 1%% intervals")
    args = p.parse_args()

    initial_res = 1.54
    base_rng = CppMT19937(args.seed if args.seed else int(time.time()))
    out_res_scale = args.phantom_res / args.desired_res

    for i in range(1, args.trees+1):
        seed_i = base_rng.random_uint()
        scale = (1.3 + 1.0*base_rng.random()) * \
            (initial_res / args.phantom_res)
        print(
            f"\n=== Tree {i}/{args.trees}  seed={seed_i}  scale={scale:.4f} ===")
        ret = build_tree(args.Nx, args.Ny, args.Nz,
                         args.endpoints, i,
                         scale, seed_i,
                         out_res_scale,
                         args.phantom,
                         args.out,
                         progress_interval=args.progress_interval)
        while ret != 0:
            print("Got stuck — retrying...")
            seed_i = base_rng.random_uint()
            ret = build_tree(args.Nx, args.Ny, args.Nz,
                             args.endpoints, i,
                             scale, seed_i,
                             out_res_scale,
                             args.phantom,
                             args.out,
                             progress_interval=args.progress_interval)
    print(f"\nSuccessfully generated {args.trees} vessel tree(s).")
