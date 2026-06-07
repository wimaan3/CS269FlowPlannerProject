"""Frenet coordinate projection utilities.

Used by ModelInputProcessor when kinematic='frenet'. Converts
(x, y) Cartesian trajectories to (s, d) Frenet coordinates relative
to a reference centerline polyline, and inverts the projection for eval.

All tensors are batched, on the same device, and dtype float32.

v8 Frenet-fix env vars (opt-in via environment; default behavior unchanged):
  - FRENET_TANH_D=1
        cartesian_to_frenet returns (s, tanh(d / FRENET_TANH_D_SCALE)) instead of (s, d).
        frenet_to_cartesian inverts via d = SCALE * atanh(d_compressed.clamp(±0.999)).
        FRENET_TANH_D_SCALE defaults to 3.0 m (Hallgarten IV 2024 recommendation).
        Rationale: physical |d| is <2 m for in-lane driving but tail extends to
        ±100 m due to centerline mis-selection. tanh squashes the tail so MSE
        does not dominate the bulk.
  - FRENET_SMART_CENTERLINE=1
        select_reference_centerline considers ALL valid lanes (not just route),
        anchored by ego_past trajectory rather than ego_current alone. Picks
        the lane most aligned with ego's recent motion.
"""
from __future__ import annotations

import os
import numpy as np
import torch


# ---- v8 Frenet-fix knobs (read once per import; cheap to call repeatedly) ----
def _tanh_d_scale() -> float | None:
    """Return the tanh-compression scale m (e.g., 3.0) if FRENET_TANH_D is set, else None."""
    if os.environ.get('FRENET_TANH_D', '').lower() in ('1', 'true', 'yes', 'on'):
        try:
            return float(os.environ.get('FRENET_TANH_D_SCALE', '3.0'))
        except ValueError:
            return 3.0
    return None


def _smart_centerline_enabled() -> bool:
    return os.environ.get('FRENET_SMART_CENTERLINE', '').lower() in ('1', 'true', 'yes', 'on')


# Indices into the 12-dim per-lane feature vector.
# See flow_planner/data/data_process/map_process.py:lane_polyline_process
LANE_XY_IDX = slice(0, 2)        # (x, y) of centerline point
LANE_GEOMETRY_IDX = slice(0, 8)  # the 8 geometry features (xy + tangent + boundaries)


def _lane_is_valid(lanes: torch.Tensor) -> torch.Tensor:
    """Check which lanes have non-zero geometry (i.e. are not padding).

    Args:
        lanes: (B, N_lanes, N_points, D) with D >= 8.

    Returns:
        (B, N_lanes) bool tensor — True where the lane has any non-zero
        geometric data.
    """
    return torch.any(lanes[..., LANE_GEOMETRY_IDX] != 0, dim=(-2, -1))


def select_reference_centerline(
    route_lanes: torch.Tensor,
    lanes: torch.Tensor,
    ego_xy: torch.Tensor | None = None,
    out_points: int = 100,
    max_extent: float = 250.0,
    ego_past_xy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a horizon-covering reference centerline by concatenating route lanes.

    Why this exists
    ---------------
    Earlier versions returned a *single* lane polyline (length ~60 m). Ego's 8 s
    future trajectory covers 60-120 m, so a single lane could not span the
    horizon: the projection bottomed out at the lane's end with huge `d` values
    (max 99 m observed). The Frenet transform is only geometrically valid for
    `d < 1/κ` (Li et al., IEEE T-ITS 2022), so single-lane references put us
    outside the valid regime for most scenarios.

    This version concatenates the on-route lane polylines into one long
    polyline (greedy nearest-end ordering), so `d` stays small everywhere ego
    actually drives. No model architecture change — purely a data-side fix.

    Strategy per batch element:
        1. Collect valid route lanes' (x, y) point sets.
        2. Seed: pick the route lane with the point closest to ego.
        3. Greedy extension: repeatedly append the unused route lane whose
           start is closest to the current path's end (forward extension only).
        4. If no route lanes are valid, fall back to the nearest non-route lane.
        5. Resample the concatenated polyline to a fixed `out_points` length so
           cartesian_to_frenet sees a uniform tensor shape across the batch.

    Args:
        route_lanes: (B, N_route, N_points_lane, D) — planned route lanes.
        lanes:       (B, N_lanes, N_points_lane, D) — all visible lanes (fallback).
        ego_xy:      optional (B, 2). Defaults to origin (ego frame).
        out_points:  number of points in the returned centerline polyline.
        max_extent:  cap the total polyline length (m) to bound compute.

    Returns:
        (B, out_points, 2) tensor of centerline (x, y) coordinates.
    """
    B, N_route, N_points_lane, _ = route_lanes.shape
    device = route_lanes.device
    dtype = route_lanes.dtype

    if ego_xy is None:
        ego_xy = torch.zeros((B, 2), device=device, dtype=dtype)

    # v8 Frenet-fix: when FRENET_SMART_CENTERLINE is set, pick the centerline
    # from ALL lanes (not just route lanes), anchored by ego's recent past
    # trajectory. The route-only heuristic picks lanes the route planner
    # WANTS ego to use, which can differ from the lane ego is actually in
    # (mid-lane-change, intersection, off-route) — driving d-std to 9.4 m
    # (range ±100 m) due to centerline mis-selection. Anchoring by ego_past
    # selects the lane ego is actually following.
    smart = _smart_centerline_enabled()
    if smart and ego_past_xy is not None:
        # Concatenate route + non-route lanes so the smart selector sees all
        # candidates. Build a synthetic "all valid lanes" route_lanes-shaped
        # tensor by stacking.
        # The downstream loop body uses route_xy/route_valid as the primary
        # source and lanes_xy/lanes_valid only as fallback. To make smart mode
        # consider both equally, we feed the union to the route_xy slot.
        route_lanes_eff = torch.cat([route_lanes, lanes], dim=1)   # (B, N_route+N_lanes, ...)
    else:
        route_lanes_eff = route_lanes

    route_valid = _lane_is_valid(route_lanes_eff)        # (B, N_route_eff)
    lanes_valid = _lane_is_valid(lanes)                  # (B, N_lanes)

    # Default: a long straight line along +x at 1 m spacing. Used when no valid
    # route or visible lanes exist.
    default_centerline = torch.zeros((B, out_points, 2), device=device, dtype=dtype)
    default_centerline[:, :, 0] = torch.linspace(0.0, float(out_points - 1), out_points, device=device, dtype=dtype)

    # Build per-sample centerlines on CPU then stack. This is per-batch-element
    # Python work, but the loop is small (B ~ 4-32) and the cost is negligible
    # vs the model forward pass.
    out_list = []
    # Use route_lanes_eff so smart mode sees the union of route + non-route lanes.
    route_xy_cpu = route_lanes_eff[..., :2].detach().cpu().numpy()
    lanes_xy_cpu = lanes[..., :2].detach().cpu().numpy()
    route_valid_cpu = route_valid.detach().cpu().numpy()
    lanes_valid_cpu = lanes_valid.detach().cpu().numpy()
    # When smart mode is on and ego_past_xy is given, use the recent past as
    # the anchor instead of ego_current (which is always at the origin in the
    # ego frame). The recent-past anchor biases selection toward the lane ego
    # is actually FOLLOWING, not the lane the route planner WANTS.
    if smart and ego_past_xy is not None:
        # Average over the most recent positions for a stable anchor.
        # ego_past_xy: (B, T_past, 2). Use last min(5, T_past) timesteps.
        T_past = ego_past_xy.shape[1]
        n_recent = min(5, T_past)
        anchor = ego_past_xy[:, -n_recent:, :].mean(dim=1)  # (B, 2)
        ego_cpu = anchor.detach().cpu().numpy()
    else:
        ego_cpu = ego_xy.detach().cpu().numpy()

    for b in range(B):
        polyline = _build_horizon_polyline(
            route_xy=route_xy_cpu[b],
            route_valid=route_valid_cpu[b],
            lanes_xy=lanes_xy_cpu[b],
            lanes_valid=lanes_valid_cpu[b],
            ego_xy=ego_cpu[b],
            out_points=out_points,
            max_extent=max_extent,
        )
        out_list.append(polyline)

    out = torch.from_numpy(np.stack(out_list, axis=0)).to(device=device, dtype=dtype)

    # Safety: replace any all-zero rows with the default centerline.
    has_content = (out.abs().sum(dim=(-1, -2)) > 1e-3)                # (B,)
    out = torch.where(has_content[:, None, None], out, default_centerline)
    return out


def _build_horizon_polyline(
    route_xy,           # np (N_route, N_points_lane, 2)
    route_valid,        # np (N_route,) bool
    lanes_xy,           # np (N_lanes, N_points_lane, 2)
    lanes_valid,        # np (N_lanes,) bool
    ego_xy,             # np (2,)
    out_points: int,
    max_extent: float,
):
    """Greedy nearest-end concatenation of route lanes into one polyline.

    Falls back to the nearest visible lane if no route lanes are valid.
    Returns np array of shape (out_points, 2).
    """
    valid_route_idx = np.where(route_valid)[0]

    if len(valid_route_idx) == 0:
        # No route lanes — fall back to single nearest visible lane.
        return _fallback_nearest_lane(lanes_xy, lanes_valid, ego_xy, out_points)

    # Filter out any all-zero rows inside otherwise-valid lanes (padding inside lanes).
    candidates = []
    for i in valid_route_idx:
        pts = route_xy[i]                                              # (N_points_lane, 2)
        nonzero_mask = np.any(pts != 0, axis=-1)
        if nonzero_mask.sum() < 2:
            continue
        candidates.append(pts[nonzero_mask])
    if not candidates:
        return _fallback_nearest_lane(lanes_xy, lanes_valid, ego_xy, out_points)

    # Seed: route lane with the point closest to ego.
    def lane_min_dist_to(p_target, lane_pts):
        return np.min(np.linalg.norm(lane_pts - p_target[None, :], axis=-1))

    seed_costs = [lane_min_dist_to(ego_xy, c) for c in candidates]
    seed_idx = int(np.argmin(seed_costs))
    used = [seed_idx]
    polyline = candidates[seed_idx].copy()

    # Orient the seed so it points "away from ego" along its longer extent.
    if np.linalg.norm(polyline[0] - ego_xy) > np.linalg.norm(polyline[-1] - ego_xy):
        polyline = polyline[::-1]

    # Greedy extension: append lanes whose start is closest to the current tail,
    # up to max_extent meters of total arc length.
    def polyline_arc_length(p):
        if len(p) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())

    while polyline_arc_length(polyline) < max_extent and len(used) < len(candidates):
        tail = polyline[-1]
        best, best_dist, best_orient = None, np.inf, None
        for i, c in enumerate(candidates):
            if i in used:
                continue
            d_start = np.linalg.norm(c[0] - tail)
            d_end = np.linalg.norm(c[-1] - tail)
            if min(d_start, d_end) < best_dist:
                best_dist = min(d_start, d_end)
                best = i
                best_orient = 'fwd' if d_start <= d_end else 'rev'
        if best is None or best_dist > 30.0:
            break  # no nearby continuation
        used.append(best)
        nxt = candidates[best].copy()
        if best_orient == 'rev':
            nxt = nxt[::-1]
        # Skip the first point of nxt if it's near tail to avoid duplicate vertex
        if np.linalg.norm(nxt[0] - tail) < 1e-3:
            nxt = nxt[1:]
        if len(nxt) > 0:
            polyline = np.concatenate([polyline, nxt], axis=0)

    # Resample to a uniform out_points length so the returned tensor shape is consistent.
    return _resample_polyline(polyline, out_points)


def _fallback_nearest_lane(lanes_xy, lanes_valid, ego_xy, out_points):
    """Last-resort: pick the single nearest visible lane and resample."""
    valid_idx = np.where(lanes_valid)[0]
    if len(valid_idx) == 0:
        # Return a straight +x ray
        ray = np.zeros((out_points, 2), dtype=np.float32)
        ray[:, 0] = np.arange(out_points, dtype=np.float32)
        return ray
    best, best_d = None, np.inf
    for i in valid_idx:
        pts = lanes_xy[i]
        nz = np.any(pts != 0, axis=-1)
        if nz.sum() < 2:
            continue
        d = np.min(np.linalg.norm(pts[nz] - ego_xy[None, :], axis=-1))
        if d < best_d:
            best_d = d
            best = pts[nz]
    if best is None:
        ray = np.zeros((out_points, 2), dtype=np.float32)
        ray[:, 0] = np.arange(out_points, dtype=np.float32)
        return ray
    return _resample_polyline(best, out_points)


def _resample_polyline(polyline, out_points: int):
    """Resample a (N, 2) polyline uniformly by arc length to (out_points, 2)."""
    polyline = np.asarray(polyline, dtype=np.float32)
    if len(polyline) < 2:
        # Degenerate — fan out into a straight ray from the single point.
        ray = np.zeros((out_points, 2), dtype=np.float32)
        ray[:] = polyline[0] if len(polyline) > 0 else (0.0, 0.0)
        ray[:, 0] += np.arange(out_points, dtype=np.float32)
        return ray
    seg_len = np.linalg.norm(np.diff(polyline, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        ray = np.zeros((out_points, 2), dtype=np.float32)
        ray[:] = polyline[0]
        ray[:, 0] += np.arange(out_points, dtype=np.float32)
        return ray
    targets = np.linspace(0.0, total, out_points)
    out = np.empty((out_points, 2), dtype=np.float32)
    for j, t in enumerate(targets):
        i = int(np.searchsorted(cum, t, side='right') - 1)
        i = max(0, min(i, len(polyline) - 2))
        seg = cum[i + 1] - cum[i]
        alpha = 0.0 if seg < 1e-6 else (t - cum[i]) / seg
        out[j] = polyline[i] * (1 - alpha) + polyline[i + 1] * alpha
    return out


def cartesian_to_frenet(
    xy: torch.Tensor,
    centerline: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project Cartesian (x, y) points onto a reference centerline.

    For each point and each centerline segment, computes the perpendicular
    projection onto the segment (clamped to segment bounds), then picks the
    segment with minimum distance to the point.

    Args:
        xy: (B, T, 2) — trajectory in Cartesian.
        centerline: (B, N, 2) — reference centerline.
        eps: tolerance for zero-length segments.

    Returns:
        (B, T, 2) tensor — (s, d) per timestep. s is signed arc length from
        the start of the centerline. d is signed perpendicular offset
        (positive = LEFT of direction of travel along the centerline).
    """
    B, T, _ = xy.shape
    _, N, _ = centerline.shape
    device = xy.device

    # Segment starts and ends. There are N-1 segments.
    seg_start = centerline[:, :-1, :]            # (B, N-1, 2)
    seg_end   = centerline[:, 1:, :]             # (B, N-1, 2)
    seg_vec   = seg_end - seg_start              # (B, N-1, 2)
    seg_len_sq = (seg_vec ** 2).sum(dim=-1)      # (B, N-1)
    seg_len_sq = seg_len_sq.clamp(min=eps)
    seg_len    = seg_len_sq.sqrt()               # (B, N-1)

    # Cumulative arc length along the centerline up to seg_start of each segment.
    seg_cumlen = torch.cat([
        torch.zeros((B, 1), device=device, dtype=xy.dtype),
        seg_len.cumsum(dim=-1)[:, :-1]
    ], dim=-1)                                   # (B, N-1)

    # For each point and each segment, compute projection parameter t ∈ [0, 1].
    # Broadcasting: (B, T, 1, 2) - (B, 1, N-1, 2) → (B, T, N-1, 2).
    rel = xy.unsqueeze(2) - seg_start.unsqueeze(1)             # (B, T, N-1, 2)
    dot = (rel * seg_vec.unsqueeze(1)).sum(dim=-1)             # (B, T, N-1)
    t_param = (dot / seg_len_sq.unsqueeze(1)).clamp(0.0, 1.0)  # (B, T, N-1)

    # Projected point on the segment.
    proj = seg_start.unsqueeze(1) + t_param.unsqueeze(-1) * seg_vec.unsqueeze(1)  # (B, T, N-1, 2)
    dist_vec = xy.unsqueeze(2) - proj                                              # (B, T, N-1, 2)
    dist = torch.linalg.vector_norm(dist_vec, dim=-1)                              # (B, T, N-1)

    # Pick the segment with minimum distance.
    best_seg = dist.argmin(dim=-1)                                                 # (B, T)

    # Gather everything for the best segment.
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, T)
    time_idx  = torch.arange(T, device=device)[None, :].expand(B, T)

    best_t           = t_param[batch_idx, time_idx, best_seg]                      # (B, T)
    best_seg_vec     = seg_vec[batch_idx, best_seg]                                # (B, T, 2)
    best_seg_len     = seg_len[batch_idx, best_seg]                                # (B, T)
    best_cumlen      = seg_cumlen[batch_idx, best_seg]                             # (B, T)
    best_dist_vec    = dist_vec[batch_idx, time_idx, best_seg]                     # (B, T, 2)

    # s = cumulative arc length up to seg_start + t * seg_len
    s = best_cumlen + best_t * best_seg_len                                        # (B, T)

    # d = signed perpendicular distance. Sign is positive if point is LEFT of
    # the segment direction (i.e. cross product of seg_vec and dist_vec is positive).
    cross = best_seg_vec[..., 0] * best_dist_vec[..., 1] - best_seg_vec[..., 1] * best_dist_vec[..., 0]
    dist_magnitude = torch.linalg.vector_norm(best_dist_vec, dim=-1)
    d = torch.sign(cross) * dist_magnitude                                         # (B, T)

    # v8 Frenet-fix: optionally compress d into a bounded range via tanh.
    # The bulk of d is |d|<2 m (in-lane) but the tail extends to ±100 m due
    # to centerline mis-selection. tanh(d/SCALE) squashes the tail so MSE does
    # not dominate the bulk. At inference, frenet_to_cartesian inverts via
    # d = SCALE * atanh(d_compressed.clamp(±0.999)).
    _scale = _tanh_d_scale()
    if _scale is not None and _scale > 0:
        d = torch.tanh(d / _scale)

    return torch.stack([s, d], dim=-1)                                             # (B, T, 2)


def frenet_to_cartesian(
    sd: torch.Tensor,
    centerline: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Invert Frenet (s, d) projection back to Cartesian (x, y).

    For each (s, d), find the centerline segment containing arc length s,
    interpolate position along it, then offset perpendicular to the tangent.

    Args:
        sd: (B, T, 2) — Frenet trajectory.
        centerline: (B, N, 2) — same centerline used for forward projection.
        eps: tolerance for zero-length segments.

    Returns:
        (B, T, 2) tensor — (x, y) per timestep.
    """
    B, T, _ = sd.shape
    _, N, _ = centerline.shape
    device = sd.device

    seg_start  = centerline[:, :-1, :]
    seg_end    = centerline[:, 1:, :]
    seg_vec    = seg_end - seg_start                  # (B, N-1, 2)
    seg_len    = torch.linalg.vector_norm(seg_vec, dim=-1).clamp(min=eps)   # (B, N-1)

    # Cumulative arc length at seg_start of each segment.
    seg_cumlen_start = torch.cat([
        torch.zeros((B, 1), device=device, dtype=sd.dtype),
        seg_len.cumsum(dim=-1)[:, :-1]
    ], dim=-1)                                        # (B, N-1)

    # Cumulative arc length at seg_end of each segment.
    seg_cumlen_end = seg_cumlen_start + seg_len       # (B, N-1)
    total_arc_length = seg_cumlen_end[:, -1:]         # (B, 1)

    s = sd[..., 0]                                    # (B, T)
    d = sd[..., 1]                                    # (B, T)

    # v8 Frenet-fix: invert tanh compression if it was applied at projection time.
    # cartesian_to_frenet wrote d_compressed = tanh(d / SCALE); recover d via
    # d = SCALE * atanh(d_compressed.clamp(±0.999)). The clamp is necessary
    # because the model can predict |d_compressed| slightly above 1 (since the
    # MSE-trained head isn't constrained to the tanh range exactly).
    _scale = _tanh_d_scale()
    if _scale is not None and _scale > 0:
        d = _scale * torch.atanh(d.clamp(-0.999, 0.999))

    # Clamp s to [0, total_arc_length] to handle out-of-range predictions.
    s = s.clamp(min=torch.zeros_like(s))
    s = torch.minimum(s, total_arc_length.expand_as(s))

    # For each (B, T), find the segment index i where seg_cumlen_start[i] <= s <= seg_cumlen_end[i].
    # bucketize requires monotonic sorted boundaries (which they are).
    # We use right boundaries (seg_cumlen_end) - 1 to find the segment.
    # Equivalent: searchsorted on seg_cumlen_end with s gives the segment index.
    # Need to expand seg_cumlen_end to (B, T, N-1) for per-sample lookup.
    # Use a batched searchsorted via comparison:
    # idx = number of seg_cumlen_end values that are strictly less than s.
    # Cap at N-2 (last valid segment index).
    boundaries = seg_cumlen_end.unsqueeze(1)          # (B, 1, N-1)
    s_exp = s.unsqueeze(-1)                           # (B, T, 1)
    seg_idx = (boundaries < s_exp).sum(dim=-1)        # (B, T)
    seg_idx = seg_idx.clamp(max=N - 2)

    batch_idx = torch.arange(B, device=device)[:, None].expand(B, T)
    best_start    = seg_start[batch_idx, seg_idx]                # (B, T, 2)
    best_seg_vec  = seg_vec[batch_idx, seg_idx]                  # (B, T, 2)
    best_seg_len  = seg_len[batch_idx, seg_idx]                  # (B, T)
    best_cumlen_start = seg_cumlen_start[batch_idx, seg_idx]     # (B, T)

    # Local parameter along the segment.
    t_local = ((s - best_cumlen_start) / best_seg_len).clamp(0.0, 1.0)   # (B, T)

    # Interpolated centerline position at arc length s.
    center_at_s = best_start + t_local.unsqueeze(-1) * best_seg_vec      # (B, T, 2)

    # Unit tangent and left-handed normal.
    tangent = best_seg_vec / best_seg_len.unsqueeze(-1)                  # (B, T, 2)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)    # (B, T, 2)

    # Offset perpendicular to tangent.
    xy = center_at_s + d.unsqueeze(-1) * normal                          # (B, T, 2)

    return xy
