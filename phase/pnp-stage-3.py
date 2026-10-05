"""
Stage-3 plug-and-play phase-error SDP: certified continuum cell relaxation.

This file implements the Stage-3 chain used in main.tex:

    p_{m_A,m_B}
        -- exact push-forward u_X = tau m_X -->
    nu_tau on (tau Z_{\ge 0})^2
        -- drop the lattice support -->
    arbitrary continuum measure nu on R_+^2
        -- certified cell enclosure -->
    finite cell masses q_j = nu(C_j).

The physical mean-energy bound is carried through the same chain:

    E[M_X] <= n_bar/(2 tau)
        <=>  integral u_X d nu_tau <= n_bar/2
        -->  integral u_X d nu <= n_bar/2
        -->  sum_j lower_u_{X,j} q_j <= n_bar/2.

The first arrow is exact. Dropping the lattice support is an outer
relaxation. The final cell representation is also an outer relaxation:
q_j is the mass somewhere inside cell C_j, not a point mass at a chosen
representative c_j.

For each state pair s,t the exact finite-tau continuum extension is

    K_tau^{st}(u_A,u_B)
      = exp[b_A^{st} u_A + b_B^{st} u_B],

    b_X^{st}
      = Log(1 - tau + tau exp(i Delta phi_X^{st})) / tau.

It agrees exactly with the physical binomial-thinning kernel on the lattice.
For a representative c_j in cell C_j,

    integral_{C_j} K_tau dnu
      = q_j K_tau(c_j) + r_{st,j},

with

    |r_{st,j}| <= q_j delta_{st,j},

where delta_{st,j} rigorously bounds the kernel variation over the whole
cell. The omitted continuum tail outside D_U = {u_A+u_B <= U_MAX} contributes
at most eps_tail because |K_tau| <= 1.

Hence the finite SOC constraint is

    | sum_z G[(z,s),(z,t)] - sum_j q_j K_tau^{st}(c_j) |
       <= sum_j q_j delta_{st,j} + eps_tail + SLACK_FLOOR.

Every feasible continuum measure maps to a feasible point of this finite
problem, so maximizing the phase error gives a certified outer bound.

The limiting kernel K_0 remains useful analytically for the mixed-Poisson
interpretation, but is not used in the certified numerical constraints here.
"""

import math

import cvxpy as cp
import numpy as np


# --------------------------------------------------------------------------
# Protocol / channel parameters

M = 4
F_EC = 1.15
ETA_DET = 0.85
XI = 0.2
E_ALI = 0.015
P_DC = 5e-8
LOSS_ONLY = False

# Trusted attenuation transmissivity tau = t^2.
T2 = 1e-8

# Continuum truncation:
#   D_U = {(u_A,u_B) >= 0 : u_A + u_B <= U_MAX}.
#
# CELL_MIN is the first positive cell boundary. The remaining boundaries are
# logarithmic up to U_MAX. N_U is the number of intervals per coordinate
# before clipping the rectangles against the triangular domain.
U_MAX = 20.0
CELL_MIN = 1e-4
N_U = 60

# Tiny numerical allowance, independent of the physical cell/tail slacks.
SLACK_FLOOR = 1e-12

Z_P, Z_M, Z_F = 0, 1, 2
enc_phases = [(0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)]


# --------------------------------------------------------------------------
# Basic helpers

def state_phases(M):
    out = []
    for m in range(M):
        phi = np.pi * m / M
        for dA, dB in enc_phases:
            out.append((phi + dA, phi + dB))
    return out


def h2(e):
    return 0.0 if e <= 0 or e >= 1 else (
        -e * math.log2(e) - (1 - e) * math.log2(1 - e)
    )


def channel(n_bar, L):
    eta = ETA_DET * 10 ** (-XI * L / 10.0)
    p_dc = 0.0 if LOSS_ONLY else P_DC
    e_ali = 0.0 if LOSS_ONLY else E_ALI
    P_sig = 1.0 - np.exp(-eta * n_bar)
    P_pass = P_sig + p_dc
    return P_pass, (e_ali * P_sig + 0.5 * p_dc) / P_pass


# --------------------------------------------------------------------------
# Certified continuum cells

def u_edges(u_max=U_MAX, n_u=N_U, cell_min=CELL_MIN):
    """
    Coordinate boundaries for the cell partition.

    Returns n_u + 1 boundaries, starting at exactly zero and ending at u_max.
    The first cell is [0, cell_min]; all later boundaries are logarithmic.
    """
    if not (u_max > 0):
        raise ValueError("U_MAX must be positive.")
    if n_u < 2:
        raise ValueError("N_U must be at least 2.")
    if not (0 < cell_min < u_max):
        raise ValueError("CELL_MIN must satisfy 0 < CELL_MIN < U_MAX.")

    positive = np.logspace(
        math.log10(cell_min), math.log10(u_max), n_u
    )
    return np.concatenate(([0.0], positive))


def _clip_polygon_sum_leq(vertices, u_max, tol=1e-14):
    """
    Clip a convex polygon against x + y <= u_max.

    Sutherland-Hodgman clipping is used so that cells crossing the triangular
    boundary are retained rather than dropped.
    """
    if not vertices:
        return []

    def inside(p):
        return p[0] + p[1] <= u_max + tol

    def intersection(p, q):
        sp = p[0] + p[1]
        sq = q[0] + q[1]
        den = sq - sp
        if abs(den) <= tol:
            return p
        t = (u_max - sp) / den
        t = min(1.0, max(0.0, t))
        return (
            p[0] + t * (q[0] - p[0]),
            p[1] + t * (q[1] - p[1]),
        )

    output = []
    prev = vertices[-1]
    prev_in = inside(prev)

    for curr in vertices:
        curr_in = inside(curr)
        if curr_in:
            if not prev_in:
                output.append(intersection(prev, curr))
            output.append(curr)
        elif prev_in:
            output.append(intersection(prev, curr))
        prev, prev_in = curr, curr_in

    # Remove adjacent numerical duplicates.
    clean = []
    for p in output:
        if not clean or np.linalg.norm(np.asarray(p) - np.asarray(clean[-1])) > 1e-12:
            clean.append(p)
    if len(clean) > 1 and np.linalg.norm(
        np.asarray(clean[0]) - np.asarray(clean[-1])
    ) <= 1e-12:
        clean.pop()

    return clean


def build_cells(u_max=U_MAX, n_u=N_U, cell_min=CELL_MIN):
    """
    Partition D_U = {(u_A,u_B)>=0 : u_A+u_B<=u_max} into clipped cells.

    Each base rectangle is clipped against the triangular boundary. For each
    nonempty two-dimensional cell we store:

      center : a representative point c_j inside the clipped polygon,
      lower  : coordinate-wise lower bounds on the whole cell,
      upper  : coordinate-wise upper bounds on the whole cell,
      radius : max coordinate deviations from c_j over the cell.

    The representative is only used to evaluate K_tau. It is NOT a support
    restriction on the continuum measure.
    """
    edges = u_edges(u_max, n_u, cell_min)

    centers = []
    lowers = []
    uppers = []
    radii = []
    polygons = []

    for ia in range(len(edges) - 1):
        x0, x1 = edges[ia], edges[ia + 1]
        for ib in range(len(edges) - 1):
            y0, y1 = edges[ib], edges[ib + 1]

            # The minimum x+y in the rectangle is at (x0,y0). If even that
            # exceeds U_MAX, the rectangle has no 2D intersection with D_U.
            if x0 + y0 >= u_max:
                continue

            rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            poly = _clip_polygon_sum_leq(rect, u_max)
            if len(poly) < 3:
                continue

            verts = np.asarray(poly, dtype=float)

            # Mean of the polygon vertices is a convex combination and is
            # therefore guaranteed to lie in the convex clipped cell.
            c = np.mean(verts, axis=0)
            lo = np.min(verts, axis=0)
            hi = np.max(verts, axis=0)
            rad = np.max(np.abs(verts - c), axis=0)

            centers.append(c)
            lowers.append(lo)
            uppers.append(hi)
            radii.append(rad)
            polygons.append(verts)

    if not centers:
        raise RuntimeError("Cell construction produced no cells.")

    return {
        "center": np.asarray(centers),
        "lower": np.asarray(lowers),
        "upper": np.asarray(uppers),
        "radius": np.asarray(radii),
        "polygons": polygons,
        "edges": edges,
    }


def exact_kernel_coeff(dphi, t2):
    """
    b(dphi) = Log(1 - t2 + t2 exp(i dphi)) / t2.

    log1p is used for numerical stability when t2 is extremely small.
    For 0 < t2 < 1/2 the argument stays in the open right half-plane, so the
    principal logarithm gives a continuous extension of the exact lattice
    kernel.
    """
    if not (0.0 < t2 < 0.5):
        raise ValueError("Certified continuum extension currently assumes 0 < t2 < 1/2.")
    return np.log1p(t2 * (np.exp(1j * dphi) - 1.0)) / t2


def kernel_and_cell_radii(phases, pairs, cells, t2):
    """
    Return:
      K[pair, cell]      = K_tau^{st}(c_j)
      delta[pair, cell]  >= sup_{u in C_j} |K_tau^{st}(u)-K_tau^{st}(c_j)|

    We use the global derivative bound
        |partial_{u_X} K_tau| <= |b_X|
    because |K_tau| <= 1. Hence
        delta <= |b_A| radius_A + |b_B| radius_B.
    """
    C = cells["center"]
    R = cells["radius"]
    NC = len(C)

    K = np.empty((len(pairs), NC), dtype=complex)
    delta = np.empty((len(pairs), NC), dtype=float)

    for i, (s1, s2) in enumerate(pairs):
        dA = phases[s2][0] - phases[s1][0]
        dB = phases[s2][1] - phases[s1][1]

        bA = exact_kernel_coeff(dA, t2)
        bB = exact_kernel_coeff(dB, t2)

        K[i, :] = np.exp(bA * C[:, 0] + bB * C[:, 1])
        delta[i, :] = (
            abs(bA) * R[:, 0]
            + abs(bB) * R[:, 1]
        )

    return K, delta


# --------------------------------------------------------------------------
# SDP

def solve_eph(M, n_bar, L, t2=T2, solver=None, solver_opts=None):
    phases = state_phases(M)
    NS = 4 * M
    P_pass, e_obs = channel(n_bar, L)

    P_plus = np.zeros(NS)
    P_minus = np.zeros(NS)
    for s in range(NS):
        hit, miss = (1 - e_obs) * P_pass, e_obs * P_pass
        P_plus[s], P_minus[s] = (
            (hit, miss) if (s % 4) in (0, 3) else (miss, hit)
        )
    P_fail = 1.0 - P_plus - P_minus
    prob_of = {Z_P: P_plus, Z_M: P_minus, Z_F: P_fail}

    # Zero-norm reduction: a zero Gram diagonal forces its whole row/column to
    # vanish under PSD, so remove such vectors to avoid strict-feasibility
    # problems in the numerical solver.
    active = [
        (z, s)
        for z in (Z_P, Z_M, Z_F)
        for s in range(NS)
        if prob_of[z][s] > 1e-12
    ]
    pos = {zs: k for k, zs in enumerate(active)}
    DIM = len(active)

    G = cp.Variable((DIM, DIM), hermitian=True)
    cons = [G >> 0]

    def ent(z1, s1, z2, s2):
        k1, k2 = pos.get((z1, s1)), pos.get((z2, s2))
        return 0 if (k1 is None or k2 is None) else G[k1, k2]

    # ------------------------------------------------------------------
    # Continuum measure -> certified cell masses
    #
    # q_j = nu(C_j), eps_tail = nu(D_U^c).
    # q_j does NOT mean a point mass at the representative c_j.

    cells = build_cells()
    center = cells["center"]
    lower = cells["lower"]
    NC = len(center)

    q = cp.Variable(NC, nonneg=True)
    eps_tail = cp.Variable(nonneg=True)

    cons.append(cp.sum(q) + eps_tail == 1.0)

    # The monitored energy bound is carried into the cell variables using
    # coordinate-wise LOWER bounds. This is the safe direction for an upper
    # bound on a mean: every true continuum measure automatically satisfies
    # these inequalities.
    cons.append(lower[:, 0] @ q <= n_bar / 2)
    cons.append(lower[:, 1] @ q <= n_bar / 2)

    # Outside D_U every point obeys u_A+u_B > U_MAX, so the tail can be
    # charged by U_MAX against the combined energy budget. It cannot be
    # charged separately to both arms.
    cons.append(
        (lower[:, 0] + lower[:, 1]) @ q
        + U_MAX * eps_tail
        <= n_bar
    )

    # ------------------------------------------------------------------
    # Exact finite-tau overlap kernel and cell-variation enclosure

    pairs = [
        (s1, s2)
        for s1 in range(NS)
        for s2 in range(s1 + 1, NS)
    ]
    K, delta = kernel_and_cell_radii(phases, pairs, cells, t2)

    terms = []
    for z in (Z_P, Z_M, Z_F):
        rows, cols, mask = [], [], []
        for (s1, s2) in pairs:
            k1, k2 = pos.get((z, s1)), pos.get((z, s2))
            if k1 is None or k2 is None:
                rows.append(0)
                cols.append(0)
                mask.append(0.0)
            else:
                rows.append(k1)
                cols.append(k2)
                mask.append(1.0)
        terms.append(cp.multiply(np.asarray(mask), G[rows, cols]))

    gram_overlap = sum(terms)
    cell_overlap = K @ q
    diff = gram_overlap - cell_overlap

    # For each pair:
    #   |diff_st|
    #     <= sum_j q_j delta_{st,j} + eps_tail + SLACK_FLOOR.
    #
    # The first term covers arbitrary placement of each q_j inside C_j.
    # eps_tail covers the omitted continuum tail because |K_tau| <= 1.
    slack = delta @ q + eps_tail + SLACK_FLOOR

    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(cp.norm(resid, 2, axis=1) <= slack)

    # Observed announcement probabilities are Gram diagonals.
    cons.append(
        cp.real(cp.diag(G))
        == np.asarray([prob_of[z][s] for (z, s) in active])
    )

    # ------------------------------------------------------------------
    # Objective: Eq. A7 on the key basis (m = 0)

    bracket = (
        ent(Z_P, 0, Z_P, 3)
        - ent(Z_M, 0, Z_M, 3)
        - ent(Z_P, 1, Z_P, 2)
        + ent(Z_M, 1, Z_M, 2)
    )
    obj = cp.real(bracket)

    # e_ph <= 1/2 is imposed explicitly, equivalently obj <= 0.
    cons.append(obj <= 0.0)

    prob = cp.Problem(cp.Maximize(obj), cons)

    if solver is None:
        try:
            import mosek  # noqa: F401
            solver, opts = cp.MOSEK, {}
        except ImportError:
            solver, opts = cp.CLARABEL, {}
    else:
        opts = {}

    if solver_opts:
        opts = {**opts, **solver_opts}
    if solver is not cp.MOSEK:
        opts.pop("mosek_params", None)

    prob.solve(solver=solver, **opts)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return (
            0.5, e_obs, P_pass, prob.status, None,
            {
                "n_cells": NC,
                "eps_tail": None,
                "cell_mass": None,
                "lower_mean_A": None,
                "lower_mean_B": None,
            },
        )

    e_ph = float(np.clip(
        0.5 + prob.value / (4 * P_pass),
        0.0,
        0.5,
    ))

    q_val = np.asarray(q.value).ravel()
    eps_val = float(eps_tail.value)
    diagnostics = {
        "n_cells": NC,
        "eps_tail": eps_val,
        "cell_mass": float(np.sum(q_val)),
        "lower_mean_A": float(lower[:, 0] @ q_val),
        "lower_mean_B": float(lower[:, 1] @ q_val),
    }

    return e_ph, e_obs, P_pass, prob.status, q_val, diagnostics


# --------------------------------------------------------------------------

if __name__ == "__main__":
    n_bar = 0.01
    cells = build_cells()

    print(
        f"M = {M}, n_bar = {n_bar}, t^2 = {T2:g}, "
        f"U_MAX = {U_MAX:g}, N_U = {N_U}, "
        f"{len(cells['center'])} clipped cells"
    )
    print(
        "Certified Stage-3 cell outer relaxation using the exact finite-tau "
        "kernel K_tau."
    )
    print(
        f"{'L(km)':>7} {'e_ph':>8} {'e_bit':>8} {'P_pass':>11} "
        f"{'R':>11} {'eps_tail':>11} {'EA_lower':>11}  status"
    )

    for L in (0.0, 2.0, 5.0, 10.0):
        e_ph, e_bit, P_pass, st, q_val, diag = solve_eph(
            M, n_bar, L
        )
        R = P_pass * max(
            0.0,
            1 - h2(e_ph) - F_EC * h2(e_bit),
        )

        if q_val is None:
            print(
                f"{L:7.1f} {'-':>8} {'-':>8} {P_pass:11.3e} "
                f"{'-':>11} {'-':>11} {'-':>11}  {st}"
            )
            continue

        print(
            f"{L:7.1f} {e_ph:8.4f} {e_bit:8.4f} {P_pass:11.3e} "
            f"{R:11.3e} {diag['eps_tail']:11.3e} "
            f"{diag['lower_mean_A']:11.3e}  {st}",
            flush=True,
        )

    print(
        "\nRefinement note: smaller cells reduce the local kernel radii, "
        "but independent-disk relaxations at two resolutions are not "
        "automatically nested. For a guaranteed monotone hierarchy, retain "
        "parent-cell constraints when adding child cells."
    )
