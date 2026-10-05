"""
Stage-4 plug-and-play phase-error SDP: per-round upper-energy monitoring.

Stage 4 keeps the Stage-3 Gram SDP and exact finite-tau kernel, but replaces
the mean-only source information by a stronger statement about monitor-
accepted rounds.

Let A_mon denote acceptance by a trusted per-round energy monitor.  We assume
a hardware-independent security interface

    Pr[ U_A > uA_cap or U_B > uB_cap | A_mon ] <= epsilon_mon.

No lower energy bound is assumed.  This is intentional: the Stage-3 failure
mechanism is the rare-bright-pulse tail, so an upper-energy veto is the
essential extra information.  Allowing arbitrarily weak or vacuum accepted
pulses is conservative.

Conditioned on A_mon, decompose the accepted-round measure into

    nu_acc = nu_good + nu_bad,

where
    supp(nu_good) subset [0,uA_cap] x [0,uB_cap],
    beta := mass(nu_bad) <= epsilon_mon.

The bad component is otherwise arbitrary.  Because |K_tau| <= 1, its total
contribution to every overlap has modulus at most beta.

The good component is represented by the lifted certified cells developed in
Stage 3.  For each cell C_j we retain

    q_j  = nu_good(C_j),
    xA_j = integral_{C_j} u_A dnu_good,
    xB_j = integral_{C_j} u_B dnu_good.

The exact finite-tau continuum kernel is

    K_tau^{st}(u_A,u_B)
      = exp[b_A^{st} u_A + b_B^{st} u_B],

    b_X^{st}
      = Log(1 - tau + tau exp(i Delta phi_X^{st})) / tau.

Linearizing K_tau about the cell representative c_j and integrating the
linear term exactly gives

    integral_{C_j} K_tau dnu
      = K_tau(c_j) [
            q_j
          + b_A (xA_j - cA_j q_j)
          + b_B (xB_j - cB_j q_j)
        ]
        + r_{st,j},

with the certified second-order remainder

    |r_{st,j}|
      <= q_j * eta_{st,j},

    eta_{st,j}
      <= min{ rho_{st,j}^2 exp(rho_{st,j})/2, 2 + rho_{st,j} },

    rho_{st,j}
      = sup_{u in C_j}
          | b_A (u_A-cA_j) + b_B (u_B-cB_j) |.

Therefore

    | sum_z G[(z,s),(z,t)] - Lambda_good_lifted |
       <= sum_j q_j eta_{st,j} + beta,

with
    sum_j q_j + beta = 1,
    0 <= beta <= epsilon_mon.

This is a certified outer relaxation of the generic monitor statement above.

Important rate convention
-------------------------
The synthetic P_pass produced below is interpreted as the central-node pass
probability conditioned on monitor acceptance.  The reported key rate is
therefore per monitor-accepted round:

    R_acc = P_pass|mon [1-h2(e_ph)-f_EC h2(e_bit)].

To obtain a rate per original incoming round, multiply by the physical monitor
acceptance probability P_mon.  P_mon is measurement-model dependent and is
not specified by the generic epsilon_mon abstraction.
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

# Generic monitor parameters.
#
# U_CAP_FACTOR is relative to the nominal honest per-arm delivered energy
# n_bar/2.  For example 1.2 means an upper threshold 20% above the nominal
# value.  There is deliberately NO lower threshold.
U_CAP_FACTOR = 1.2

# Placeholder monitor leakage probability.  This is an abstract security
# parameter until a concrete detector model/calibration derives it.
EPS_MON = 1e-6

# Number of uniform intervals per arm on [0,u_cap].  The Stage-4 domain is
# compact and small, so a uniform grid is more natural than the logarithmic
# tail grid used in Stage 3.
N_U = 60

# Tiny numerical allowance, independent of monitor leakage and cell remainders.
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
    """
    Synthetic observed statistics for monitor-accepted rounds.

    n_bar remains the nominal honest total returned mean used to generate
    P_pass and e_bit.  Security of the accepted ensemble is constrained by
    u_cap and epsilon_mon, not by a mean-energy inequality on nu_acc.
    """
    eta = ETA_DET * 10 ** (-XI * L / 10.0)
    p_dc = 0.0 if LOSS_ONLY else P_DC
    e_ali = 0.0 if LOSS_ONLY else E_ALI
    P_sig = 1.0 - np.exp(-eta * n_bar)
    P_pass = P_sig + p_dc
    return P_pass, (e_ali * P_sig + 0.5 * p_dc) / P_pass


def exact_kernel_coeff(dphi, t2):
    """
    b(dphi) = Log(1 - t2 + t2 exp(i dphi)) / t2.

    log1p is stable for very small t2.  For 0 < t2 < 1/2 the argument lies
    in the open right half-plane, so the principal logarithm is unambiguous.
    """
    if not (0.0 < t2 < 0.5):
        raise ValueError(
            "Certified continuum extension currently assumes 0 < t2 < 1/2."
        )
    return np.log1p(t2 * (np.exp(1j * dphi) - 1.0)) / t2


# --------------------------------------------------------------------------
# Compact upper-monitor cells

def build_monitor_cells(uA_cap, uB_cap, n_u=N_U):
    """
    Partition [0,uA_cap] x [0,uB_cap] into uniform rectangular cells.

    Returns arrays of representatives, coordinate bounds, radii, and polygon
    vertices.  Representatives are only linearization points; they are not
    support points of the measure.
    """
    if uA_cap <= 0 or uB_cap <= 0:
        raise ValueError("Upper monitor caps must be positive.")
    if n_u < 1:
        raise ValueError("N_U must be positive.")

    eA = np.linspace(0.0, uA_cap, n_u + 1)
    eB = np.linspace(0.0, uB_cap, n_u + 1)

    centers = []
    lowers = []
    uppers = []
    radii = []
    polygons = []

    for ia in range(n_u):
        x0, x1 = eA[ia], eA[ia + 1]
        for ib in range(n_u):
            y0, y1 = eB[ib], eB[ib + 1]

            verts = np.asarray(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                dtype=float,
            )
            c = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2])

            centers.append(c)
            lowers.append([x0, y0])
            uppers.append([x1, y1])
            radii.append([(x1 - x0) / 2, (y1 - y0) / 2])
            polygons.append(verts)

    return {
        "center": np.asarray(centers),
        "lower": np.asarray(lowers),
        "upper": np.asarray(uppers),
        "radius": np.asarray(radii),
        "polygons": polygons,
    }


def kernel_linearization_data(phases, pairs, cells, t2):
    """
    Exact finite-tau kernel values plus certified second-order cell remainder.

    Returns
      K[pair,cell]   = K_tau^{st}(c_j)
      KA[pair,cell]  = K_tau^{st}(c_j) b_A^{st}
      KB[pair,cell]  = K_tau^{st}(c_j) b_B^{st}
      rem[pair,cell] certified coefficient eta_{st,j}.
    """
    C = cells["center"]
    polygons = cells["polygons"]
    NC = len(C)

    K = np.empty((len(pairs), NC), dtype=complex)
    KA = np.empty((len(pairs), NC), dtype=complex)
    KB = np.empty((len(pairs), NC), dtype=complex)
    rem = np.empty((len(pairs), NC), dtype=float)

    for i, (s1, s2) in enumerate(pairs):
        dA = phases[s2][0] - phases[s1][0]
        dB = phases[s2][1] - phases[s1][1]

        bA = exact_kernel_coeff(dA, t2)
        bB = exact_kernel_coeff(dB, t2)

        K_i = np.exp(bA * C[:, 0] + bB * C[:, 1])
        K[i, :] = K_i
        KA[i, :] = K_i * bA
        KB[i, :] = K_i * bB

        for j, verts in enumerate(polygons):
            du = verts - C[j]
            z = bA * du[:, 0] + bB * du[:, 1]
            rho = float(np.max(np.abs(z)))

            taylor_bound = 0.5 * rho * rho * math.exp(min(rho, 50.0))
            crude_bound = 2.0 + rho
            rem[i, j] = min(taylor_bound, crude_bound)

    return K, KA, KB, rem


# --------------------------------------------------------------------------
# SDP

def solve_eph(
    M,
    n_bar,
    L,
    t2=T2,
    uA_cap=None,
    uB_cap=None,
    eps_mon=EPS_MON,
    n_u=N_U,
    solver=None,
    solver_opts=None,
):
    """
    Certified Stage-4 phase-error upper bound.

    uA_cap/uB_cap are upper bounds on the rescaled energies U_A,U_B for the
    good monitor-accepted component.  If omitted, both default to
        U_CAP_FACTOR * n_bar / 2.

    eps_mon bounds the conditional bad mass:
        Pr[U_A>uA_cap or U_B>uB_cap | monitor accept] <= eps_mon.
    """
    if uA_cap is None:
        uA_cap = U_CAP_FACTOR * n_bar / 2
    if uB_cap is None:
        uB_cap = U_CAP_FACTOR * n_bar / 2
    if not (0.0 <= eps_mon <= 1.0):
        raise ValueError("eps_mon must lie in [0,1].")

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
    # Accepted measure = good compact component + arbitrary monitor leakage

    cells = build_monitor_cells(uA_cap, uB_cap, n_u=n_u)
    center = cells["center"]
    lower = cells["lower"]
    upper = cells["upper"]
    NC = len(center)

    q = cp.Variable(NC, nonneg=True)
    xA = cp.Variable(NC, nonneg=True)
    xB = cp.Variable(NC, nonneg=True)
    beta = cp.Variable(nonneg=True)

    cons += [
        cp.sum(q) + beta == 1.0,
        beta <= eps_mon,
    ]

    # Perspective constraints for the good component.
    cons += [
        xA >= cp.multiply(lower[:, 0], q),
        xA <= cp.multiply(upper[:, 0], q),
        xB >= cp.multiply(lower[:, 1], q),
        xB <= cp.multiply(upper[:, 1], q),
    ]

    # No Stage-3 mean-energy constraint is imposed on the monitor-conditioned
    # ensemble.  The upper-support guarantee is the Stage-4 source
    # information.  If an experiment separately certifies a conditional mean,
    # it can be added here as an additional valid constraint.

    # ------------------------------------------------------------------
    # Exact finite-tau lifted overlap of the good component

    pairs = [
        (s1, s2)
        for s1 in range(NS)
        for s2 in range(s1 + 1, NS)
    ]
    K, KA, KB, rem = kernel_linearization_data(
        phases, pairs, cells, t2
    )

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

    dA = xA - cp.multiply(center[:, 0], q)
    dB = xB - cp.multiply(center[:, 1], q)
    good_overlap = K @ q + KA @ dA + KB @ dB

    diff = gram_overlap - good_overlap

    # beta bounds the completely arbitrary bad monitor-accepted component.
    slack = rem @ q + beta + SLACK_FLOOR

    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(cp.norm(resid, 2, axis=1) <= slack)

    # Observed announcement probabilities for the monitor-accepted ensemble.
    cons.append(
        cp.real(cp.diag(G))
        == np.asarray([prob_of[z][s] for (z, s) in active])
    )

    # ------------------------------------------------------------------
    # A7 objective on the key basis

    bracket = (
        ent(Z_P, 0, Z_P, 3)
        - ent(Z_M, 0, Z_M, 3)
        - ent(Z_P, 1, Z_P, 2)
        + ent(Z_M, 1, Z_M, 2)
    )
    obj = cp.real(bracket)

    # Equivalent to e_ph <= 1/2.
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
    if solver != cp.MOSEK:
        opts.pop("mosek_params", None)

    prob.solve(solver=solver, **opts)

    base_diag = {
        "n_cells": NC,
        "uA_cap": uA_cap,
        "uB_cap": uB_cap,
        "eps_mon": eps_mon,
        "beta": None,
        "good_mass": None,
        "mean_A_good": None,
        "mean_B_good": None,
    }

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return 0.5, e_obs, P_pass, prob.status, None, base_diag

    e_ph = float(np.clip(
        0.5 + prob.value / (4 * P_pass),
        0.0,
        0.5,
    ))

    q_val = np.asarray(q.value).ravel()
    xA_val = np.asarray(xA.value).ravel()
    xB_val = np.asarray(xB.value).ravel()

    diagnostics = {
        **base_diag,
        "beta": float(beta.value),
        "good_mass": float(np.sum(q_val)),
        "mean_A_good": float(np.sum(xA_val)),
        "mean_B_good": float(np.sum(xB_val)),
    }

    return e_ph, e_obs, P_pass, prob.status, q_val, diagnostics


# --------------------------------------------------------------------------

if __name__ == "__main__":
    n_bar = 0.01
    u_cap = U_CAP_FACTOR * n_bar / 2

    print(
        f"M = {M}, n_bar = {n_bar}, t^2 = {T2:g}, "
        f"u_cap = {u_cap:g} per arm, eps_mon = {EPS_MON:g}, "
        f"N_U = {N_U}, {N_U**2} cells"
    )
    print(
        "Certified Stage-4 upper-energy monitor relaxation; no lower-energy "
        "threshold is assumed."
    )
    print(
        f"{'L(km)':>7} {'e_ph':>8} {'e_bit':>8} {'Ppass|mon':>11} "
        f"{'R_acc':>11} {'beta':>11} {'EA_good':>11}  status"
    )

    for L in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        e_ph, e_bit, P_pass, st, q_val, diag = solve_eph(
            M,
            n_bar,
            L,
            uA_cap=u_cap,
            uB_cap=u_cap,
            eps_mon=EPS_MON,
        )

        # Rate per monitor-accepted round.  Multiply by P_mon for the rate per
        # original incoming round once a physical monitor model supplies it.
        R_acc = P_pass * max(
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
            f"{R_acc:11.3e} {diag['beta']:11.3e} "
            f"{diag['mean_A_good']:11.3e}  {st}",
            flush=True,
        )

    print(
        "\nR_acc is conditional on monitor acceptance.  A physical detector "
        "model must provide P_mon and justify eps_mon before converting this "
        "to a per-original-round key rate."
    )
