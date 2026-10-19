"""
Stage-5B plug-and-play phase-error SDP:
joint PNA beam splitter + additional trusted attenuation.

Architecture
------------
For each arm,

    Eve input
        -> trusted PNA beam splitter (signal transmissivity tau_s)
        -> destructive phase-insensitive monitor
        -> accepted surviving signal with photon number N
        -> trusted pure-loss attenuator of transmissivity lambda_att
        -> secret phase encoding.

Stage 5B is intended for the regime where the PNA beam splitter is moderate
(e.g. 50:50), so the accepted pre-attenuation signal photon number N can still
be very large.  Keeping one SDP variable for each N is then impractical.

The PNA first certifies an accepted upper tail

    Pr(
        N_A > N_A_plus or N_B > N_B_plus
        | both monitors accept
    ) <= epsilon_5.

The extra trusted attenuator then motivates the rescaling

    u_X = lambda_X N_X.

For fixed N=n, the post-attenuation phase kernel is exact:

    E[exp(i theta K) | N=n]
      = (1-lambda + lambda exp(i theta))^n
      = exp[b_lambda(theta) u],

where

    b_lambda(theta)
      = Log(1-lambda + lambda exp(i theta)) / lambda.

Thus the physical accepted distribution is supported on the fine lattice

    u_X in lambda_X Z_{>=0}.

On the good component, the PNA tail gives the compact rectangle

    0 <= u_A <= lambda_A N_A_plus,
    0 <= u_B <= lambda_B N_B_plus.

Dropping the lattice support gives a continuum OUTER relaxation.  As in
Stages 3 and 4, this continuum is discretized by certified lifted cells.  For
each cell C_j we keep

    q_j  = nu_good(C_j),
    xA_j = integral_Cj u_A dnu_good,
    xB_j = integral_Cj u_B dnu_good,

so constant and first-order kernel variation are represented exactly.  The
remaining second-order Taylor error is bounded rigorously.

The arbitrary bad accepted component has mass

    beta <= epsilon_5

and contributes at most beta in modulus to every overlap.

Important distinctions
----------------------
* Stage 5A needs no continuum relaxation because it works directly with a
  small surviving Fock cutoff and PNA-derived occupancy constraints.
* Stage 5B uses the continuum only as a numerical relaxation for the very
  large photon number entering the extra attenuator.
* Additional attenuation does not create the common phase orbit; monitoring
  and attenuation both occur before the secret phase encoding.
* lambda_att = 1 is mathematically the no-extra-attenuation limit, but the
  continuum representation is useful specifically when lambda_att is small
  enough that the physical lattice in u is very fine.

This script currently uses the ideal hard-threshold PNA response

    a_c = 1[c <= c_max]

to derive epsilon_5.  The continuum/Gram part itself only needs the certified
tail interface and therefore can later accept a realistic detector-derived
epsilon_5 without changing the SDP.

No solve is performed on import.  The guarded main() runs a parallel benchmark
scan only when the file is explicitly executed.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pnp_benchmark as benchmark
from pnp_results import result_row, report, stage_results_dir

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------
# Protocol / synthetic accepted-ensemble parameters

M_DEFAULT = 4
F_EC = 1.15
ETA_DET = 0.85
XI = 0.2
E_ALI = 0.015
P_DC = 5e-8
LOSS_ONLY = False

Z_P, Z_M, Z_F = 0, 1, 2
Z_ALL = (Z_P, Z_M, Z_F)

ENC_PHASES = (
    (0.0, 0.0),
    (0.0, np.pi),
    (np.pi, 0.0),
    (np.pi, np.pi),
)

# Continuum-cell resolution.
N_U = 50

# Numerical SOC allowance, independent of the physical tail beta and the
# certified within-cell remainder.
SLACK_FLOOR = 1e-12

# Default benchmark architecture:
#   50:50 PNA beam splitter;
#   ideal upper threshold c_max on the monitor arm;
#   strong additional attenuation chosen so a ~10^6-photon pre-attenuation
#   signal maps to u ~ 5e-3 per arm.
TAU_S = 0.5
C_MAX = 1_000_000
LAMBDA_ATT = 5e-9
P_MON_BOTH = 0.99

# The good cutoff is scanned as N_plus = C_MAX + offset.  For a 50:50 PNA
# splitter, useful source-independent tails require N_plus to be of roughly the
# same order as c_max; offsets of several thousand probe the transition from
# weak to extremely small false-accept tails.
N_PLUS_OFFSETS = (4_000, 6_000, 8_000, 10_000)

DISTANCES_KM = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
MAX_WORKERS = min(4, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Data containers


@dataclass(frozen=True)
class TailInterface:
    tau_s: float
    c_max: int
    n_plus: int
    m_star: int
    delta_one_arm: float
    epsilon_5: float
    p_mon_both: float
    u_cap: float


@dataclass
class Stage5BResult:
    e_ph_upper: float
    e_bit: float
    p_pass_mon: float
    r_acc: float
    r_original: float | None
    status: str
    diagnostics: dict


# ---------------------------------------------------------------------------
# Elementary helpers


def _check_probability(name: str, value: float, *, positive: bool = False) -> None:
    if positive:
        ok = 0.0 < value <= 1.0
        interval = "(0,1]"
    else:
        ok = 0.0 <= value <= 1.0
        interval = "[0,1]"
    if not ok:
        raise ValueError(f"{name} must lie in {interval}.")


def _check_tau(name: str, value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must lie strictly between 0 and 1.")


def _check_nonnegative_int(name: str, value: int) -> None:
    if int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")


def h2(e: float) -> float:
    if e <= 0.0 or e >= 1.0:
        return 0.0
    return float(-e * math.log2(e) - (1.0 - e) * math.log2(1.0 - e))


def state_phases(M: int) -> list[tuple[float, float]]:
    if M < 1:
        raise ValueError("M must be positive.")
    out = []
    for x in range(M):
        phi = np.pi * x / M
        for dA, dB in ENC_PHASES:
            out.append((phi + dA, phi + dB))
    return out


# ---------------------------------------------------------------------------
# Synthetic accepted-ensemble central statistics


def synthetic_announcement_probabilities(
    M: int, n_bar: float, distance_km: float
) -> tuple[dict[int, np.ndarray], float, float]:
    """
    Shared two-arm coherent-reference synthetic stats (NOT experimental data).
    L is total Alice--Bob distance; n_bar is post-local-optics nominal mean.
    For basis-dependent extra attenuation use the PM script's version.
    """
    return benchmark.coherent_reference_probabilities(
        M, n_bar, distance_km,
        eta_det=ETA_DET, fiber_loss_db_per_km=XI,
        p_dc=P_DC, e_ali=E_ALI, loss_only=LOSS_ONLY,
    )


def synthetic_channel(n_bar: float, distance_km: float) -> tuple[float,float]:
    """Key-basis statistics from the shared coherent-reference benchmark."""
    _, p_pass, e_bit = synthetic_announcement_probabilities(
        M_DEFAULT, n_bar, distance_km
    )
    return p_pass, e_bit


# ---------------------------------------------------------------------------
# Ideal PNA tail -> Stage-5B compact u-domain


def ideal_delta_exact(n_plus: int, tau_s: float, c_max: int) -> tuple[float, int]:
    """
    Exact Stage-5A ideal hard-threshold tail:

        delta(N_plus)
          = sup_m Pr(A_mon, N>N_plus | M=m)

    with maximizer

        m_star = c_max + N_plus + 1.
    """
    _check_nonnegative_int("n_plus", n_plus)
    _check_nonnegative_int("c_max", c_max)
    _check_tau("tau_s", tau_s)

    m_star = c_max + n_plus + 1
    delta = float(binom.sf(n_plus, m_star, tau_s))
    return delta, m_star


def ideal_tail_interface(
    *,
    tau_s: float,
    c_max: int,
    n_plus: int,
    lambda_att: float,
    p_mon_both: float,
) -> TailInterface:
    """
    Two-arm symmetric ideal-PNA interface for Stage 5B.

    The union bound gives

        epsilon_5 <= 2 delta(N_plus) / P_mon,both.

    The compact continuum cap after the additional attenuator is

        u_cap = lambda_att * N_plus.
    """
    _check_tau("lambda_att", lambda_att)
    _check_probability("p_mon_both", p_mon_both, positive=True)

    delta, m_star = ideal_delta_exact(n_plus, tau_s, c_max)
    eps5 = min(1.0, 2.0 * delta / p_mon_both)

    return TailInterface(
        tau_s=float(tau_s),
        c_max=int(c_max),
        n_plus=int(n_plus),
        m_star=int(m_star),
        delta_one_arm=float(delta),
        epsilon_5=float(eps5),
        p_mon_both=float(p_mon_both),
        u_cap=float(lambda_att * n_plus),
    )


# ---------------------------------------------------------------------------
# Exact post-attenuation continuum kernel


def exact_kernel_coeff(dphi: float, lambda_att: float) -> complex:
    """
    b_lambda(dphi)
      = Log(1-lambda + lambda exp(i dphi)) / lambda.

    On the physical lattice u=lambda*n,

        exp[b_lambda(dphi) u]
          = (1-lambda + lambda exp(i dphi))^n.
    """
    if not (0.0 < lambda_att < 0.5):
        raise ValueError(
            "The present continuum implementation assumes 0 < lambda_att < 1/2."
        )
    return (
        np.log1p(lambda_att * (np.exp(1j * dphi) - 1.0))
        / lambda_att
    )


def build_monitor_cells(uA_cap: float, uB_cap: float, n_u: int = N_U):
    """
    Uniform rectangular cells on the PNA-certified compact continuum domain

        [0,uA_cap] x [0,uB_cap].
    """
    if uA_cap <= 0.0 or uB_cap <= 0.0:
        raise ValueError("u caps must be positive.")
    if n_u < 1:
        raise ValueError("n_u must be positive.")

    eA = np.linspace(0.0, uA_cap, n_u + 1)
    eB = np.linspace(0.0, uB_cap, n_u + 1)

    centers = []
    lowers = []
    uppers = []
    polygons = []

    for ia in range(n_u):
        x0, x1 = eA[ia], eA[ia + 1]
        for ib in range(n_u):
            y0, y1 = eB[ib], eB[ib + 1]

            verts = np.asarray(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                dtype=float,
            )
            centers.append([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
            lowers.append([x0, y0])
            uppers.append([x1, y1])
            polygons.append(verts)

    return {
        "center": np.asarray(centers),
        "lower": np.asarray(lowers),
        "upper": np.asarray(uppers),
        "polygons": polygons,
    }


def kernel_linearization_data(
    phases: Sequence[tuple[float, float]],
    pairs: Sequence[tuple[int, int]],
    cells,
    lambda_a: float,
    lambda_b: float,
):
    """
    Exact finite-lambda kernel at each cell representative plus a certified
    second-order within-cell remainder.

    For

        K(u_A,u_B) = exp(b_A u_A + b_B u_B),

    retain the integrated affine term exactly using cell masses and first
    moments.  If

        rho_j >= sup_{u in C_j}
                    |b_A(u_A-c_Aj)+b_B(u_B-c_Bj)|,

    then

        |R_j| <= q_j min{rho_j^2 exp(rho_j)/2, 2+rho_j}.
    """
    C = cells["center"]
    polygons = cells["polygons"]
    nc = len(C)

    K = np.empty((len(pairs), nc), dtype=complex)
    KA = np.empty((len(pairs), nc), dtype=complex)
    KB = np.empty((len(pairs), nc), dtype=complex)
    rem = np.empty((len(pairs), nc), dtype=float)

    for i, (s, t) in enumerate(pairs):
        dA = phases[t][0] - phases[s][0]
        dB = phases[t][1] - phases[s][1]

        bA = exact_kernel_coeff(dA, lambda_a)
        bB = exact_kernel_coeff(dB, lambda_b)

        K_i = np.exp(bA * C[:, 0] + bB * C[:, 1])
        K[i, :] = K_i
        KA[i, :] = K_i * bA
        KB[i, :] = K_i * bB

        for j, verts in enumerate(polygons):
            du = verts - C[j]
            z = bA * du[:, 0] + bB * du[:, 1]
            rho = float(np.max(np.abs(z)))

            taylor = 0.5 * rho * rho * math.exp(min(rho, 50.0))
            crude = 2.0 + rho
            rem[i, j] = min(taylor, crude)

    return K, KA, KB, rem


# ---------------------------------------------------------------------------
# Complete Stage-5B lifted continuum Gram SDP


def solve_stage5b_sdp(
    *,
    M: int,
    uA_cap: float,
    uB_cap: float,
    epsilon_5: float,
    lambda_a: float,
    lambda_b: float,
    prob_of: dict[int, Sequence[float]],
    p_mon: float | None = None,
    n_u: int = N_U,
    f_ec: float = F_EC,
    solver=None,
    solver_opts=None,
) -> Stage5BResult:
    """
    Solve the Stage-5B compact-continuum outer relaxation.

    The physical accepted pre-attenuation Fock distribution is pushed forward
    by u_X=lambda_X N_X.  We then drop the fine lattice support and optimize
    over an arbitrary continuum measure on the PNA-certified good rectangle,
    plus bad mass beta <= epsilon_5.

    No conditional mean-energy constraint is imposed.
    """
    if M < 1:
        raise ValueError("M must be positive.")
    if uA_cap <= 0.0 or uB_cap <= 0.0:
        raise ValueError("u caps must be positive.")
    _check_probability("epsilon_5", epsilon_5)
    if p_mon is not None:
        _check_probability("p_mon", p_mon, positive=True)

    ns = 4 * M
    phases = state_phases(M)

    probs = {}
    for z in Z_ALL:
        if z not in prob_of:
            raise ValueError(f"prob_of is missing announcement key {z}.")
        arr = np.asarray(prob_of[z], dtype=float)
        if arr.shape != (ns,):
            raise ValueError(
                f"prob_of[{z}] must have shape ({ns},), got {arr.shape}."
            )
        if np.any(arr < -1e-12) or np.any(arr > 1.0 + 1e-12):
            raise ValueError("Observed probabilities must lie in [0,1].")
        probs[z] = np.clip(arr, 0.0, 1.0)

    total = sum(probs[z] for z in Z_ALL)
    if not np.allclose(total, 1.0, atol=1e-9):
        raise ValueError(
            "For each signal label, accepted-ensemble probabilities must sum to 1."
        )

    p_pass_mon = 0.25 * sum(
        probs[Z_P][ab] + probs[Z_M][ab]
        for ab in range(4)
    )
    if p_pass_mon <= 0.0:
        raise ValueError("Observed P_pass|mon must be positive.")

    error_prob = 0.25 * (
        probs[Z_M][0]
        + probs[Z_P][1]
        + probs[Z_P][2]
        + probs[Z_M][3]
    )
    e_bit = float(error_prob / p_pass_mon)

    active = [
        (z, s)
        for z in Z_ALL
        for s in range(ns)
        if probs[z][s] > 1e-12
    ]
    pos = {zs: k for k, zs in enumerate(active)}
    dim = len(active)

    G = cp.Variable((dim, dim), hermitian=True)
    cons = [G >> 0]

    def ent(z1: int, s1: int, z2: int, s2: int):
        k1 = pos.get((z1, s1))
        k2 = pos.get((z2, s2))
        if k1 is None or k2 is None:
            return 0
        return G[k1, k2]

    cons.append(
        cp.real(cp.diag(G))
        == np.asarray([probs[z][s] for (z, s) in active])
    )

    # ------------------------------------------------------------------
    # Compact continuum good component + arbitrary bad mass

    cells = build_monitor_cells(uA_cap, uB_cap, n_u=n_u)
    center = cells["center"]
    lower = cells["lower"]
    upper = cells["upper"]
    nc = len(center)

    q = cp.Variable(nc, nonneg=True)
    xA = cp.Variable(nc, nonneg=True)
    xB = cp.Variable(nc, nonneg=True)
    beta = cp.Variable(nonneg=True)

    cons += [
        cp.sum(q) + beta == 1.0,
        beta <= epsilon_5,
        xA >= cp.multiply(lower[:, 0], q),
        xA <= cp.multiply(upper[:, 0], q),
        xB >= cp.multiply(lower[:, 1], q),
        xB <= cp.multiply(upper[:, 1], q),
    ]

    # ------------------------------------------------------------------
    # Lifted exact finite-lambda kernel

    pairs = [
        (s, t)
        for s in range(ns)
        for t in range(s + 1, ns)
    ]

    K, KA, KB, rem = kernel_linearization_data(
        phases,
        pairs,
        cells,
        lambda_a,
        lambda_b,
    )

    dA = xA - cp.multiply(center[:, 0], q)
    dB = xB - cp.multiply(center[:, 1], q)
    good_overlap = K @ q + KA @ dA + KB @ dB

    gram_terms = []
    for s, t in pairs:
        gram_terms.append(sum(ent(z, s, z, t) for z in Z_ALL))
    gram_overlap = cp.hstack(gram_terms)

    diff = gram_overlap - good_overlap
    slack = rem @ q + beta + SLACK_FLOOR

    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(cp.norm(resid, 2, axis=1) <= slack)

    # ------------------------------------------------------------------
    # A7 objective on key-basis labels 00,01,10,11 = 0,1,2,3.

    bracket = (
        ent(Z_P, 0, Z_P, 3)
        - ent(Z_M, 0, Z_M, 3)
        - ent(Z_P, 1, Z_P, 2)
        + ent(Z_M, 1, Z_M, 2)
    )
    objective_scalar = cp.real(bracket)

    prob = cp.Problem(cp.Maximize(objective_scalar), cons)

    if solver is None:
        try:
            import mosek  # noqa: F401
            solver = cp.MOSEK
            opts = {}
        except ImportError:
            solver = cp.CLARABEL
            opts = {}
    else:
        opts = {}

    if solver_opts:
        opts = {**opts, **solver_opts}
    if solver != cp.MOSEK:
        opts.pop("mosek_params", None)

    prob.solve(solver=solver, **opts)

    base_diag = {
        "epsilon_5": float(epsilon_5),
        "beta": None,
        "good_mass": None,
        "uA_cap": float(uA_cap),
        "uB_cap": float(uB_cap),
        "lambda_a": float(lambda_a),
        "lambda_b": float(lambda_b),
        "n_cells": int(nc),
        "gram_dim": int(dim),
        "p_mon": None if p_mon is None else float(p_mon),
        "mean_uA_good": None,
        "mean_uB_good": None,
        "max_cell_remainder": float(np.max(rem)),
    }

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return Stage5BResult(
            e_ph_upper=1.0,
            e_bit=e_bit,
            p_pass_mon=float(p_pass_mon),
            r_acc=0.0,
            r_original=None if p_mon is None else 0.0,
            status=prob.status,
            diagnostics=base_diag,
        )

    e_ph_upper = float(
        np.clip(
            0.5 + float(prob.value) / (4.0 * p_pass_mon),
            0.0,
            1.0,
        )
    )

    if e_ph_upper >= 0.5:
        privacy_term = 0.0
    else:
        privacy_term = max(
            0.0,
            1.0 - h2(e_ph_upper) - f_ec * h2(e_bit),
        )

    r_acc = float(p_pass_mon * privacy_term)
    r_original = None if p_mon is None else float(p_mon * r_acc)

    q_val = np.asarray(q.value).ravel()
    xA_val = np.asarray(xA.value).ravel()
    xB_val = np.asarray(xB.value).ravel()

    diagnostics = {
        **base_diag,
        "beta": float(beta.value),
        "good_mass": float(np.sum(q_val)),
        "mean_uA_good": float(np.sum(xA_val)),
        "mean_uB_good": float(np.sum(xB_val)),
        "objective_scalar": float(prob.value),
    }

    return Stage5BResult(
        e_ph_upper=e_ph_upper,
        e_bit=e_bit,
        p_pass_mon=float(p_pass_mon),
        r_acc=r_acc,
        r_original=r_original,
        status=prob.status,
        diagnostics=diagnostics,
    )


def solve_stage5b_ideal_benchmark(
    *,
    M: int,
    n_bar: float,
    distance_km: float,
    tau_s: float,
    c_max: int,
    n_plus: int,
    lambda_att: float,
    p_mon_both: float,
    n_u: int = N_U,
    solver=None,
    solver_opts=None,
) -> tuple[Stage5BResult, TailInterface]:
    """
    Symmetric ideal-PNA Stage-5B benchmark.

    The ideal PNA supplies epsilon_5 and N_plus; the additional attenuator maps
    the good pre-attenuation Fock support into the compact continuum cap

        u_cap = lambda_att * N_plus.
    """
    interface = ideal_tail_interface(
        tau_s=tau_s,
        c_max=c_max,
        n_plus=n_plus,
        lambda_att=lambda_att,
        p_mon_both=p_mon_both,
    )

    prob_of, _, _ = synthetic_announcement_probabilities(
        M,
        n_bar,
        distance_km,
    )

    result = solve_stage5b_sdp(
        M=M,
        uA_cap=interface.u_cap,
        uB_cap=interface.u_cap,
        epsilon_5=interface.epsilon_5,
        lambda_a=lambda_att,
        lambda_b=lambda_att,
        prob_of=prob_of,
        p_mon=p_mon_both,
        n_u=n_u,
        solver=solver,
        solver_opts=solver_opts,
    )

    return result, interface


# ---------------------------------------------------------------------------
# Parallel benchmark scan


def _scan_one(task):
    (
        M,
        n_bar,
        distance_km,
        tau_s,
        c_max,
        n_plus,
        lambda_att,
        p_mon_both,
        n_u,
        solver,
        solver_opts,
    ) = task

    result, interface = solve_stage5b_ideal_benchmark(
        M=M,
        n_bar=n_bar,
        distance_km=distance_km,
        tau_s=tau_s,
        c_max=c_max,
        n_plus=n_plus,
        lambda_att=lambda_att,
        p_mon_both=p_mon_both,
        n_u=n_u,
        solver=solver,
        solver_opts=solver_opts,
    )

    return {
        "L_km": float(distance_km),
        "N_plus": int(n_plus),
        "offset": int(n_plus - c_max),
        "u_cap": float(interface.u_cap),
        "epsilon_5": float(interface.epsilon_5),
        "delta_one_arm": float(interface.delta_one_arm),
        "e_ph": float(result.e_ph_upper),
        "e_bit": float(result.e_bit),
        "Ppass_mon": float(result.p_pass_mon),
        "R_acc": float(result.r_acc),
        "R_original": (
            None if result.r_original is None else float(result.r_original)
        ),
        "beta": result.diagnostics.get("beta"),
        "mean_uA_good": result.diagnostics.get("mean_uA_good"),
        "status": result.status,
    }


def run_scan(
    *,
    M: int,
    n_bar: float,
    tau_s: float = TAU_S,
    c_max: int = C_MAX,
    lambda_att: float = LAMBDA_ATT,
    p_mon_both: float = P_MON_BOTH,
    offsets=N_PLUS_OFFSETS,
    distances=DISTANCES_KM,
    n_u: int = N_U,
    max_workers: int = MAX_WORKERS,
    solver=None,
    solver_opts=None,
):
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    tasks = []
    for offset in offsets:
        _check_nonnegative_int("offset", offset)
        n_plus = c_max + offset
        for distance_km in distances:
            tasks.append(
                (
                    M,
                    n_bar,
                    float(distance_km),
                    tau_s,
                    c_max,
                    n_plus,
                    lambda_att,
                    p_mon_both,
                    n_u,
                    solver,
                    solver_opts,
                )
            )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scan_one, task) for task in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (r["offset"], r["L_km"]))
    return results


def print_scan(results) -> None:
    print(
        f"{'dN':>7} {'u_cap':>10} {'eps5':>11} {'L(km)':>7} "
        f"{'e_ph':>8} {'Ppass|mon':>11} {'R_acc':>11} "
        f"{'R_orig':>11} {'beta':>11}  status"
    )

    for r in results:
        r_orig = (
            "-"
            if r["R_original"] is None
            else f"{r['R_original']:.3e}"
        )
        beta = "-" if r["beta"] is None else f"{r['beta']:.3e}"

        print(
            f"{r['offset']:7d} {r['u_cap']:10.4e} "
            f"{r['epsilon_5']:11.3e} {r['L_km']:7.1f} "
            f"{r['e_ph']:8.4f} {r['Ppass_mon']:11.3e} "
            f"{r['R_acc']:11.3e} {r_orig:>11} {beta:>11}  "
            f"{r['status']}",
            flush=True,
        )


def plot_scan(results, output_dir=None):
    if not results:
        raise ValueError("Cannot plot an empty scan.")

    if output_dir is None:
        output_dir = stage_results_dir("stage5b")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    offsets = sorted({r["offset"] for r in results})
    distances = sorted({r["L_km"] for r in results})

    # Phase error versus distance.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for offset in offsets:
        rows = sorted(
            (r for r in results if r["offset"] == offset),
            key=lambda r: r["L_km"],
        )
        ax.plot(
            [r["L_km"] for r in rows],
            [r["e_ph"] for r in rows],
            marker="o",
            label=fr"$N_+-c_{{\max}}={offset}$",
        )

    ax.axhline(
        0.292,
        linestyle="--",
        linewidth=1.0,
        label=r"rough positive-key threshold $e_{\rm ph}\simeq0.292$",
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel(r"$e_{\rm ph}$")
    ax.set_title("Stage 5B: phase-error bound")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    eph_path = output_dir / "stage5b-eph-vs-distance.png"
    fig.savefig(eph_path, dpi=220)
    plt.close(fig)

    # Accepted-round key rate.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for offset in offsets:
        rows = sorted(
            (r for r in results if r["offset"] == offset),
            key=lambda r: r["L_km"],
        )
        ax.plot(
            [r["L_km"] for r in rows],
            [r["R_acc"] for r in rows],
            marker="o",
            label=fr"$N_+-c_{{\max}}={offset}$",
        )

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel(r"$R_{\rm acc}$")
    ax.set_title("Stage 5B: key rate per monitor-accepted round")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    rate_path = output_dir / "stage5b-rate-vs-distance.png"
    fig.savefig(rate_path, dpi=220)
    plt.close(fig)

    # PNA leakage and effective continuum cap versus guard offset.
    rows = [
        next(r for r in results if r["offset"] == offset)
        for offset in offsets
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(
        [r["offset"] for r in rows],
        [max(r["epsilon_5"], np.finfo(float).tiny) for r in rows],
        marker="o",
    )
    ax.set_xlabel(r"Guard offset $N_+-c_{\max}$")
    ax.set_ylabel(r"$\epsilon_5$")
    ax.set_title("Stage 5B: ideal-PNA bad-tail bound")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    eps_path = output_dir / "stage5b-epsilon5-vs-offset.png"
    fig.savefig(eps_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        [r["offset"] for r in rows],
        [r["u_cap"] for r in rows],
        marker="o",
    )
    ax.set_xlabel(r"Guard offset $N_+-c_{\max}$")
    ax.set_ylabel(r"$u_+=\lambda N_+$")
    ax.set_title("Stage 5B: compact continuum cap")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    cap_path = output_dir / "stage5b-ucap-vs-offset.png"
    fig.savefig(cap_path, dpi=220)
    plt.close(fig)

    return eph_path, rate_path, eps_path, cap_path


# ---------------------------------------------------------------------------
# Direct execution


def main() -> None:
    n_bar = 0.01

    print("Stage 5B ideal-PNA + extra-attenuation continuum benchmark")
    print(
        f"M = {M_DEFAULT}, n_bar = {n_bar}, tau_s = {TAU_S:g}, "
        f"c_max = {C_MAX}, lambda = {LAMBDA_ATT:g}, "
        f"P_mon,both = {P_MON_BOTH:g}, N_U = {N_U}, "
        f"workers = {MAX_WORKERS}"
    )
    print(
        "Scanning distance and guard offset N_plus-c_max in parallel. "
        "This uses a 50:50 ideal PNA benchmark followed by strong trusted "
        "attenuation and the certified Stage-3/4 lifted continuum relaxation."
    )

    results = run_scan(
        M=M_DEFAULT,
        n_bar=n_bar,
    )

    print()
    print_scan(results)

    paths = plot_scan(results)
    print("\nSaved Stage-5B plots:")
    for path in paths:
        print(f"  {path}")

    print(
        "\nThis is an appendix-style benchmark.  The PNA response and "
        "P_mon,both are still idealized/placeholder inputs; a realistic "
        "detector model should replace the ideal tail interface before "
        "publication-level physical claims are made."
    )


if __name__ == "__main__":
    n_bar=0.01
    results=run_scan(M=M_DEFAULT,n_bar=n_bar)
    rows=[]
    for r in results:
        rows.append(result_row(
            "stage5b",f"offset={r['offset']}",r["L_km"],
            M=M_DEFAULT,n_bar=n_bar,e_ph_upper=r["e_ph"],
            e_bit=r["e_bit"],P_pass_cond=r["Ppass_mon"],
            R_acc=r["R_acc"],R_original=r["R_original"],
            status=r["status"],P_monitor=P_MON_BOTH,
            P_monitor_scope="illustrative lower bound",
            epsilon_tail=r["epsilon_5"],beta=r["beta"],
            tail_kind="PNA-certified pre-extra-loss tail",
            phase_bound_scope="unrestricted e_ph SDP; conservative entropy when >=1/2",
            parameters={"offset":r["offset"],"N_plus":r["N_plus"],
                        "tau_s":TAU_S,"c_max":C_MAX,
                        "lambda_att":LAMBDA_ATT,"N_U":N_U},
            notes="Ideal PNA plus extra trusted loss/continuum; acceptance 0.99 is a placeholder.",
        ))
    report(rows,stage="stage5b")
    plot_scan(results)
