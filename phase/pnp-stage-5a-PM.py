"""
Stage 5A--PM: plug-and-play phase-matching / basis-dependent test attenuation.

Derivation: main.tex, Section "Stage 5A--PM".
This intentionally imports the original Stage 5A *monitor certification* functions
rather than modifying phase/pnp-stage-5a.py.

For a PNA-accepted untrusted source with arbitrary joint Fock diagonal q,
the source-overlap kernel for labels s,t is

 h_X(s,t) = sqrt((1-lambda_Xs)(1-lambda_Xt))
          + sqrt(lambda_Xs lambda_Xt) exp(i (phi_Xt-phi_Xs)),

 Lambda_st = sum_{nA,nB} q[nA,nB] h_A(s,t)**nA h_B(s,t)**nB.

A trusted vacuum-environment pure-loss dilation implements lambda_Xx.
Its loss environment is conservatively handed to Eve in the overlap proof.
The physical PNA--BS tail and occupancy interface is *unchanged*.  Only
the overlap kernel and setting-dependent observed diagonal probabilities
change.  The finite Gram SDP is still an OUTER RELAXATION of the exact
accepted-state physical image.

Default test data are a COHERENT-REFERENCE SYNTHETIC CHANNEL, not measured
detector statistics and not a certified PNA acceptance probability.
Use calibrated P(z | s, A_mon), monitor response and measured P_mon for a
physical security claim.  R_original is conditional-key-rate times the
supplied acceptance lower bound, not a certified PLOB comparison.

No CVXPY solve happens at import time. Direct execution runs kernel identity
tests and a profile comparison; --optimize instead runs a multiprocessing
search of the test-basis transmissions. This search is heuristic, not a
global-optimality claim. The synthetic monitor-data limitations still apply.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import math
import os
from multiprocessing import get_context
import random
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pnp_benchmark as benchmark
from pnp_results import result_row, report, stage_results_dir

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

# Keep the existing PNA certification implementation as the single source
# of truth; the file name has hyphens so it cannot be imported normally.
_STAGE5A_PATH = Path(__file__).resolve().with_name("pnp-stage-5a.py")
_module_spec = importlib.util.spec_from_file_location("pnp_stage5a_base", _STAGE5A_PATH)
if _module_spec is None or _module_spec.loader is None:
    raise ImportError(f"Could not load {_STAGE5A_PATH}")
S5 = importlib.util.module_from_spec(_module_spec)
sys.modules[_module_spec.name] = S5
_module_spec.loader.exec_module(S5)

Z_P, Z_M, Z_F, Z_ALL = S5.Z_P, S5.Z_M, S5.Z_F, S5.Z_ALL
F_EC = S5.F_EC
SLACK_FLOOR = S5.SLACK_FLOOR
Stage5AResult = S5.Stage5AResult
good_fock_support = S5.good_fock_support
_check_probability = S5._check_probability
_check_nonnegative_int = S5._check_nonnegative_int
h2 = S5.h2


@dataclass(frozen=True)
class PMBasisSetting:
    """Basis x: two phases and two trusted transmissions after PNA acceptance."""
    theta_a: float
    theta_b: float
    lambda_a: float
    lambda_b: float


def make_pm_settings(
    lambda_by_basis: Sequence[float],
    *,
    lambda_a_by_basis: Sequence[float] | None = None,
    lambda_b_by_basis: Sequence[float] | None = None,
) -> tuple[PMBasisSetting, ...]:
    """Default theta_x = x*pi/M; input values represent LOSS transmissivities."""
    M = len(lambda_by_basis)
    if M < 1:
        raise ValueError("At least one basis is needed.")
    la = lambda_by_basis if lambda_a_by_basis is None else lambda_a_by_basis
    lb = lambda_by_basis if lambda_b_by_basis is None else lambda_b_by_basis
    if len(la) != M or len(lb) != M:
        raise ValueError("All setting lists must have the same length.")
    settings = tuple(
        PMBasisSetting(
            theta_a=x * np.pi / M,
            theta_b=x * np.pi / M,
            lambda_a=float(la[x]),
            lambda_b=float(lb[x]),
        )
        for x in range(M)
    )
    validate_pm_settings(settings, M)
    return settings


def validate_pm_settings(settings: Sequence[PMBasisSetting], M: int) -> None:
    if len(settings) != M:
        raise ValueError(f"Expected M={M} basis settings.")
    for x, setting in enumerate(settings):
        for name in ("lambda_a", "lambda_b"):
            value = getattr(setting, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Basis {x}: {name} must be in [0,1].")
        if not np.isfinite(setting.theta_a) or not np.isfinite(setting.theta_b):
            raise ValueError(f"Basis {x}: phase angles must be finite.")


def pm_state_labels(
    settings: Sequence[PMBasisSetting],
) -> list[tuple[float, float, float, float]]:
    """
    Four bit pairs per basis, ordered 00,01,10,11.
    Each label (phase A,phase B,loss A,loss B) is normalized before
    constructing the Gram SDP; loss is bit-independent within one basis.
    """
    out = []
    for b in settings:
        for da, db in S5.ENC_PHASES:
            out.append((
                b.theta_a + da, b.theta_b + db,
                b.lambda_a, b.lambda_b,
            ))
    return out


def one_arm_pm_factor(
    phi_s: float, phi_t: float, lambda_s: float, lambda_t: float
) -> complex:
    return complex(
        np.sqrt((1.0 - lambda_s) * (1.0 - lambda_t))
        + np.sqrt(lambda_s * lambda_t) * np.exp(1j * (phi_t - phi_s))
    )


def pm_overlap_kernel(
    states: Sequence[tuple[float, float, float, float]],
    pairs: Sequence[tuple[int, int]],
    support: np.ndarray,
) -> np.ndarray:
    """K[(s,t),(nA,nB)]=hA**nA * hB**nB, affine in surviving-Fock q."""
    support = np.asarray(support, dtype=int)
    if support.ndim != 2 or support.shape[1] != 2 or np.any(support < 0):
        raise ValueError("support must have shape (n,2) with nonnegative entries.")
    result = np.empty((len(pairs), len(support)), dtype=complex)
    for k, (s, t) in enumerate(pairs):
        pa_s, pb_s, la_s, lb_s = states[s]
        pa_t, pb_t, la_t, lb_t = states[t]
        h_a = one_arm_pm_factor(pa_s, pa_t, la_s, la_t)
        h_b = one_arm_pm_factor(pb_s, pb_t, lb_s, lb_t)
        result[k, :] = h_a ** support[:, 0] * h_b ** support[:, 1]
    return result


def check_pm_kernel_identities() -> None:
    """No solver needed: checks diagonal, Hermiticity, Stinespring and limits."""
    M = 4
    supp = S5.good_fock_support(3, 3)
    fixed = make_pm_settings([1.0]*M)
    phases = S5.state_phases(M)
    states = pm_state_labels(fixed)
    pairs = [(0, 3), (0, 5), (4, 10), (11, 15)]
    assert np.allclose(
        pm_overlap_kernel(states, pairs, supp),
        S5.good_overlap_kernel(phases, pairs, supp),
        atol=2e-14, rtol=2e-14,
    ), "lambda=1 must reproduce Stage 5A exactly"
    for lam in (0.0, 0.3, 1.0):
        states = pm_state_labels(make_pm_settings([lam]*M))
        for s,t in pairs:
            hs = one_arm_pm_factor(states[s][0], states[t][0],lam,lam)
            expect = 1-lam + lam*np.exp(1j*(states[t][0]-states[s][0]))
            assert abs(hs-expect)<2e-14, "uniform loss limit"
    varied = pm_state_labels(make_pm_settings([1.0,0.8,0.5,0.2]))
    for s,t in pairs:
        ks = pm_overlap_kernel(varied, [(s,t),(t,s)], supp)
        assert np.allclose(ks[1],np.conjugate(ks[0]),atol=2e-14)
    # Shared observed-statistics regression: fixed PM and Stage5A are
    # exactly the same coherent-reference detection benchmark.
    for length in (0.0, 20.0):
        actual=pm_synthetic_announcement_probabilities(
            fixed,0.01,length,loss_only=False
        )
        expected,_,_=S5.synthetic_announcement_probabilities(M,0.01,length)
        for z in Z_ALL:
            assert np.allclose(actual[z],expected[z],rtol=0,atol=1e-15), (
                "PM lambda=1 must reproduce Stage5A detector statistics"
            )
    for n in range(5):
        ls,lt,delta = 0.2,0.75,0.43
        h = one_arm_pm_factor(0.0,delta,ls,lt)
        explicit = sum(
            math.comb(n,k) * (ls*lt)**(k/2)
            * ((1-ls)*(1-lt))**((n-k)/2)
            * np.exp(1j*k*delta)
            for k in range(n+1)
        )
        assert abs(h**n-explicit)<2e-14, "isometric Fock overlap"


def pm_synthetic_announcement_probabilities(
    settings: Sequence[PMBasisSetting],
    n_bar_before_loss: float,
    distance_km: float,
    *,
    loss_only: bool = False,
) -> dict[int, np.ndarray]:
    """
    Same coherent-reference two-arm detector model as Stage 5A.
    Only the basis-dependent trusted attenuation and phase settings differ.
    Consequently the PM lambda_x=1 baseline MUST reproduce the Stage5A
    synthetic probabilities and SDP (same monitor parameters/cutoff).
    """
    M=len(settings)
    validate_pm_settings(settings,M)
    observed,_,_=benchmark.coherent_reference_probabilities(
        M,n_bar_before_loss,distance_km,
        lambda_a=[s.lambda_a for s in settings],
        lambda_b=[s.lambda_b for s in settings],
        theta_a=[s.theta_a for s in settings],
        theta_b=[s.theta_b for s in settings],
        eta_det=S5.ETA_DET,fiber_loss_db_per_km=S5.XI,
        p_dc=S5.P_DC,e_ali=S5.E_ALI,loss_only=loss_only,
    )
    return observed


def solve_stage5a_pm_sdp(
    *,
    settings: Sequence[PMBasisSetting],
    M: int,
    nA_plus: int,
    nB_plus: int,
    epsilon_5: float,
    prob_of: dict[int, Sequence[float]],
    p_mon: float | None = None,
    marginal_cap_a: Sequence[float] | None = None,
    marginal_cap_b: Sequence[float] | None = None,
    joint_cap: np.ndarray | None = None,
    f_ec: float = F_EC,
    solver=None,
    solver_opts=None,
) -> Stage5AResult:
    """
    Solve the finite Stage-5A-PM Gram outer relaxation with variable test losses.

    Inputs
    ------
    M
        Number of phase bases.  There are NS=4*M signal labels.
    nA_plus, nB_plus
        Certified good-component surviving-photon cutoffs.
    epsilon_5
        Certified two-arm conditional bad mass:

            Pr(
                N_A>nA_plus or N_B>nB_plus
                | A_mon
            ) <= epsilon_5.

    prob_of
        Accepted-ensemble observed probabilities.  Keys must be
        Z_P, Z_M, Z_F and each value must have length 4*M:

            prob_of[z][s] = P(z | s, A_mon).

    p_mon
        Optional observed physical monitor acceptance probability.  If
        supplied, R_original = p_mon * R_acc is reported.  It does not enter
        the conditional Gram SDP itself.
    marginal_cap_a, marginal_cap_b
        Optional rigorous upper bounds on accepted-ensemble marginal
        occupancies for n=0,...,nA_plus / nB_plus.  These constrain the good
        masses by
            sum_{n_B} q[n_A,n_B] <= marginal_cap_a[n_A],
            sum_{n_A} q[n_A,n_B] <= marginal_cap_b[n_B].
    joint_cap
        Optional array of shape (nA_plus+1,nB_plus+1) with rigorous upper
        bounds on individual accepted joint occupancies q[n_A,n_B].

    The SDP variables are
        G      : Gram matrix of central-node residual vectors,
        q      : unnormalised good accepted Fock masses,
        beta   : arbitrary bad accepted mass.

    Constraints are
        G >= 0,
        diag(G) = observed accepted-ensemble statistics,
        q >= 0,
        0 <= beta <= epsilon_5,
        sum(q) + beta = 1,
        optional monitor-derived marginal/joint occupancy caps,
        |Gamma_st(G) - Lambda_good_st(q)| <= beta.

    The incoming source distribution p_{m_A,m_B} does not appear.  This is an
    intentional outer relaxation of the exact PNA-BS map.
    """
    if M < 1:
        raise ValueError("M must be positive.")
    _check_nonnegative_int("nA_plus", nA_plus)
    _check_nonnegative_int("nB_plus", nB_plus)
    _check_probability("epsilon_5", epsilon_5)

    if p_mon is not None:
        _check_probability("p_mon", p_mon, positive=True)

    cap_a = None
    cap_b = None
    cap_joint = None

    if marginal_cap_a is not None:
        cap_a = np.asarray(marginal_cap_a, dtype=float)
        if cap_a.shape != (nA_plus + 1,):
            raise ValueError(
                "marginal_cap_a must have length nA_plus+1."
            )
        if np.any(cap_a < 0.0) or np.any(cap_a > 1.0 + 1e-12):
            raise ValueError("marginal_cap_a entries must lie in [0,1].")
        cap_a = np.clip(cap_a, 0.0, 1.0)

    if marginal_cap_b is not None:
        cap_b = np.asarray(marginal_cap_b, dtype=float)
        if cap_b.shape != (nB_plus + 1,):
            raise ValueError(
                "marginal_cap_b must have length nB_plus+1."
            )
        if np.any(cap_b < 0.0) or np.any(cap_b > 1.0 + 1e-12):
            raise ValueError("marginal_cap_b entries must lie in [0,1].")
        cap_b = np.clip(cap_b, 0.0, 1.0)

    if joint_cap is not None:
        cap_joint = np.asarray(joint_cap, dtype=float)
        expected = (nA_plus + 1, nB_plus + 1)
        if cap_joint.shape != expected:
            raise ValueError(
                f"joint_cap must have shape {expected}, got "
                f"{cap_joint.shape}."
            )
        if np.any(cap_joint < 0.0) or np.any(cap_joint > 1.0 + 1e-12):
            raise ValueError("joint_cap entries must lie in [0,1].")
        cap_joint = np.clip(cap_joint, 0.0, 1.0)

    ns = 4 * M
    validate_pm_settings(settings, M)
    pm_states = pm_state_labels(settings)

    # Validate observed probabilities.
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
            "For each signal label, accepted-ensemble announcement "
            "probabilities must sum to 1."
        )

    # The key-basis pass probability is an observed constant.
    p_pass_mon = 0.25 * sum(
        probs[Z_P][ab] + probs[Z_M][ab]
        for ab in range(4)
    )
    if p_pass_mon <= 0.0:
        raise ValueError("Observed P_pass|mon must be positive.")

    # Bit error from the same key-basis observed statistics.  For 00 and 11,
    # '+' is the nominal correct result; for 01 and 10, '-' is correct.
    error_prob = 0.25 * (
        probs[Z_M][0]
        + probs[Z_P][1]
        + probs[Z_P][2]
        + probs[Z_M][3]
    )
    e_bit = float(error_prob / p_pass_mon)

    # Remove zero-norm residual vectors.  PSD forces their whole row/column to
    # zero, and removing them improves numerical strict feasibility.
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

    # Observed accepted-ensemble announcement probabilities.
    cons.append(
        cp.real(cp.diag(G))
        == np.asarray([probs[z][s] for (z, s) in active])
    )

    # Finite good accepted-state Fock masses plus arbitrary bad mass.
    support = good_fock_support(nA_plus, nB_plus)
    nq = len(support)

    q = cp.Variable(nq, nonneg=True)
    beta = cp.Variable(nonneg=True)

    cons += [
        cp.sum(q) + beta == 1.0,
        beta <= epsilon_5,
    ]

    # Optional PNA-derived occupancy caps.  q contains only the good
    # rectangular component, so each good marginal is no larger than the
    # corresponding full accepted marginal; the same upper bounds remain
    # valid after discarding the bad component.
    if cap_joint is not None:
        joint_vec = np.asarray(
            [cap_joint[nA, nB] for nA, nB in support],
            dtype=float,
        )
        cons.append(q <= joint_vec)

    if cap_a is not None:
        for nA in range(nA_plus + 1):
            idx = np.where(support[:, 0] == nA)[0]
            cons.append(cp.sum(q[idx]) <= cap_a[nA])

    if cap_b is not None:
        for nB in range(nB_plus + 1):
            idx = np.where(support[:, 1] == nB)[0]
            cons.append(cp.sum(q[idx]) <= cap_b[nB])

    # Every unordered signal pair gets one overlap-conservation SOC.
    pairs = [
        (s, t)
        for s in range(ns)
        for t in range(s + 1, ns)
    ]

    kernel = pm_overlap_kernel(pm_states, pairs, support)
    good_overlap = kernel @ q

    gram_terms = []
    for s, t in pairs:
        gram_terms.append(sum(ent(z, s, z, t) for z in Z_ALL))
    gram_overlap = cp.hstack(gram_terms)

    diff = gram_overlap - good_overlap
    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(
        cp.norm(resid, 2, axis=1)
        <= beta + SLACK_FLOOR
    )

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
        "n_good_states": int(nq),
        "gram_dim": int(dim),
        "n_pairs": int(len(pairs)),
        "p_mon": None if p_mon is None else float(p_mon),
        "objective_scalar": None,
        "uses_marginal_cap_a": cap_a is not None,
        "uses_marginal_cap_b": cap_b is not None,
        "uses_joint_cap": cap_joint is not None,
        "q_marginal_a": None,
        "q_marginal_b": None,
    }

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return Stage5AResult(
            e_ph_upper=1.0,
            e_bit=e_bit,
            p_pass_mon=float(p_pass_mon),
            r_acc=0.0,
            r_original=None if p_mon is None else 0.0,
            status=prob.status,
            q_good=None,
            support=support,
            diagnostics=base_diag,
        )

    # A7:
    #   e_ph = 1/2 + Re(bracket)/(4 P_pass|mon).
    e_ph_upper = float(
        np.clip(
            0.5 + float(prob.value) / (4.0 * p_pass_mon),
            0.0,
            1.0,
        )
    )

    # For privacy amplification, if the only information is e_ph <= u with
    # u >= 1/2, the worst binary entropy in that interval is h2(1/2)=1, so
    # the key rate is zero.  Do not feed h2(u>1/2) directly into the rate.
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
    q_marginal_a = np.asarray([
        np.sum(q_val[support[:, 0] == nA])
        for nA in range(nA_plus + 1)
    ])
    q_marginal_b = np.asarray([
        np.sum(q_val[support[:, 1] == nB])
        for nB in range(nB_plus + 1)
    ])

    diagnostics = {
        **base_diag,
        "beta": float(beta.value),
        "good_mass": float(np.sum(q_val)),
        "objective_scalar": float(prob.value),
        "q_marginal_a": q_marginal_a,
        "q_marginal_b": q_marginal_b,
    }

    return Stage5AResult(
        e_ph_upper=e_ph_upper,
        e_bit=e_bit,
        p_pass_mon=float(p_pass_mon),
        r_acc=r_acc,
        r_original=r_original,
        status=prob.status,
        q_good=q_val,
        support=support,
        diagnostics=diagnostics,
    )



def solve_pm_ideal_benchmark(
    *,
    settings: Sequence[PMBasisSetting],
    n_bar_before_loss: float,
    distance_km: float,
    tau_s: float,
    c_max: int,
    n_plus: int,
    p_mon_both_lower: float,
    loss_only: bool = False,
    solver=None,
    solver_opts=None,
) -> tuple[Stage5AResult,S5.IdealTwoArmInterface]:
    """
    Ideal hard-threshold PNA--BS certification plus PM Gram SDP.

    p_mon_both_lower is an externally justified *lower bound* on the
    both-monitor acceptance probability; it is not synthesized from n_bar.
    """
    M = len(settings)
    validate_pm_settings(settings, M)
    interface = S5.ideal_two_arm_interface(
        tau_s_a=tau_s, tau_s_b=tau_s,
        c_max_a=c_max, c_max_b=c_max,
        nA_plus=n_plus, nB_plus=n_plus,
        p_mon_both=p_mon_both_lower,
    )
    cap_a,cap_b,cap_joint = S5.ideal_two_arm_occupancy_caps(
        nA_plus=n_plus,nB_plus=n_plus,
        tau_s_a=tau_s,tau_s_b=tau_s,
        c_max_a=c_max,c_max_b=c_max,
        p_mon_both=p_mon_both_lower,
    )
    observed = pm_synthetic_announcement_probabilities(
        settings,n_bar_before_loss,distance_km,loss_only=loss_only,
    )
    solution = solve_stage5a_pm_sdp(
        settings=settings, M=M,
        nA_plus=n_plus,nB_plus=n_plus,
        epsilon_5=interface.epsilon_5,
        prob_of=observed,
        p_mon=p_mon_both_lower,
        marginal_cap_a=cap_a,marginal_cap_b=cap_b,
        joint_cap=cap_joint,
        solver=solver,solver_opts=solver_opts,
    )
    return solution,interface


def scan_pm_profiles(
    *,
    profiles: dict[str, Sequence[PMBasisSetting]],
    distances_km: Sequence[float],
    n_bar_before_loss: float,
    tau_s: float,
    c_max: int,
    n_plus: int,
    p_mon_both_lower: float,
    loss_only: bool = False,
    solver=None,
) -> list[dict]:
    """Compare chosen (not automatically optimized) test-loss profiles."""
    output = []
    for name,settings in profiles.items():
        for distance in distances_km:
            result,interface = solve_pm_ideal_benchmark(
                settings=settings,
                n_bar_before_loss=n_bar_before_loss,
                distance_km=distance,tau_s=tau_s,c_max=c_max,
                n_plus=n_plus,p_mon_both_lower=p_mon_both_lower,
                loss_only=loss_only,solver=solver,
            )
            output.append(dict(
                profile=name,L_km=distance,
                e_ph=result.e_ph_upper,e_bit=result.e_bit,
                p_pass=result.p_pass_mon,
                R_acc=result.r_acc,R_original=result.r_original,
                epsilon_5=interface.epsilon_5,
                beta=result.diagnostics.get("beta"),
                N_plus=n_plus,
                status=result.status,
            ))
    return output


def plot_pm_profiles(results: Sequence[dict], output_dir=None, *, filename_prefix="pm"):
    """Plot phase-error and rate comparisons with equal key-basis loss."""
    if not results:
        raise ValueError("No results to plot.")
    if output_dir is None:
        output_dir = stage_results_dir("stage5a-PM")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    for ykey,ylabel,filename in (
        ("e_ph",r"$e_{\rm ph}^{\rm upper}$",f"{filename_prefix}-eph-comparison.png"),
        ("R_original",r"$R_{\rm original}$",f"{filename_prefix}-rate-comparison.png"),
    ):
        fig,ax=plt.subplots(figsize=(7.0,4.5))
        for profile in dict.fromkeys(r["profile"] for r in results):
            rows = sorted(
                (r for r in results if r["profile"]==profile),
                key=lambda r:r["L_km"],
            )
            ax.plot(
                [r["L_km"] for r in rows],
                [r[ykey] for r in rows],marker="o",label=profile,
            )
        ax.set_xlabel("Total Alice-Bob distance (km)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        target=output_dir/filename
        fig.savefig(target,dpi=180)
        plt.close(fig)
        print(f"Saved {target}")


def main() -> None:
    check_pm_kernel_identities()
    print("Basis-dependent loss kernel identities: PASS")
    M=4
    # The key basis x=0 has lambda=1 in BOTH profiles, so key-state
    # physics is the same; only test-state intensities differ.
    profiles={
        "fixed (Stage5A)":make_pm_settings([1.0]*M),
        "basis-loss PM":make_pm_settings([1.0,0.8,0.5,0.2]),
    }
    print(
        "Ideal PNA+BS benchmark, coherent-reference synthetic accepted stats; "
        "P_mon LOWER BOUND is placeholder 0.99. "
        "No claim of certified detector performance or PLOB beating."
    )
    results=scan_pm_profiles(
        profiles=profiles,
        distances_km=(0.0,20.0,40.0),
        n_bar_before_loss=0.01,
        tau_s=1e-8,c_max=1_000_000,
        n_plus=3,p_mon_both_lower=0.99,
    )
    plot_pm_profiles(results)


# ---------------------------------------------------------------------------
# Parallel PM profile optimization (lambda_0 fixed at one)
# ---------------------------------------------------------------------------

def latin_hypercube(n: int, dimension: int, lo: float, rng: random.Random):
    """Reproducible stratified candidates without any optimization dependency."""
    if n < 0 or dimension < 1 or not 0.0 <= lo < 1.0:
        raise ValueError("Invalid Latin-hypercube arguments")
    axes = []
    for _ in range(dimension):
        values = [lo + (1.0 - lo) * (k + rng.random()) / n for k in range(n)] if n else []
        rng.shuffle(values)
        axes.append(values)
    return [tuple(axes[j][i] for j in range(dimension)) for i in range(n)]


def _key(test_transmissions):
    return tuple(round(float(x), 7) for x in test_transmissions)


def _evaluate(task):
    """Top-level pickleable worker: one SDP solve with no nested parallelism."""
    distance, test, config = task
    started = time.monotonic()
    record = {
        "distance_km": float(distance),
        "lambda_by_basis": [1.0, *test],
        "status": "error",
        "e_ph": None,
        "e_bit": None,
        "p_pass": None,
        "R_acc": None,
        "R_original": None,
        "elapsed_s": None,
        "error": None,
    }
    try:
        settings = make_pm_settings(record["lambda_by_basis"])
        solver = None if config["solver"] == "auto" else getattr(cp, config["solver"].upper())
        result, _ = solve_pm_ideal_benchmark(
            settings=settings,
            n_bar_before_loss=config["n_bar"],
            distance_km=distance,
            tau_s=config["tau_s"],
            c_max=config["c_max"],
            n_plus=config["n_plus"],
            p_mon_both_lower=config["p_mon"],
            loss_only=config["loss_only"],
            solver=solver,
        )
        record.update(
            status=str(result.status),
            e_ph=float(result.e_ph_upper),
            e_bit=float(result.e_bit),
            p_pass=float(result.p_pass_mon),
            R_acc=float(result.r_acc),
            R_original=None if result.r_original is None else float(result.r_original),
        )
        if any(not math.isfinite(record[k]) for k in ("e_ph", "e_bit", "p_pass", "R_acc")):
            record["status"] = "nonfinite"
            for name in ("e_ph", "e_bit", "p_pass", "R_acc", "R_original"):
                if record[name] is not None and not math.isfinite(record[name]):
                    record[name] = None
        if record["status"] != "optimal":
            record["error"] = "Not an accepted optimum; requires solver verification"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    record["elapsed_s"] = round(time.monotonic() - started, 3)
    return record


def _rank(record):
    """Maximize accepted-round rate; minimize phase error to break zero-rate ties."""
    if record["status"] != "optimal":
        return (-math.inf, -math.inf)
    return (record["R_acc"], -record["e_ph"])


def _seed_profiles(dimension, lo):
    """Always include fixed, the original hand-picked PM profile and low loss."""
    fixed = (1.0,) * dimension
    if dimension == 3:
        handpicked = (0.8, 0.5, 0.2)
    else:
        handpicked = tuple(1.0 - 0.8 * (i + 1) / dimension for i in range(dimension))
    handpicked = tuple(max(lo, value) for value in handpicked)
    return [fixed, handpicked, (lo,) * dimension, ((1.0 + lo) / 2,) * dimension]


def optimize_distance(executor, distance, config, *, random_samples, refine_rounds, rng):
    """Parallel batch search, with exact-evaluation caching within this distance."""
    dimension = config["M"] - 1
    lo = config["lambda_min"]
    seen = {}

    def evaluate_batch(points):
        pending = {}
        for point in points:
            point = _key(tuple(min(1.0, max(lo, value)) for value in point))
            if point not in seen and point not in pending:
                pending[point] = executor.submit(_evaluate, (distance, point, config))
        by_future = {future: key for key, future in pending.items()}
        for future in as_completed(by_future):
            key = by_future[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"distance_km": distance, "lambda_by_basis": [1.0, *key],
                          "status": "worker_error", "R_acc": None, "e_ph": None,
                          "error": f"{type(exc).__name__}: {exc}"[:500]}
            seen[key] = result
        return len(pending)

    initial = _seed_profiles(dimension, lo)
    initial += latin_hypercube(random_samples, dimension, lo, rng)
    evaluated = evaluate_batch(initial)
    valid = [row for row in seen.values() if row["status"] == "optimal"]
    if not valid:
        raise RuntimeError(f"Distance {distance:g} km: no solver-certified 'optimal' status. "
                           "Try --solver mosek or inspect the saved solver diagnostics.")
    best = max(valid, key=_rank)
    step = 0.25 * (1.0 - lo)
    for _ in range(refine_rounds):
        test = best["lambda_by_basis"][1:]
        neighbors = []
        for j in range(dimension):
            for direction in (-1, 1):
                neighbor = list(test)
                neighbor[j] += direction * step
                neighbors.append(tuple(neighbor))
        evaluated += evaluate_batch(neighbors)
        candidate = max((r for r in seen.values() if r["status"] == "optimal"), key=_rank)
        if _rank(candidate) > _rank(best):
            best = candidate
        else:
            step *= 0.5
        if step < 0.002:
            break
    fixed = seen[_key((1.0,) * dimension)]
    return best, fixed, list(seen.values())


def parse_pm_opt_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimize", action="store_true", help="Run parallel PM transmission optimization")
    parser.add_argument("--distances", type=float, nargs="+", default=[0.0, 20.0, 40.0])
    parser.add_argument("--M", type=int, default=4, help="Number of bases, including key basis")
    parser.add_argument("--n-bar", type=float, default=0.01, help="Input intensity before trusted basis loss")
    parser.add_argument("--tau-s", type=float, default=1e-8)
    parser.add_argument("--c-max", type=int, default=1_000_000)
    parser.add_argument("--n-plus", type=int, default=3)
    parser.add_argument("--p-mon", type=float, default=0.99,
                        help="Externally justified monitor-acceptance lower bound; 0.99 is placeholder")
    parser.add_argument("--loss-only", action="store_true")
    parser.add_argument("--lambda-min", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=18, help="Global Latin-hypercube samples per distance")
    parser.add_argument("--refine-rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--solver", choices=["auto", "mosek", "clarabel"], default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=stage_results_dir("stage5a-PM") / "optimization-results.json")
    parser.add_argument("--plot", action="store_true", help="Plot optimized and fixed profiles")
    args = parser.parse_args(argv)
    if not 2 <= args.M or not 0 <= args.lambda_min < 1:
        parser.error("Require M >= 2 and 0 <= --lambda-min < 1")
    if args.workers < 1 or args.samples < 0 or args.refine_rounds < 0:
        parser.error("workers >= 1, samples >= 0 and refine-rounds >= 0 required")
    if not 0 < args.p_mon <= 1 or not 0 <= args.tau_s <= 1 or args.n_bar <= 0:
        parser.error("Invalid monitor or mean-energy parameters")
    if any(d < 0 for d in args.distances):
        parser.error("Distances must be nonnegative")
    return args


def main_optimize(argv=None):
    args = parse_pm_opt_args(argv)
    config = {"M": args.M, "n_bar": args.n_bar, "tau_s": args.tau_s,
              "c_max": args.c_max, "n_plus": args.n_plus, "p_mon": args.p_mon,
              "loss_only": args.loss_only, "lambda_min": args.lambda_min,
              "solver": args.solver}
    rng = random.Random(args.seed)
    data = {"config": {**config, "distances_km": args.distances, "samples": args.samples,
                       "refine_rounds": args.refine_rounds, "workers": args.workers,
                       "seed": args.seed},
            "warning": "Synthetic coherent-reference benchmark; p_mon is externally supplied. "
                       "A numerical SDP optimum is not a rigorous dual certificate. "
                       "Parallel search is heuristic, not globally optimal.",
            "best_by_distance": [], "evaluations": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Optimizing Stage 5A--PM test transmissions (lambda_0=1; key unchanged).", flush=True)
    print("WARNING: coherent-reference synthetic statistics and illustrative acceptance bound.", flush=True)
    plot_rows = []
    # Spawn clean worker interpreters rather than forking CVXPY's initialized
    # BLAS state. Child processes see these limits before importing NumPy.
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        for distance in args.distances:
            best, fixed, evaluated = optimize_distance(
                pool, distance, config, random_samples=args.samples,
                refine_rounds=args.refine_rounds, rng=rng,
            )
            data["evaluations"].extend(evaluated)
            data["best_by_distance"].append({"distance_km": distance, "optimized": best,
                                             "fixed": fixed, "n_evaluated": len(evaluated)})
            for name, record in (("fixed (Stage5A)", fixed), ("optimized basis-loss PM", best)):
                plot_rows.append({"profile": name, "L_km": distance,
                                  "e_ph": record["e_ph"], "R_original": record.get("R_original")})
            print(f"{distance:g} km: {len(evaluated)} evaluations; fixed R_acc={fixed.get('R_acc')}; "
                  f"best R_acc={best['R_acc']:.6g}, e_ph={best['e_ph']:.6g}, "
                  f"lambda={best['lambda_by_basis']}", flush=True)
            # Checkpoint completed distances even if a later distance fails.
            args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    if args.plot:
        plot_pm_profiles(plot_rows, output_dir=stage_results_dir("stage5a-PM"), filename_prefix="pm-optimized")
    print(f"Saved {args.output}")






def main_profiles() -> None:
    check_pm_kernel_identities()
    print("Kernel and synthetic-statistics unit-limit checks: PASS")
    M=4
    profiles={
        "fixed (Stage5A)":make_pm_settings([1.0]*M),
        "basis-loss PM":make_pm_settings([1.0,0.8,0.5,0.2]),
    }
    n_bar=0.01
    n_plus=3
    p_mon=0.99  # illustrative, NOT physically derived/verified
    results=scan_pm_profiles(
        profiles=profiles,
        distances_km=S5.DISTANCES_KM,
        n_bar_before_loss=n_bar,
        tau_s=1e-8,c_max=1_000_000,
        n_plus=n_plus,p_mon_both_lower=p_mon,
    )
    rows=[]
    for r in results:
        chosen=profiles[r["profile"]]
        rows.append(result_row(
            "stage5a-PM",r["profile"],r["L_km"],
            M=M,n_bar=n_bar,e_ph_upper=r["e_ph"],
            e_bit=r["e_bit"],P_pass_cond=r["p_pass"],
            R_acc=r["R_acc"],R_original=r["R_original"],
            status=r["status"],P_monitor=p_mon,
            P_monitor_scope="illustrative lower bound",
            epsilon_tail=r["epsilon_5"],beta=r["beta"],
            tail_kind="PNA-certified conditional surviving-Fock tail",
            phase_bound_scope="unrestricted e_ph SDP; conservative entropy when >=1/2",
            parameters={"N_plus":n_plus,"tau_s":1e-8,"c_max":1_000_000,
                       "lambda_a":[s.lambda_a for s in chosen],
                       "lambda_b":[s.lambda_b for s in chosen]},
            notes="PM basis-dependent post-monitor pure loss; coherent-reference stats; P_monitor=0.99 placeholder.",
        ))
    report(rows,stage="stage5a-PM")
    plot_pm_profiles(results)


if __name__ == "__main__":
    if "--optimize" in sys.argv[1:]:
        check_pm_kernel_identities()
        main_optimize(sys.argv[1:])
    else:
        main_profiles()
