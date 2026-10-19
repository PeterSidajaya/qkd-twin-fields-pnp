"""
Common COHERENT-REFERENCE SYNTHETIC announcement model for PnP SDP scans.

NOT an observed data set and NOT a proof that an untrusted source is coherent.
All stages use total Alice--Bob fiber distance L_km, with two equal arms
of length L_km/2 and arm transmissivity
    eta_arm = eta_det * 10**(-fiber_loss_db_per_km*L_km/20).

n_bar is the nominal TOTAL pre-setting-attenuation mean photon number
in the two signal arms at the users' encoding stage.  Stage 2/3's source
energy constraints still use their existing, separately stated conventions;
only synthetic central-node statistics are generated here.

Basis x can have calibrated trusted post-monitor losses lambda_a[x],
lambda_b[x]. The default is no additional loss, lambda_a=lambda_b=1.

Charlie interferes coherent reference fields on a balanced beam splitter.
We model imperfect visibility V=1-2*e_ali and independent on/off click
detectors with per-detector dark probability 1-exp(-p_dc). Outcomes
are exactly one "+" click, exactly one "-" click, or failure (zero/two clicks).

This is a consistent benchmark across scripts, not a physically derived
monitor acceptance probability or a general adversarial-source channel model.
"""
from __future__ import annotations

import numpy as np


def coherent_reference_probabilities(
    M: int,
    n_bar: float,
    L_km: float,
    *,
    lambda_a=None,
    lambda_b=None,
    theta_a=None,
    theta_b=None,
    eta_det: float = 0.85,
    fiber_loss_db_per_km: float = 0.2,
    p_dc: float = 5e-8,
    e_ali: float = 0.015,
    loss_only: bool = False,
):
    """Return (prob_of[z][s], key-basis P_pass, key-basis e_bit)."""
    if M < 1 or int(M) != M:
        raise ValueError("M must be a positive integer.")
    if n_bar < 0 or L_km < 0:
        raise ValueError("n_bar and L_km must be nonnegative.")
    if not 0 <= eta_det <= 1 or fiber_loss_db_per_km < 0:
        raise ValueError("Invalid transmission parameters.")
    if not 0 <= p_dc < 1 or not 0 <= e_ali <= 0.5:
        raise ValueError("Invalid detector parameters.")
    la = np.ones(M) if lambda_a is None else np.asarray(lambda_a, dtype=float)
    lb = np.ones(M) if lambda_b is None else np.asarray(lambda_b, dtype=float)
    ta = np.arange(M) * np.pi / M if theta_a is None else np.asarray(theta_a, dtype=float)
    tb = np.arange(M) * np.pi / M if theta_b is None else np.asarray(theta_b, dtype=float)
    if any(array.shape != (M,) for array in (la, lb, ta, tb)):
        raise ValueError("Each basis-setting array must contain exactly M elements.")
    if not np.all(np.isfinite(la)) or not np.all(np.isfinite(lb)):
        raise ValueError("Attenuation values must be finite.")
    if np.any(la < 0) or np.any(la > 1) or np.any(lb < 0) or np.any(lb > 1):
        raise ValueError("Trusted loss transmissivities must lie in [0,1].")
    if not np.all(np.isfinite(ta)) or not np.all(np.isfinite(tb)):
        raise ValueError("Basis phases must be finite.")

    eta_arm = eta_det * 10 ** (-fiber_loss_db_per_km * L_km / 20.0)
    darkness = 0.0 if loss_only else p_dc
    visibility = 1.0 if loss_only else 1.0 - 2.0 * e_ali

    p_plus = np.zeros(4 * M)
    p_minus = np.zeros(4 * M)
    for x in range(M):
        a_mean = eta_arm * n_bar * la[x] / 2.0
        b_mean = eta_arm * n_bar * lb[x] / 2.0
        for bitpair, (da, db) in enumerate(
            ((0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi))
        ):
            label = 4*x+bitpair
            dphi = ta[x] + da - tb[x] - db
            cross = visibility * np.sqrt(a_mean*b_mean) * np.cos(dphi)
            mean_plus = max(0.0, (a_mean+b_mean)/2.0 + cross)
            mean_minus = max(0.0, (a_mean+b_mean)/2.0 - cross)
            p_plus[label] = -np.expm1(-mean_plus-darkness) * np.exp(-mean_minus-darkness)
            p_minus[label] = -np.expm1(-mean_minus-darkness) * np.exp(-mean_plus-darkness)
    p_fail = 1 - p_plus - p_minus
    if np.min(p_fail) < -1e-12:
        raise ValueError("Invalid click probabilities.")
    prob_of = {0: p_plus, 1: p_minus, 2: np.maximum(p_fail, 0.0)}
    p_pass = float(np.mean((p_plus+p_minus)[:4]))
    if p_pass <= 0:
        raise ValueError("No accepted key-basis clicks; choose nonzero brightness or dark counts.")
    e_bit = float((p_minus[0] + p_plus[1] + p_plus[2] + p_minus[3])/(4*p_pass))
    return prob_of, p_pass, e_bit
