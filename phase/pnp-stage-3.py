"""
Plug-and-play phase-error SDP in the strong-attenuation regime.

Companion to continuum-attenuation.tex, which derives everything here.

The Fock-grid version (pnp-attenuation.py) cannot reach the attenuation a
real device uses. Alice monitors the energy leaving her lab, so the input
energy is n_bar / (2 t^2) and the grid has to grow with it: t^2 = 0.1 already
needs N_IN ~ 60, and 60 to 100 dB of attenuation would need ~1e6 points.

The way out is that the SDP never needs the Fock distribution, only the
overlaps, and those depend on it through a generating function evaluated at
one point per state pair. Rescaling to u = t^2 m, with u the photons the mode
actually delivers, and taking t^2 -> 0 at fixed u gives

    Lambda_{s1,s2}(nu) = integral dnu(uA, uB)
        exp[ -uA (1 - exp(i dphi_A)) - uB (1 - exp(i dphi_B)) ]

which is linear in nu and free of t^2. The grid range is now set by n_bar
rather than by 1/t^2.

The kernel is the Poisson characteristic function, so the statement is that
heavily attenuated light has mixed Poisson statistics with mixing measure nu.
Coherent is nu = delta at n_bar/2. That is the precise version of "attenuation
makes it look coherent", and it also shows why it does not save the protocol:

    g2 = E[u^2] / E[u]^2 = 1 + Var(u) / E[u]^2

so the adversary's bunching is the spread of nu, and a first-moment
constraint does not bound a variance. nu = (1-eps) delta_0 + eps delta_{m}
with eps m = n_bar/2 gives g2 = 1/eps for any eps.

Two things make this a bound rather than an approximation:

  - finite t^2. The limit kernel differs from the exact z^(u/t^2) by at most
    4 n_bar t^2 exp(4 u_max t^2) integrated against nu, added to every pair's
    slack as DELTA_T below. It is 4e-4 at t^2 = 1e-2 and 4e-8 at 1e-6, so it
    is negligible exactly where the Fock grid was impossible.

  - the lattice. At finite t^2 the measure lives on t^2 Z^2, not on all of
    [0, inf)^2. Dropping that enlarges the feasible set, which is the safe
    direction.
"""

import numpy as np
import cvxpy as cp
import math

# --------------------------------------------------------------------------

M = 4
F_EC = 1.15
ETA_DET = 0.85
XI = 0.2
E_ALI = 0.015
P_DC = 5e-8
LOSS_ONLY = False

# Physical attenuation. Enters ONLY through the finite-t^2 correction below;
# the rest of the problem does not see it. Set it to what the device does.
T2 = 1e-8

# u grid. Logarithmic, since nu is expected to sit near n_bar/2 while the
# attack puts a little mass far out. U_MAX sets the tail slack, n_bar/U_MAX,
# and costs only grid points now.
U_MAX = 20.0
N_U = 60

SLACK_FLOOR = 1e-14

Z_P, Z_M, Z_F = 0, 1, 2
enc_phases = [(0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)]


def u_grid():
    """Per-mode grid: exact zero, then logarithmic out to U_MAX."""
    return np.concatenate([[0.0], np.logspace(-4, math.log10(U_MAX), N_U)])


def state_phases(M):
    out = []
    for m in range(M):
        phi = np.pi * m / M
        for dA, dB in enc_phases:
            out.append((phi + dA, phi + dB))
    return out


def h2(e):
    return 0.0 if e <= 0 or e >= 1 else -e * math.log2(e) - (1 - e) * math.log2(1 - e)


def channel(n_bar, L):
    eta = ETA_DET * 10 ** (-XI * L / 10.0)
    p_dc = 0.0 if LOSS_ONLY else P_DC
    e_ali = 0.0 if LOSS_ONLY else E_ALI
    P_sig = 1.0 - np.exp(-eta * n_bar)
    P_pass = P_sig + p_dc
    return P_pass, (e_ali * P_sig + 0.5 * p_dc) / P_pass


# --------------------------------------------------------------------------

def solve_eph(M, n_bar, L, t2=T2, solver=None, solver_opts=None):
    phases = state_phases(M)
    NS = 4 * M
    P_pass, e_obs = channel(n_bar, L)

    P_plus = np.zeros(NS)
    P_minus = np.zeros(NS)
    for s in range(NS):
        hit, miss = (1 - e_obs) * P_pass, e_obs * P_pass
        P_plus[s], P_minus[s] = (hit, miss) if (s % 4) in (0, 3) else (miss, hit)
    P_fail = 1.0 - P_plus - P_minus
    prob_of = {Z_P: P_plus, Z_M: P_minus, Z_F: P_fail}

    # Zero-norm reduction: a vector with zero probability puts a zero on the
    # Gram diagonal, forcing its whole row to vanish by positive
    # semidefiniteness, which kills strict feasibility.
    active = [(z, s) for z in (Z_P, Z_M, Z_F) for s in range(NS)
              if prob_of[z][s] > 1e-12]
    pos = {zs: k for k, zs in enumerate(active)}
    DIM = len(active)

    G = cp.Variable((DIM, DIM), hermitian=True)
    cons = [G >> 0]

    def ent(z1, s1, z2, s2):
        k1, k2 = pos.get((z1, s1)), pos.get((z2, s2))
        return 0 if (k1 is None or k2 is None) else G[k1, k2]

    # --- the mixing measure ---
    ug = u_grid()
    UA, UB = np.meshgrid(ug, ug, indexing="ij")
    UA, UB = UA.ravel(), UB.ravel()
    NU = UA.size

    nu = cp.Variable(NU, nonneg=True)
    cons.append(cp.sum(nu) <= 1)
    eps_nu = 1 - cp.sum(nu)

    # The unassigned mass is the mass outside the grid, where
    # max(u_A, u_B) > U_MAX and hence u_A + u_B > U_MAX. It must be CHARGED
    # in the energy budget, not merely bounded by a separate Markov
    # inequality: the two are equivalent only when the in-grid energy is
    # zero, and here the adversary otherwise saturates the in-grid energy AND
    # the Markov cap at the same time, which the physics forbids.
    #
    # Charge it against the SUM, never per arm: mass outside the grid can sit
    # entirely in one mode, so u_A > U_MAX does not follow from being
    # outside. Same structure as the per-arm energy error in the Fock code.
    cons.append((UA + UB) @ nu + U_MAX * eps_nu <= n_bar)

    # Alice and Bob monitor the mean photon number leaving their labs, which
    # in these variables is E[u_A] directly. No t^2 appears. Per arm the
    # outside-grid mass is dropped rather than charged, which is sound
    # because it contributes non-negatively.
    cons.append(UA @ nu <= n_bar / 2)
    cons.append(UB @ nu <= n_bar / 2)

    # --- overlaps ---
    pairs = [(s1, s2) for s1 in range(NS) for s2 in range(s1 + 1, NS)]
    K = np.zeros((len(pairs), NU), dtype=complex)
    for i, (s1, s2) in enumerate(pairs):
        dA = phases[s2][0] - phases[s1][0]
        dB = phases[s2][1] - phases[s1][1]
        K[i, :] = np.exp(-UA * (1 - np.exp(1j * dA))
                         - UB * (1 - np.exp(1j * dB)))

    # Slack, three sources, all additive:
    #   finite t^2, Eq. (13) of the companion note
    #   tail beyond U_MAX, by Markov, kernel modulus at most 1
    #   unassigned mass eps_nu, kernel modulus at most 1
    # Slack, two sources:
    #   finite t^2, Eq. (13) of the companion note
    #   mass outside the grid, whose kernel modulus is at most 1
    delta_t = 4.0 * n_bar * t2 * math.exp(min(4.0 * U_MAX * t2, 50.0))
    slack = delta_t + eps_nu + SLACK_FLOOR

    terms = []
    for z in (Z_P, Z_M, Z_F):
        rows, cols, mask = [], [], []
        for (s1, s2) in pairs:
            k1, k2 = pos.get((z, s1)), pos.get((z, s2))
            if k1 is None or k2 is None:
                rows.append(0); cols.append(0); mask.append(0.0)
            else:
                rows.append(k1); cols.append(k2); mask.append(1.0)
        terms.append(cp.multiply(np.array(mask), G[rows, cols]))

    diff = sum(terms) - K @ nu
    resid = cp.vstack([cp.real(diff), cp.imag(diff)]).T
    cons.append(cp.norm(resid, 2, axis=1) <= slack * np.ones(len(pairs)))

    cons.append(cp.real(cp.diag(G)) ==
                np.array([prob_of[z][s] for (z, s) in active]))

    # --- objective: Eq. A7 on the key basis (m = 0) ---
    bracket = (ent(Z_P, 0, Z_P, 3) - ent(Z_M, 0, Z_M, 3)
               - ent(Z_P, 1, Z_P, 2) + ent(Z_M, 1, Z_M, 2))
    obj = cp.real(bracket)
    cons.append(obj <= 0.0)

    prob = cp.Problem(cp.Maximize(obj), cons)
    if solver is None:
        try:
            import mosek
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
        return 0.5, e_obs, P_pass, prob.status, None
    e_ph = float(np.clip(0.5 + prob.value / (4 * P_pass), 0.0, 0.5))
    return e_ph, e_obs, P_pass, prob.status, np.asarray(nu.value)


def diagnostics(nu_val):
    """Mean and g2 of the delivered light, from the mixing measure."""
    ug = u_grid()
    UA, _ = np.meshgrid(ug, ug, indexing="ij")
    UA = UA.ravel()
    EA = float(UA @ nu_val)
    E2 = float((UA ** 2) @ nu_val)
    return EA, (E2 / EA ** 2 if EA > 0 else float("nan"))


# --------------------------------------------------------------------------

if __name__ == "__main__":
    n_bar = 0.01
    print(f"M = {M}, n_bar = {n_bar}, t^2 = {T2:g}, "
          f"U_MAX = {U_MAX:g}, {(N_U+1)**2} grid points")
    print(f"slack: finite-t^2 {4*n_bar*T2:.2e} + outside-grid mass "
          f"<= {n_bar/U_MAX:.2e}")
    print("Fock-grid reference at t^2 = 0.1, L = 0: e_ph = 0.2094\n")
    print(f"{'L(km)':>7} {'e_ph':>8} {'e_bit':>8} {'P_pass':>11} "
          f"{'R':>11} {'E[u_A]':>10} {'g2':>9}  status")

    for L in (0.0, 2.0, 5.0, 10.0):
        e_ph, e_bit, P_pass, st, nu_val = solve_eph(M, n_bar, L)
        R = P_pass * max(0.0, 1 - h2(e_ph) - F_EC * h2(e_bit))
        if nu_val is None:
            print(f"{L:7.1f} {'-':>8} {'-':>8} {P_pass:11.3e} "
                  f"{'-':>11} {'-':>10} {'-':>9}  {st}")
            continue
        EA, g2 = diagnostics(nu_val)
        print(f"{L:7.1f} {e_ph:8.4f} {e_bit:8.4f} {P_pass:11.3e} "
              f"{R:11.3e} {EA:10.3e} {g2:9.2f}  {st}", flush=True)

    # Grid convergence. U_MAX enters only through the tail slack n_bar/U_MAX,
    # so e_ph should fall as U_MAX grows and flatten once the tail stops
    # binding. If it does not flatten, the grid is still the limiting factor.
    print("\nGrid check (e_ph should fall then flatten):")
    import sys
    this = sys.modules[__name__]
    for umax in (10.0, 15.0, 20.0, 30.0):
        this.U_MAX = umax
        e_ph, _, _, st, nv = solve_eph(M, n_bar, 0.0)
        print(f"  U_MAX={umax:8.0e}  e_ph={e_ph:.4f}  "
              f"tail slack={n_bar/umax:.2e}  {st}", flush=True)
    this.U_MAX = 20.0