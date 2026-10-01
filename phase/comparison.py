"""
Trusted-source phase-matching MDI-QKD (fig6.py) against plug-and-play with an
untrusted source (pnp.py), on one axis.

Both are put on total loss in dB so the curves are directly comparable, and
both are run loss-only so the comparison isolates the source assumption
rather than detector imperfections.

The two channel models are not the same and cannot be made the same:

  fig6  Alice and Bob each hold a laser and send mu photons per pulse. The
        node sits midway, so each arm carries sqrt(eta) and the click rate
        is 1 - exp(-2 mu sqrt(eta)). Intensity is theirs to choose, and the
        scan optimises mu_key and mu_test at every loss.

  pnp   The node supplies the light. Alice and Bob monitor the mean energy
        n_bar leaving their labs and modulate phase only. The click rate is
        1 - exp(-eta_pnp n_bar), with eta_pnp the one-way Alice-to-node
        transmittance. The inbound leg does not appear: Alice monitors what
        leaves her lab, so loss on the way in is Eve's problem, not hers.
        They cannot choose the intensity, so the scan optimises n_bar as the
        quantity the energy monitor would be designed around, not as a knob
        turned per round.

The two have the same loss dependence once written this way: identify 2 mu
with n_bar and sqrt(eta) with eta_pnp, and both click rates read
1 - exp(-(arriving intensity)). The total Alice-to-Bob transmittance is the
square of the per-arm one in both, which is the conversion used below.

One convention difference remains and is not reconciled: fig6 splits the
detector efficiency across the two arms, giving sqrt(ETA_DET) per arm, while
pnp applies the whole of it to each. That is a factor sqrt(0.85) = 0.92 in
arriving intensity, small next to everything else here.

Test-state counts are matched: M = 2 gives two test states, M = 3 gives
four, in both files.
"""

import importlib.util
import math
import multiprocessing as mp

import numpy as np

N_PROC = 4
# Fine at low loss, where the plug-and-play curve with Eve's source lives.
# Nothing below 1.41 dB is reachable: pnp's per-arm transmittance is capped
# at ETA_DET = 0.85, and the total is its square.
LOSS_DB = np.concatenate([np.arange(1.5, 6.0, 0.25),
                          np.arange(6.0, 12.0, 1.0),
                          np.arange(12.0, 65.0, 4.0)])

MU_KEY_GRID = np.logspace(-3.0, -0.3, 8)
MU_TEST_GRID = np.logspace(-2.0, 0.3, 6)
NBAR_GRID = np.logspace(-3.5, -0.5, 13)

M_LIST = [2, 3]          # two and four test states


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup():
    """Load both modules and put them in loss-only mode."""
    fig6 = _load("fig6", "fig6.py")
    pnp = _load("pnp", "pnp.py")
    fig6.LOSS_ONLY = True
    fig6.MATCHED_BASES_ONLY = True
    pnp.LOSS_ONLY = True
    pnp.F_EC = 1.0                     # irrelevant at e_bit = 0; kept explicit
    return fig6, pnp


def _fig6_point(task):
    Mv, dB, mk, mt = task
    fig6, _ = _setup()
    eta = 10 ** (-dB / 10.0)
    try:
        e_ph, e_bit, P_pass, status = fig6.solve_eph(
            Mv, [mk] + [mt] * (Mv - 1), eta, 0.0, 0.0,
            solver_opts={"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}})
    except Exception:
        return ("fig6", Mv, dB, 0.0, None)
    R = P_pass * max(0.0, 1.0 - fig6.h2(e_ph) - fig6.h2(e_bit))
    return ("fig6", Mv, dB, R, (mk, mt, e_ph))


def _pnp_point(task):
    Mv, dB, n_bar, source = task
    _, pnp = _setup()
    pnp.SOURCE = source
    # Invert pnp's channel model to hit this total loss.
    #
    # In both protocols the total Alice-to-Bob transmittance is the square of
    # the per-arm one. fig6 scales amplitudes by eta**0.25, so intensity by
    # sqrt(eta) per arm and 2 mu sqrt(eta) arrives in total. pnp has a single
    # fibre factor, eta_pnp = ETA_DET 10^(-XI L / 10), and n_bar eta_pnp
    # arrives, because the inbound leg does not enter: Alice monitors the
    # energy leaving her lab, so whatever Eve loses on the way in she makes
    # up by sending brighter. Identify 2 mu with n_bar and sqrt(eta) with
    # eta_pnp, so eta = eta_pnp^2 and eta_pnp = 10^(-dB/20).
    t = 10 ** (-dB / 20.0)
    if t > pnp.ETA_DET:
        return (f"pnp-{source}", Mv, dB, 0.0, None)
    L = 10.0 * math.log10(pnp.ETA_DET / t) / pnp.XI
    try:
        R, e_ph, e_bit, P_pass, status, p_val = pnp.key_rate(Mv, n_bar, L)
    except Exception:
        return (f"pnp-{source}", Mv, dB, 0.0, None)
    return (f"pnp-{source}", Mv, dB, R, (n_bar, L, e_ph))


def _dispatch(task):
    kind = task[0]
    return _fig6_point(task[1:]) if kind == "fig6" else _pnp_point(task[1:])


def plob(eta):
    return float("inf") if eta >= 1.0 else -math.log2(1.0 - eta)


if __name__ == "__main__":
    tasks = []
    for Mv in M_LIST:
        for dB in LOSS_DB:
            for mk in MU_KEY_GRID:
                for mt in MU_TEST_GRID:
                    tasks.append(("fig6", Mv, dB, mk, mt))
            for source in ("fixed", "eve"):
                for nb in NBAR_GRID:
                    tasks.append(("pnp", Mv, dB, nb, source))

    print(f"{len(tasks)} solves on {N_PROC} processes", flush=True)

    best = {}
    done = 0
    with mp.Pool(processes=N_PROC, maxtasksperchild=25) as pool:
        for label, Mv, dB, R, arg in pool.imap_unordered(_dispatch, tasks,
                                                         chunksize=1):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
            key = (label, Mv, dB)
            if R > best.get(key, (0.0, None))[0]:
                best[key] = (R, arg)

    LABELS = ["fig6", "pnp-fixed", "pnp-eve"]
    for Mv in M_LIST:
        print()
        print(f"M = {Mv}  ({2*(Mv-1)} test states)")
        print(f"{'loss(dB)':>9} {'PLOB':>11} " +
              " ".join(f"{lab:>12}" for lab in LABELS))
        for dB in LOSS_DB:
            eta = 10 ** (-dB / 10.0)
            row = [best.get((lab, Mv, dB), (0.0, None))[0] for lab in LABELS]
            print(f"{dB:9.1f} {plob(eta):11.3e} " +
                  " ".join(f"{v:12.3e}" for v in row))

    # ----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE = {
        "fig6":      dict(ls="-",  label="trusted source"),
        "pnp-fixed": dict(ls="--", label="plug-and-play, trusted source"),
        "pnp-eve":   dict(ls=":",  label="plug-and-play, Eve's source"),
    }
    COL = {2: "tab:blue", 3: "tab:orange"}

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.plot(LOSS_DB, [plob(10 ** (-d / 10.0)) for d in LOSS_DB],
            "r--", lw=1.8, label="PLOB bound")

    for Mv in M_LIST:
        for lab in LABELS:
            xs, ys = [], []
            for dB in LOSS_DB:
                R = best.get((lab, Mv, dB), (0.0, None))[0]
                if R > 0.0:
                    xs.append(dB)
                    ys.append(R)
            if xs:
                st = dict(STYLE[lab])
                st["label"] = f"{st['label']}, {2*(Mv-1)} test states"
                ax.plot(xs, ys, color=COL[Mv], lw=1.7, **st)

    ax.set_yscale("log")
    ax.set_xlabel("total loss (dB) between Alice and Bob")
    ax.set_ylabel("secret key rate (bits/channel use)")
    ax.set_xlim(0, LOSS_DB.max())
    ax.set_ylim(1e-9, 1e0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Trusted source vs plug-and-play, loss only")
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=160)
    print("\nwrote comparison.png")