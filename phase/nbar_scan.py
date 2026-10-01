"""
Two-dimensional scan of the plug-and-play continuum SDP: mean photon number
against distance.

Why this exists. n_bar was pinned at 0.01 in every run so far and never
optimised. It is not a free parameter of the protocol the way an intensity
setting would be - Eve supplies the light - but it IS the quantity Alice and
Bob design their energy monitor around, so it is theirs to choose once. At
L = 0 the optimum is near 0.1 and worth roughly six times the rate at 0.01:

    n_bar    e_ph    P_pass      R          g2
    0.003   0.2021  2.55e-3   3.68e-4    129.2
    0.01    0.2026  8.46e-3   1.22e-3     35.8
    0.03    0.2093  2.52e-2   3.29e-3     11.8
    0.10    0.2348  8.15e-2   6.89e-3      3.9
    0.15    0.2531  1.20e-1   6.53e-3      2.8
    0.30    0.3066  2.25e-1   0            1.7

P_pass grows linearly in n_bar while e_ph creeps up, so the rate follows
P_pass until e_ph reaches the threshold (0.292 at e_bit = 0.015, f_ec =
1.15) somewhere between 0.15 and 0.3.

But brighter costs range. At n_bar = 0.1, L = 2 km gives e_ph = 0.3123 and
no key, where n_bar = 0.01 at the same distance still gave 0.2823 and a
positive rate. So the optimum falls with distance and has to be scanned per
point, which is what this does.

Run it as-is; every knob is at the top. With MOSEK expect seconds per solve;
CLARABEL takes a couple of minutes and reports optimal_inaccurate throughout.

Reads pnp-continuum.py from the same directory and does not modify it.
"""

import importlib.util
import math
import multiprocessing as mp
import time

import numpy as np

# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------

MODULE_PATH = "pnp-continuum.py"

# Fine below 3 km: the cutoff sits there and that is the whole question.
# Nothing past ~5 km has produced a key at any n_bar tried so far.
L_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
NBAR_GRID = np.logspace(math.log10(0.003), math.log10(0.3), 10)

# Workers, not cores. Each holds a canonicalised SDP; memory is the binding
# resource, and MOSEK is internally multithreaded so each worker is pinned to
# one thread below to stop them oversubscribing the machine.
N_PROC = 4

# Overrides for the module's own globals. None means leave as the module has
# it. U_MAX = 20 is near the conditioning ceiling: 50 returns points that
# violate the energy constraint, so raise it only with the grid check in
# pnp-continuum.py passing.
M_OVERRIDE = None          # number of bases; module default 4
T2_OVERRIDE = None         # attenuation; module default 1e-6
UMAX_OVERRIDE = None       # u-grid range; module default 20.0

OUT_CSV = "nbar-scan.csv"
OUT_PNG = "nbar-scan.png"

# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

_MOD = None


def _load():
    """Load pnp-continuum.py once per worker process, not once per task."""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location("pnpc", MODULE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if M_OVERRIDE is not None:
            mod.M = M_OVERRIDE
        if T2_OVERRIDE is not None:
            mod.T2 = T2_OVERRIDE
        if UMAX_OVERRIDE is not None:
            mod.U_MAX = UMAX_OVERRIDE
        _MOD = mod
    return _MOD


def _one(task):
    """
    One (n_bar, L) point.

    Returns the status verbatim. solve_eph returns e_ph = 0.5 when the solver
    fails, which is indistinguishable from a genuine e_ph = 0.5, so the
    caller has to look at the status rather than the number.
    """
    nb, L = task
    m = _load()
    t0 = time.time()
    try:
        e_ph, e_bit, P_pass, status, nu = m.solve_eph(
            m.M, nb, L, t2=m.T2,
            solver_opts={"mosek_params": {"MSK_IPAR_NUM_THREADS": 1}})
    except Exception as exc:
        return dict(n_bar=nb, L=L, ok=False,
                    status=f"exception: {type(exc).__name__}",
                    e_ph=np.nan, e_bit=np.nan, P_pass=np.nan, R=0.0,
                    EA=np.nan, g2=np.nan, eps_nu=np.nan, feasible=False,
                    secs=time.time() - t0)

    R = P_pass * max(0.0, 1.0 - m.h2(e_ph) - m.F_EC * m.h2(e_bit))

    if nu is None:
        EA = g2 = eps_nu = np.nan
        feasible = False
    else:
        EA, g2 = m.diagnostics(nu)
        eps_nu = 1.0 - float(nu.sum())
        # Standing check. The solver returning a point that breaks its own
        # constraints is how the outside-grid-mass bug showed up: eps_nu went
        # negative and E[u_A] sat above its cap while the status still read
        # optimal. Tolerance is loose enough not to fire on rounding.
        feasible = (EA <= nb / 2 * (1 + 1e-6) + 1e-9) and (eps_nu >= -1e-9)

    return dict(n_bar=nb, L=L, ok=status in ("optimal", "optimal_inaccurate"),
                status=status, e_ph=e_ph, e_bit=e_bit, P_pass=P_pass, R=R,
                EA=EA, g2=g2, eps_nu=eps_nu, feasible=feasible,
                secs=time.time() - t0)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def make_plot(R, rows_by_key, path):
    """
    Two panels, no dual axis anywhere.

    (a) the surface R(L, n_bar) as a heatmap. Twelve n_bar values would be
        twelve line colours, which is past any categorical palette; magnitude
        over two dimensions is a heatmap's job. Sequential means one hue,
        light to dark - never a rainbow - so Blues, not viridis. Cells with
        no key are masked to a flat grey rather than plotted as a very small
        number, which on a log scale they are not.

    (b) the envelope, best R over n_bar at each distance. One series, so the
        title names it and no legend box is needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    INK = "#3a3a3a"
    MUTED = "#8a8a8a"
    GRID = "#d8d8d8"
    ACCENT = "#1f5fa8"

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    pos = R[R > 0]
    masked = np.ma.masked_where(R <= 0, R)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eceff1")          # no key, not "almost no key"

    if pos.size:
        im = ax.pcolormesh(L_GRID, NBAR_GRID, masked, cmap=cmap,
                           norm=LogNorm(vmin=pos.min(), vmax=pos.max()),
                           shading="nearest", edgecolors="white", linewidth=0.5)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("secret key rate (bits/pulse)", color=INK, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)

    # Where the optimum sits, overlaid on the surface it belongs to.
    opt_L, opt_nb = [], []
    for j, L in enumerate(L_GRID):
        col = R[:, j]
        if col.max() > 0:
            opt_L.append(L)
            opt_nb.append(NBAR_GRID[int(np.argmax(col))])
    if opt_L:
        # Named in the title rather than annotated in the axes: the optimum
        # runs through the darkest cells, where any ink the accent colour
        # would use is unreadable.
        ax.plot(opt_L, opt_nb, "o-", color=ACCENT, lw=2.0, ms=7,
                mfc="white", mew=2.0, zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("fibre distance (km)", color=INK)
    ax.set_ylabel("$\\bar{n}$", color=INK)
    ax.set_title("(a) rate over the scan; markers = best $\\bar{n}$; "
                 "grey = no key", fontsize=10, color=INK, loc="left")

    env_L = [L for j, L in enumerate(L_GRID) if R[:, j].max() > 0]
    env_R = [R[:, j].max() for j, L in enumerate(L_GRID) if R[:, j].max() > 0]
    if env_L:
        bx.plot(env_L, env_R, "-o", color=ACCENT, lw=2.0, ms=7,
                mfc="white", mew=2.0)
    bx.set_yscale("log")
    bx.set_xlim(ax.get_xlim())
    bx.set_xlabel("fibre distance (km)", color=INK)
    bx.set_ylabel("secret key rate (bits/pulse)", color=INK)
    bx.set_title("(b) rate at the best $\\bar{n}$ for each distance",
                 fontsize=10, color=INK, loc="left")

    for a in (ax, bx):
        a.grid(True, which="both", ls=":", color=GRID, alpha=0.7)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    probe = _load()
    M_used = probe.M
    print(f"M = {M_used}, t^2 = {probe.T2:g}, U_MAX = {probe.U_MAX:g}, "
          f"{'loss only' if probe.LOSS_ONLY else 'with noise'}")
    thr_lo = 0.0
    print(f"{len(NBAR_GRID)} n_bar x {len(L_GRID)} distances = "
          f"{len(NBAR_GRID)*len(L_GRID)} solves on {N_PROC} processes",
          flush=True)

    tasks = [(nb, L) for nb in NBAR_GRID for L in L_GRID]

    results = {}
    t0 = time.time()
    done = 0
    with mp.Pool(processes=N_PROC, initializer=_load,
                 maxtasksperchild=25) as pool:
        for res in pool.imap_unordered(_one, tasks, chunksize=1):
            done += 1
            results[(res["n_bar"], res["L"])] = res
            if done % 10 == 0 or done == len(tasks):
                el = time.time() - t0
                rem = (len(tasks) - done) * el / done
                print(f"  {done}/{len(tasks)}  elapsed {el/60:.1f} min  "
                      f"remaining ~{rem/60:.1f} min", flush=True)

    R = np.zeros((len(NBAR_GRID), len(L_GRID)))
    for i, nb in enumerate(NBAR_GRID):
        for j, L in enumerate(L_GRID):
            R[i, j] = results[(nb, L)]["R"]

    # --- full grid ---
    print("\nSecret key rate, rows n_bar, columns distance in km. "
          "* marks a point the solver did not return cleanly, "
          "! one that breaks its own constraints.")
    print(f"{'n_bar':>8} " + " ".join(f"{L:>11.1f}" for L in L_GRID))
    for i, nb in enumerate(NBAR_GRID):
        cells = []
        for j, L in enumerate(L_GRID):
            r = results[(nb, L)]
            flag = "!" if not r["feasible"] else ("*" if r["status"] != "optimal" else " ")
            cells.append(f"{R[i, j]:10.3e}{flag}")
        print(f"{nb:8.4f} " + " ".join(cells))

    # --- envelope ---
    print("\nBest n_bar at each distance:")
    print(f"{'L(km)':>7} {'n_bar*':>9} {'e_ph':>8} {'e_bit':>8} "
          f"{'P_pass':>11} {'R':>11} {'E[u_A]':>10} {'g2':>9} "
          f"{'eps_nu':>10}  status")
    for j, L in enumerate(L_GRID):
        col = R[:, j]
        if col.max() <= 0:
            print(f"{L:7.1f}   no key at any n_bar on this grid")
            continue
        i = int(np.argmax(col))
        r = results[(NBAR_GRID[i], L)]
        print(f"{L:7.1f} {r['n_bar']:9.4f} {r['e_ph']:8.4f} "
              f"{r['e_bit']:8.4f} {r['P_pass']:11.3e} {r['R']:11.3e} "
              f"{r['EA']:10.3e} {r['g2']:9.2f} {r['eps_nu']:10.2e}  "
              f"{r['status']}{'' if r['feasible'] else '  INFEASIBLE'}")

    bad = [r for r in results.values() if not r["feasible"]]
    if bad:
        print(f"\n{len(bad)} of {len(tasks)} points break their own "
              f"constraints. Those rows are not bounds. Rerun with MOSEK, or "
              f"lower U_MAX until they clear.")

    # --- csv ---
    cols = ["n_bar", "L", "e_ph", "e_bit", "P_pass", "R", "EA", "g2",
            "eps_nu", "feasible", "status", "secs"]
    with open(OUT_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for nb in NBAR_GRID:
            for L in L_GRID:
                r = results[(nb, L)]
                f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {OUT_CSV}")

    make_plot(R, results, OUT_PNG)