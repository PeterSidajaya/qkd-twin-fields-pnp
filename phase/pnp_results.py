"""
Shared PnP SDP output schema (v1).

Every direct-run script prints the same columns and writes identical CSV/JSON
keys to phase/results/<stage>/results.csv and results.json.  Common plots
eph-vs-distance.png and rate-vs-distance.png are generated where possible.

The columns are intentionally explicit about their *denominators*: Stage 1--3
have no monitor conditioning, Stage 4 does not supply P_monitor, while the
current ideal Stage 5A/5B scans often use an externally supplied but merely
illustrative P_monitor=0.99. R_original is not a physical-security claim
unless its P_monitor and experimental announcement data are justified.

Parameter/protocol differences remain in parameters_json and notes. A table
with identical columns does not make data from differing security relaxations
or hardware assumptions directly comparable.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

COLUMNS = (
    "stage", "profile", "L_km", "distance_kind", "M", "n_bar",
    "synthetic_model", "e_ph_upper", "e_bit", "P_pass_cond",
    "P_monitor", "R_acc", "R_original",
    "epsilon_tail", "beta", "status", "phase_bound_scope",
    "tail_kind", "P_monitor_scope", "parameters_json", "notes",
)


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def result_row(
    stage, profile, L_km, *,
    M, n_bar, e_ph_upper, e_bit, P_pass_cond,
    R_acc, status, P_monitor=None, R_original=None,
    epsilon_tail=None, beta=None, tail_kind="",
    phase_bound_scope="unrestricted", P_monitor_scope="none",
    parameters=None, notes="",
):
    """Canonical row, with absent fields represented as null/blank."""
    if P_monitor_scope == "none" and P_monitor is None:
        # Stages 1--3 have no monitor, i.e. their conditioning event is sure.
        P_monitor = 1.0
    if R_original is None and P_monitor_scope == "none":
        R_original = R_acc
    return {
        "stage": str(stage),
        "profile": str(profile),
        "L_km": _finite(L_km),
        "distance_kind": "Alice-Bob total fiber km (two equal arms)",
        "M": int(M),
        "n_bar": _finite(n_bar),
        "synthetic_model": "coherent_reference_2arm_v1",
        "e_ph_upper": _finite(e_ph_upper),
        "e_bit": _finite(e_bit),
        "P_pass_cond": _finite(P_pass_cond),
        "P_monitor": _finite(P_monitor),
        "R_acc": _finite(R_acc),
        "R_original": _finite(R_original),
        "epsilon_tail": _finite(epsilon_tail),
        "beta": _finite(beta),
        "status": str(status),
        "phase_bound_scope": str(phase_bound_scope),
        "tail_kind": str(tail_kind),
        "P_monitor_scope": str(P_monitor_scope),
        "parameters_json": json.dumps(parameters or {}, sort_keys=True),
        "notes": str(notes),
    }


def _pretty(value, width=11):
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:.4g}".rjust(width)
    return str(value).rjust(width)


def print_table(rows):
    """Same table format and column labels for every stage."""
    columns = (
        ("stage", 8), ("profile", 16), ("L_km", 8),
        ("e_ph_upper", 12), ("e_bit", 10), ("P_pass_cond", 12),
        ("R_acc", 12), ("R_original", 12), ("epsilon_tail", 13),
        ("beta", 11), ("status", 17),
    )
    print("  ".join(key.rjust(width) for key,width in columns))
    for row in rows:
        print("  ".join(_pretty(row.get(key),width) for key,width in columns), flush=True)


def write_outputs(rows, *, stage, base_dir=None, plot=True):
    if not rows:
        raise ValueError("No result rows to export.")
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent/"results"/stage
    folder = Path(base_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder/"results.csv"
    json_path = folder/"results.json"
    with csv_path.open("w",newline="",encoding="utf-8") as fh:
        writer=csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w",encoding="utf-8") as fh:
        json.dump(rows,fh,indent=2,allow_nan=False)
        fh.write("\n")

    paths=[csv_path,json_path]
    if plot:
        import matplotlib.pyplot as plt
        groups={}
        for r in rows:
            if r.get("status") not in ("optimal", "optimal_inaccurate"):
                continue
            key=r["profile"]
            groups.setdefault(key,[]).append(r)
        for metric,title,fname in (
            ("e_ph_upper","Phase-error upper bound","eph-vs-distance.png"),
            ("R_acc","Rate per accepted round","rate-vs-distance.png"),
        ):
            if len({r["L_km"] for r in rows}) < 2:
                continue
            fig,ax=plt.subplots(figsize=(7.0,4.6))
            valid=False
            for profile,pr in groups.items():
                rr=sorted((r for r in pr if r[metric] is not None), key=lambda a:a["L_km"])
                # Multiple rows at the same distance, e.g. N_plus or u_cap
                # scans, should NOT silently connect unrelated conditions.
                option_values = {r["parameters_json"] for r in rr}
                if len(option_values) > 1:
                    subsets={}
                    for r in rr:
                        subsets.setdefault(r["parameters_json"],[]).append(r)
                    for i, (parameters, subrows) in enumerate(subsets.items()):
                        sub=sorted(subrows,key=lambda r:r["L_km"])
                        if len(sub) < 2: continue
                        ax.plot([r["L_km"] for r in sub],
                                [r[metric] for r in sub],marker="o",
                                label=f"{profile} ({i+1})")
                        valid=True
                elif len(rr)>=2:
                    ax.plot([r["L_km"] for r in rr],
                            [r[metric] for r in rr],marker="o",label=profile)
                    valid=True
            if valid:
                ax.set_xlabel("Total Alice-Bob fiber distance (km)")
                ax.set_ylabel(metric)
                ax.set_title(f"{stage}: {title} (synthetic benchmark)")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
                fig.tight_layout()
                output=folder/fname
                fig.savefig(output,dpi=200)
                paths.append(output)
            plt.close(fig)
    return tuple(paths)


def report(rows, *, stage, base_dir=None, plot=True):
    print_table(rows)
    paths=write_outputs(rows,stage=stage,base_dir=base_dir,plot=plot)
    print("\nOutput (uniform schema):")
    for p in paths: print(f"  {p}")
    print("CSV/JSON columns and rate denominators are documented in pnp_results.py.")
    return paths
