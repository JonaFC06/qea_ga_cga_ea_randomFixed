"""
experiment.py
=============
Ejecuta el benchmark del artículo:
  - 3 escenarios (n=10,20,30)
  - 2 variantes (U, C)
  - 30 instancias emparejadas (semillas 2026+137r)
  - 5 algoritmos
  - 2 presupuestos (1000, 30000)
  - Óptimo certificado MILP (variante C) o analítico (variante U)
  - Wilcoxon bilateral + Holm (24 comparaciones/presupuesto)
  - Vargha–Delaney Â₁₂
  - Tablas CSV y figuras PNG

Uso:
  python experiment.py --quick      # 3 instancias, 1000 llamadas
  python experiment.py --full       # 30 instancias, 1000 y 30000
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from chromosome_patch import ChromosomeEvaluator, make_cost_matrix, make_teams
from qea_ga_cga_ea_random_fixed import (
    Config, run_qea, run_ga, run_cga, run_one_plus_one, run_rs
)


ALGOS = [("QEA", run_qea), ("GA", run_ga), ("cGA", run_cga),
         ("(1+1)-EA", run_one_plus_one), ("RS", run_rs)]


def run_instance(n: int, seed: int, variant: str, budget: int) -> dict:
    """Corre los 5 algoritmos sobre una instancia emparejada."""
    cost = make_cost_matrix(n, seed)
    constraints = make_teams(n)
    ev = ChromosomeEvaluator(n, cost, constraints, variant=variant)

    # Óptimo de referencia
    if variant == "U":
        F_star = ev.optimal_U()
        feasible_star = True
    else:
        milp = ev.solve_milp()
        F_star = milp["F_star"]
        feasible_star = milp["feasible"]

    out = {"n": n, "seed": seed, "variant": variant, "budget": budget,
           "F_star": F_star, "feasible_star": feasible_star,
           "algorithms": {}}

    for name, runner in ALGOS:
        cfg = Config(n_agents=n, variant=variant, budget=budget,
                     seed=seed, constraints=constraints)
        r = runner(cfg, ev)
        gap = 100.0 * (r.best_f - F_star) / F_star if F_star > 0 else 0.0
        feasible = ev.is_feasible(r.best_x)
        out["algorithms"][name] = {
            "best_f": r.best_f, "gap": gap, "feasible": feasible,
            "n_evals": r.n_evals, "time_s": r.time_s,
            "best_history": r.best_history, "eval_counts": r.eval_counts,
            "diversity": r.diversity,
        }
    return out


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Â₁₂ = P(X<Y) + 0.5 P(X=Y)."""
    x = np.asarray(x); y = np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def holm(pvals: list) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def statistical_analysis(results: list, budget: int) -> list:
    """
    24 comparaciones por presupuesto: 6 bloques (3 n × 2 variantes)
    × 4 baselines (GA, cGA, (1+1)-EA, RS).
    """
    rows = []
    for n in (10, 20, 30):
        for variant in ("U", "C"):
            block = [r for r in results
                     if r["n"] == n and r["variant"] == variant
                     and r["budget"] == budget]
            qea_gaps = np.array([b["algorithms"]["QEA"]["gap"] for b in block])
            for baseline in ("GA", "cGA", "(1+1)-EA", "RS"):
                base_gaps = np.array([b["algorithms"][baseline]["gap"]
                                       for b in block])
                try:
                    _, p = stats.wilcoxon(qea_gaps, base_gaps,
                                          alternative="two-sided")
                except ValueError:
                    p = 1.0
                a12 = vargha_delaney_a12(qea_gaps, base_gaps)
                rows.append({
                    "n": n, "variant": variant, "budget": budget,
                    "baseline": baseline, "p_raw": p, "a12": a12,
                })
    # Holm sobre las 24 comparaciones del presupuesto
    pvals = [r["p_raw"] for r in rows]
    adj = holm(pvals)
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
    return rows


def plot_anytime(results: list, out_path: str):
    """6 paneles: 3 n × 2 variantes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = {"QEA": "#00d4aa", "GA": "#4fc3f7", "cGA": "#ffb74d",
              "(1+1)-EA": "#e57373", "RS": "#9575cd"}
    for vi, variant in enumerate(("U", "C")):
        for ni, n in enumerate((10, 20, 30)):
            ax = axes[vi, ni]
            block = [r for r in results
                     if r["n"] == n and r["variant"] == variant]
            for name, _ in ALGOS:
                # Mediana de gaps vs llamadas (interpolar a rejilla común)
                grid = np.unique(np.concatenate(
                    [np.array(b["algorithms"][name]["eval_counts"]) for b in block]
                ))
                curves = []
                for b in block:
                    ec = np.array(b["algorithms"][name]["eval_counts"])
                    bh = np.array(b["algorithms"][name]["best_history"])
                    fstar = b["F_star"]
                    gap = 100.0 * (bh - fstar) / fstar
                    # Interpolar
                    curves.append(np.interp(grid, ec, gap,
                                            left=gap[0], right=gap[-1]))
                curves = np.array(curves)
                med = np.median(curves, axis=0)
                q1 = np.percentile(curves, 25, axis=0)
                q3 = np.percentile(curves, 75, axis=0)
                ax.plot(grid, med, color=colors[name], label=name, lw=2)
                ax.fill_between(grid, q1, q3, color=colors[name], alpha=0.15)
            ax.axvline(1000, color="k", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("Llamadas al evaluador")
            ax.set_ylabel("Gap (%)")
            ax.set_title(f"n={n} | Variante {variant}")
            ax.grid(True, alpha=0.3)
            if vi == 0 and ni == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Figura guardada: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="3 instancias, solo 1000 llamadas")
    ap.add_argument("--full", action="store_true",
                    help="30 instancias, 1000 y 30000 llamadas")
    args = ap.parse_args()

    n_instances = 3 if args.quick else 30
    budgets = [1000] if args.quick else [1000, 30000]
    seeds = [2026 + 137 * r for r in range(n_instances)]

    os.makedirs("resultados", exist_ok=True)
    all_results = []

    for n in (10, 20, 30):
        for variant in ("U", "C"):
            for budget in budgets:
                for r, seed in enumerate(seeds):
                    print(f"[n={n} {variant} budget={budget} r={r} seed={seed}]")
                    res = run_instance(n, seed, variant, budget)
                    all_results.append(res)
                    # Guardar CSV por instancia
                    with open(f"resultados/raw_n{n}_{variant}_b{budget}.csv",
                              "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["algo", "best_f", "gap", "feasible",
                                    "n_evals", "time_s"])
                        for name, _ in ALGOS:
                            a = res["algorithms"][name]
                            w.writerow([name, a["best_f"], a["gap"],
                                        a["feasible"], a["n_evals"], a["time_s"]])

    # Tabla agregada
    with open("resultados/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "variant", "budget", "algo", "gap_mean", "gap_std",
                    "opt_hits", "feasible_runs", "time_mean"])
        for n in (10, 20, 30):
            for variant in ("U", "C"):
                for budget in budgets:
                    block = [r for r in all_results
                             if r["n"] == n and r["variant"] == variant
                             and r["budget"] == budget]
                    for name, _ in ALGOS:
                        gaps = np.array([b["algorithms"][name]["gap"]
                                          for b in block])
                        times = np.array([b["algorithms"][name]["time_s"]
                                           for b in block])
                        feas = sum(b["algorithms"][name]["feasible"]
                                    for b in block)
                        opt = int(np.sum(gaps < 1e-6))
                        w.writerow([n, variant, budget, name,
                                    gaps.mean(), gaps.std(), opt, feas,
                                    times.mean()])

    # Estadística
    stats_rows = []
    for budget in budgets:
        stats_rows.extend(statistical_analysis(all_results, budget))
    with open("resultados/statistics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "variant", "budget",
                                           "baseline", "p_raw", "p_holm", "a12"])
        w.writeheader()
        for r in stats_rows:
            w.writerow(r)

    # Figura anytime
    plot_anytime(all_results, "resultados/anytime.png")
    print("Listo.")


if __name__ == "__main__":
    main()