"""
qea_ga_cga_ea_random_fixed.py
=============================
Implementación corregida de QEA, GA, cGA, (1+1)-EA y RS
que reproduce el protocolo del artículo:

  - Presupuesto en llamadas al evaluador (1000 y 30000)
  - Variantes U (calibración) y C (degree-constrained)
  - Rotación esquema I con sgn(αβ) correcto (Tabla I)
  - θ(g) = max(θ₀ exp(−λg), θ_min)
  - Golden genes anclados, nunca rotan
  - Diversidad D(q) = (1/n_g) Σ H(β²), golden contribuyen 0
  - Cruce uniforme en GA
  - K = κ sqrt(|L|) ln|L| en cGA
  - Hijo idéntico hereda fitness en (1+1)-EA
  - 30 instancias emparejadas con semillas 2026+137r
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional

from chromosome_patch import ChromosomeEvaluator, make_cost_matrix, make_teams


# ═══════════════════════════════════════════════════════════════════════════
# Configuración
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    n_agents: int = 30
    variant: str = "C"            # "U" | "C"
    budget: int = 30000           # llamadas al evaluador
    seed: int = 0
    constraints: list = field(default_factory=list)
    # QEA
    theta0: float = 0.1 * np.pi   # Tabla II: configuración seleccionada
    theta_min: float = 0.001 * np.pi
    lam: float = 1e-4             # Tabla II
    scheme: str = "I"             # Tabla II
    use_qiskit: bool = False
    # GA
    pop_size: int = 30
    crossover_rate: float = 0.9
    tournament_k: int = 3
    mutation_rate: Optional[float] = None  # None → 1/|L|
    # cGA
    cga_kappa: float = 0.5
    # (1+1)-EA
    ea_chi: float = 2.0


@dataclass
class Result:
    best_x: np.ndarray
    best_f: float
    best_history: List[float] = field(default_factory=list)
    eval_counts: List[int] = field(default_factory=list)
    diversity: List[float] = field(default_factory=list)
    n_evals: int = 0
    time_s: float = 0.0
    label: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# QEA corregido
# ═══════════════════════════════════════════════════════════════════════════

class QuantumChromosome:
    def __init__(self, n_genes: int, locked: np.ndarray):
        self.n = n_genes
        self.alpha = np.full(n_genes, 1.0 / np.sqrt(2))
        self.beta = np.full(n_genes, 1.0 / np.sqrt(2))
        self.locked = np.asarray(locked, dtype=int)
        # Golden anclados: α=0, β=1 → P(1)=1
        self.alpha[self.locked] = 0.0
        self.beta[self.locked] = 1.0

    def prob_one(self) -> np.ndarray:
        return self.beta ** 2

    def observe(self, rng: np.random.Generator) -> np.ndarray:
        r = rng.random(self.n)
        return (r >= self.alpha ** 2).astype(int)

    def rotate(self, x: np.ndarray, b: np.ndarray,
               f_worse: bool, theta: float, scheme: str):
        """
        Tabla I del PDF. Para cada gen libre:
          (xℓ, bℓ, w)   Scheme I       Scheme II    Scheme III
          (1, 0, sí)    −sgn(αβ)θ      −θ           −θ
          (0, 1, sí)    +sgn(αβ)θ      +θ           +θ
          (1, 0, no)    −θ/2           0            −θ
          (0, 1, no)    +θ/2           0            +θ
          (xℓ = bℓ)      0              0             0
        """
        for ell in range(self.n):
            if ell in self.locked:
                continue
            xl, bl = int(x[ell]), int(b[ell])
            if xl == bl:
                continue
            a, be = self.alpha[ell], self.beta[ell]
            sgn = np.sign(a * be)
            if sgn == 0:
                sgn = 1.0  # evitar 0; en la práctica αβ≠0 para genes libres

            dtheta = 0.0
            if scheme == "I":
                if xl == 1 and bl == 0 and f_worse:
                    dtheta = -sgn * theta
                elif xl == 0 and bl == 1 and f_worse:
                    dtheta = +sgn * theta
                elif xl == 1 and bl == 0:      # no worse
                    dtheta = -theta / 2.0
                elif xl == 0 and bl == 1:      # no worse
                    dtheta = +theta / 2.0
            elif scheme == "II":
                if f_worse:
                    dtheta = +theta if bl == 1 else -theta
                # no worse → 0
            elif scheme == "III":
                dtheta = +theta if bl == 1 else -theta

            if abs(dtheta) < 1e-15:
                continue
            c, s = np.cos(dtheta), np.sin(dtheta)
            na = c * a - s * be
            nb = s * a + c * be
            norm = np.hypot(na, nb)
            self.alpha[ell] = na / norm
            self.beta[ell] = nb / norm

    def diversity(self, n_genes_total: int) -> float:
        """D(q) = (1/n_g) Σ H(β²), golden contribuyen 0 (Eq. 11)."""
        p = self.beta ** 2
        # Solo genes libres contribuyen
        free = np.setdiff1d(np.arange(self.n), self.locked)
        p_free = np.clip(p[free], 1e-12, 1 - 1e-12)
        H = -p_free * np.log2(p_free) - (1 - p_free) * np.log2(1 - p_free)
        return float(np.sum(H) / n_genes_total)


def run_qea(cfg: Config, ev: ChromosomeEvaluator) -> Result:
    rng = np.random.default_rng(cfg.seed)
    locked = ev.protected_idx
    qc = QuantumChromosome(ev.n_genes, locked)

    # Inicialización (1 llamada)
    x = ev.enforce_golden(qc.observe(rng))
    f = ev.evaluate(x)
    best_x, best_f = x.copy(), f
    n_evals = 1

    b_hist = [best_f]
    e_hist = [n_evals]
    d_hist = [qc.diversity(ev.n_genes)]

    t0 = time.time()
    while n_evals < cfg.budget:
        g = n_evals - 1
        theta = max(cfg.theta0 * np.exp(-cfg.lam * g), cfg.theta_min)
        x = ev.enforce_golden(qc.observe(rng))
        f = ev.evaluate(x)
        n_evals += 1
        if f < best_f:
            best_f, best_x = f, x.copy()
        # Rotar: f_worse = f > best_f (después de actualizar best)
        qc.rotate(x, best_x, f > best_f, theta, cfg.scheme)
        b_hist.append(best_f)
        e_hist.append(n_evals)
        d_hist.append(qc.diversity(ev.n_genes))

    return Result(best_x, best_f, b_hist, e_hist, d_hist,
                  n_evals, time.time() - t0, "QEA")


# ═══════════════════════════════════════════════════════════════════════════
# GA corregido (cruce uniforme, elitismo, mutación 1/|L|)
# ═══════════════════════════════════════════════════════════════════════════

def run_ga(cfg: Config, ev: ChromosomeEvaluator) -> Result:
    rng = np.random.default_rng(cfg.seed + 1000)
    free = ev.free_idx
    L = ev.L
    mut_rate = cfg.mutation_rate if cfg.mutation_rate is not None else 1.0 / L

    def eval_pop(pop):
        return np.array([ev.evaluate(ind) for ind in pop])

    # Inicialización: N_pop llamadas
    pop = rng.integers(0, 2, size=(cfg.pop_size, ev.n_genes))
    for i in range(cfg.pop_size):
        pop[i] = ev.enforce_golden(pop[i])
    fits = eval_pop(pop)
    n_evals = cfg.pop_size

    best_idx = int(np.argmin(fits))
    best_x, best_f = pop[best_idx].copy(), float(fits[best_idx])
    b_hist, e_hist, d_hist = [best_f], [n_evals], [float(np.mean(np.std(pop[:, free], axis=0)))]

    t0 = time.time()
    while n_evals + cfg.pop_size <= cfg.budget:
        # Elitismo: preservar el mejor
        elite_idx = int(np.argmin(fits))
        elite = pop[elite_idx].copy()
        elite_f = float(fits[elite_idx])

        new_pop = [elite]
        while len(new_pop) < cfg.pop_size:
            # Torneo
            i1 = rng.choice(cfg.pop_size, cfg.tournament_k, replace=False)
            i2 = rng.choice(cfg.pop_size, cfg.tournament_k, replace=False)
            p1 = pop[i1[np.argmin(fits[i1])]].copy()
            p2 = pop[i2[np.argmin(fits[i2])]].copy()
            # Cruce uniforme
            if rng.random() < cfg.crossover_rate:
                mask = rng.random(ev.n_genes) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()
            # Mutación bit-flip solo en genes libres
            for child in (c1, c2):
                flip = rng.random(L) < mut_rate
                child[free[flip]] = 1 - child[free[flip]]
                child[:] = ev.enforce_golden(child)
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(child)

        pop = np.array(new_pop[:cfg.pop_size])
        fits = eval_pop(pop)
        n_evals += cfg.pop_size

        # Elitismo garantiza que best_f no empeora
        cur_best = float(fits.min())
        if cur_best < best_f:
            best_f = cur_best
            best_x = pop[int(np.argmin(fits))].copy()
        # El elite preservado asegura que best_f nunca sube
        b_hist.append(best_f)
        e_hist.append(n_evals)
        d_hist.append(float(np.mean(np.std(pop[:, free], axis=0))))

    return Result(best_x, best_f, b_hist, e_hist, d_hist,
                  n_evals, time.time() - t0, "GA")


# ═══════════════════════════════════════════════════════════════════════════
# cGA corregido (K = κ sqrt(|L|) ln|L|)
# ═══════════════════════════════════════════════════════════════════════════

def run_cga(cfg: Config, ev: ChromosomeEvaluator) -> Result:
    rng = np.random.default_rng(cfg.seed + 2000)
    free = ev.free_idx
    L = ev.L
    # Fórmula del PDF: K = κ sqrt(|L|) ln|L|
    K = max(cfg.cga_kappa * np.sqrt(L) * np.log(L), 2.0)
    p_min = 1.0 / L
    p_max = 1.0 - 1.0 / L

    p = np.full(L, 0.5)

    def sample():
        x = np.zeros(ev.n_genes, dtype=int)
        x[free] = (rng.random(L) < p).astype(int)
        return ev.enforce_golden(x)

    def diversity():
        q = np.clip(p, 1e-12, 1 - 1e-12)
        H = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
        return float(np.sum(H) / ev.n_genes)

    best_x, best_f = None, np.inf
    b_hist, e_hist, d_hist = [], [], []
    n_evals = 0

    t0 = time.time()
    while n_evals + 2 <= cfg.budget:
        x1, x2 = sample(), sample()
        f1 = ev.evaluate(x1)
        f2 = ev.evaluate(x2)
        n_evals += 2

        if f1 < f2 or (f1 == f2 and rng.random() < 0.5):
            winner, loser, fw = x1, x2, f1
        else:
            winner, loser, fw = x2, x1, f2

        if fw < best_f:
            best_f, best_x = fw, winner.copy()

        w, l = winner[free], loser[free]
        diff = w != l
        p[diff] += np.where(w[diff] == 1, 1.0 / K, -1.0 / K)
        np.clip(p, p_min, p_max, out=p)

        b_hist.append(best_f)
        e_hist.append(n_evals)
        d_hist.append(diversity())

    return Result(best_x, best_f, b_hist, e_hist, d_hist,
                  n_evals, time.time() - t0, "cGA")


# ═══════════════════════════════════════════════════════════════════════════
# (1+1)-EA corregido (hereda fitness si hijo idéntico)
# ═══════════════════════════════════════════════════════════════════════════

def run_one_plus_one(cfg: Config, ev: ChromosomeEvaluator) -> Result:
    rng = np.random.default_rng(cfg.seed + 3000)
    free = ev.free_idx
    L = ev.L
    rate = min(cfg.ea_chi / L, 1.0)

    x = np.zeros(ev.n_genes, dtype=int)
    x[free] = rng.integers(0, 2, size=L)
    x = ev.enforce_golden(x)
    f = ev.evaluate(x)
    n_evals = 1
    best_x, best_f = x.copy(), f
    b_hist, e_hist, d_hist = [best_f], [n_evals], [0.0]

    t0 = time.time()
    while n_evals < cfg.budget:
        flip = rng.random(L) < rate
        if not flip.any():
            # Hijo idéntico: hereda fitness sin llamada (PDF V-A)
            b_hist.append(best_f)
            e_hist.append(n_evals)
            d_hist.append(0.0)
            continue
        child = x.copy()
        child[free[flip]] = 1 - child[free[flip]]
        child = ev.enforce_golden(child)
        fc = ev.evaluate(child)
        n_evals += 1
        if fc <= f:
            x, f = child, fc
        if f < best_f:
            best_f, best_x = f, x.copy()
        b_hist.append(best_f)
        e_hist.append(n_evals)
        d_hist.append(float(flip.sum()) / L)

    return Result(best_x, best_f, b_hist, e_hist, d_hist,
                  n_evals, time.time() - t0, "(1+1)-EA")


# ═══════════════════════════════════════════════════════════════════════════
# Random Search
# ═══════════════════════════════════════════════════════════════════════════

def run_rs(cfg: Config, ev: ChromosomeEvaluator) -> Result:
    rng = np.random.default_rng(cfg.seed + 4000)
    free = ev.free_idx
    L = ev.L
    best_x, best_f = None, np.inf
    b_hist, e_hist, d_hist = [], [], []
    div = L / ev.n_genes

    t0 = time.time()
    for k in range(1, cfg.budget + 1):
        x = np.zeros(ev.n_genes, dtype=int)
        x[free] = rng.integers(0, 2, size=L)
        x = ev.enforce_golden(x)
        f = ev.evaluate(x)
        if f < best_f:
            best_f, best_x = f, x.copy()
        b_hist.append(best_f)
        e_hist.append(k)
        d_hist.append(div)

    return Result(best_x, best_f, b_hist, e_hist, d_hist,
                  cfg.budget, time.time() - t0, "RS")