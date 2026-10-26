"""
chromosome_patch.py
===================
Extiende Chromosome con las funciones que el artículo requiere
pero que la clase original no implementa:

  - evaluate_U(x):  fitness variante U = C_I(x)
  - evaluate_C(x):  fitness variante C = C_I(x) + ρV(x)
  - degree_violation(x): V(x) = unidades de grado violadas
  - is_feasible(x): factibilidad estricta (5)-(6)
  - optimal_U(cost_matrix, n, constraints): óptimo analítico variante U
  - solve_milp(cost_matrix, n, constraints): óptimo certificado variante C

NO modifica chromosome.py. Usa composición y herencia.
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from chromosome import Chromosome


# ── Mapeo gen ↔ arista (Ecuación 1 del PDF) ──────────────────────────────
def gene_index(n: int, i: int, j: int) -> int:
    """ℓ(i,j) = (2n−i)(i−1)/2 + (j−i), con i<j, 0-indexado."""
    return (2 * n - i) * (i - 1) // 2 + (j - i) - 1


def edge_from_index(n: int, ell: int):
    """Inverso de gene_index: devuelve (i,j) 1-indexados."""
    # Búsqueda directa (n ≤ 30 → 435 genes, trivial)
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            if gene_index(n, i, j) == ell:
                return i, j
    raise ValueError(f"Índice de gen inválido: {ell}")


class ChromosomeEvaluator:
    """
    Evaluador compartido para las variantes U y C del artículo.

    variant="U": F(x) = C_I(x) = Σ c_ij A_ij(x)          (Ec. 2)
    variant="C": F(x) = C_I(x) + ρ V(x), ρ = 1 + Σc_ij   (Ec. 7)

    V(x) = Σ_v max(0, lb_v − deg(v)) + max(0, deg(v) − ub_v)
    con lb_v = 3 para todos, ub_v = (n−1)/2+1 para coordinadores,
    ub_v = 3 + ⌊n/10⌋ para funcionales.
    """

    def __init__(self, n_agents: int, cost_matrix: np.ndarray,
                 constraints: list = None, variant: str = "U"):
        assert variant in ("U", "C")
        self.n = n_agents
        self.n_genes = n_agents * (n_agents - 1) // 2
        self.cost = np.asarray(cost_matrix, dtype=float)
        self.constraints = constraints or []
        self.variant = variant

        # Construir Chromosome de referencia para extraer golden_mask
        ref = self._build_chromosome(np.zeros(self.n_genes, dtype=int))
        self.golden_mask = np.array(ref.get_golden_genes(), dtype=int)
        self.free_idx = np.where(self.golden_mask == 0)[0]
        self.protected_idx = np.where(self.golden_mask == 1)[0]
        self.L = len(self.free_idx)

        # ρ = 1 + Σc_ij  (Ec. 7)
        self.rho = 1.0 + float(np.sum(np.triu(self.cost, k=1)))

        # Nodos coordinadores y funcionales
        self.coordinators = set(ref.get_master_nodes())
        self.functional = set(range(2, n_agents + 1)) - self.coordinators

        # Límites de grado (5)-(6)
        self.lb = 3
        self.ub_coord = (n_agents - 1) / 2.0 + 1.0
        self.ub_funct = 3 + int(n_agents / 10)

    # ── Construcción interna ───────────────────────────────────────────
    def _build_chromosome(self, genes: np.ndarray) -> Chromosome:
        chrom = Chromosome(self.n, self.cost.tolist())
        for c in self.constraints:
            chrom.set_constraint_org_team(c)
        golden = chrom.get_golden_genes()
        for i in range(self.n_genes):
            chrom._genes[i] = 1 if golden[i] == 1 else int(genes[i])
        return chrom

    # ── Fitness ────────────────────────────────────────────────────────
    def _cost_I(self, chrom: Chromosome) -> float:
        return chrom.get_total_cost()

    def _degree_violation(self, chrom: Chromosome) -> float:
        """V(x): suma de unidades de grado violadas."""
        deg = dict(chrom.get_graph().degree())
        V = 0.0
        for v in range(1, self.n + 1):
            d = deg.get(v, 0)
            if v in self.coordinators:
                ub = self.ub_coord
            elif v in self.functional:
                ub = self.ub_funct
            else:  # nodo 1 (AMS) — sin cota superior explícita en (5)-(6)
                ub = self.n - 1
            V += max(0.0, self.lb - d) + max(0.0, d - ub)
        return V

    def evaluate(self, genes: np.ndarray) -> float:
        chrom = self._build_chromosome(np.asarray(genes, dtype=int))
        cI = self._cost_I(chrom)
        if self.variant == "U":
            return cI
        V = self._degree_violation(chrom)
        return cI + self.rho * V

    def is_feasible(self, genes: np.ndarray) -> bool:
        if self.variant == "U":
            return True
        chrom = self._build_chromosome(np.asarray(genes, dtype=int))
        return self._degree_violation(chrom) < 1e-9

    def enforce_golden(self, genes: np.ndarray) -> np.ndarray:
        g = np.asarray(genes, dtype=int).copy()
        g[self.golden_mask == 1] = 1
        return g

    # ── Óptimo analítico variante U ────────────────────────────────────
    def optimal_U(self) -> float:
        """
        En variante U, el óptimo es poner todos los genes libres a 0
        (solo quedan las aristas golden).
        """
        x = np.zeros(self.n_genes, dtype=int)
        x[self.golden_mask == 1] = 1
        return self.evaluate(x)

    # ── Óptimo certificado variante C (MILP) ───────────────────────────
    def solve_milp(self):
        """
        Resuelve min C_I(x) s.t. restricciones de grado (5)-(6)
        usando scipy.optimize.milp (HiGHS).

        Variables: x ∈ {0,1}^{n_genes}
        Objetivo: min c^T x  (c = vector de costes por arista)
        Restricciones de grado: para cada nodo v,
            3 ≤ deg(v) ≤ ub_v
        Los golden genes se fijan a 1.
        """
        n, ng = self.n, self.n_genes
        c = np.zeros(ng)
        for ell in range(ng):
            i, j = edge_from_index(n, ell)
            c[ell] = self.cost[i - 1, j - 1]

        # Matriz de incidencia arista-nodo: B[ell, v] = 1 si v ∈ {i,j}
        B = np.zeros((ng, n))
        for ell in range(ng):
            i, j = edge_from_index(n, ell)
            B[ell, i - 1] = 1
            B[ell, j - 1] = 1

        # Restricciones de grado
        lb_deg = np.full(n, self.lb)
        ub_deg = np.empty(n)
        for v in range(1, n + 1):
            if v in self.coordinators:
                ub_deg[v - 1] = self.ub_coord
            elif v in self.functional:
                ub_deg[v - 1] = self.ub_funct
            else:
                ub_deg[v - 1] = n - 1

        constraints = [LinearConstraint(B.T, lb_deg, ub_deg)]

        # Fijar golden genes a 1
        bounds = Bounds(lb=np.zeros(ng), ub=np.ones(ng))
        integrality = np.ones(ng)
        lb = np.zeros(ng); ub = np.ones(ng)
        lb[self.golden_mask == 1] = 1
        ub[self.golden_mask == 1] = 1
        bounds = Bounds(lb=lb, ub=ub)

        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options={"disp": False})
        if not res.success:
            raise RuntimeError(f"MILP falló: {res.message}")
        x_star = np.round(res.x).astype(int)
        F_star = self.evaluate(x_star)  # incluye penalización (debe ser = c^T x*)
        return {
            "x_star": x_star,
            "F_star": float(F_star),
            "cost_star": float(res.fun),
            "feasible": self.is_feasible(x_star),
            "status": res.status,
            "message": res.message,
            "gap": float(res.mip_gap) if hasattr(res, "mip_gap") else 0.0,
        }


# ── Utilidades de instancia ────────────────────────────────────────────────
def make_teams(n: int):
    """
    Particiones exactas del PDF (III-D):
      n=10 → k=2 coordinadores → bloques 4/3
      n=20 → k=3 coordinadores → bloques 6/5/5
      n=30 → k=4 coordinadores → bloques 7/6/6/6
    Coordinadores: nodos 2, 3, ..., k+1
    Funcionales:   nodos k+2, ..., n
    """
    TEAM_SIZES = {
        10: [4, 3],
        20: [6, 5, 5],
        30: [7, 6, 6, 6],
    }
    if n not in TEAM_SIZES:
        raise ValueError(f"n={n} no está en el diseño del PDF (10/20/30)")
    sizes = TEAM_SIZES[n]
    k = len(sizes)
    func = list(range(k + 2, n + 1))   # nodos funcionales
    assert len(func) == sum(sizes), (len(func), sum(sizes))
    constraints = []
    idx = 0
    for b, s in enumerate(sizes):
        coord = b + 2                  # coordinador: 2, 3, ..., k+1
        team = [coord] + func[idx:idx + s]
        constraints.append(team)
        idx += s
    return constraints


def make_cost_matrix(n: int, seed: int) -> np.ndarray:
    """Matriz simétrica de costes U(1,6), redondeada a 3 decimales."""
    rng = np.random.default_rng(seed)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = round(float(rng.uniform(1.0, 6.0)), 3)
            C[i, j] = C[j, i] = c
    return C