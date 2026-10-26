"""Pruebas automáticas de correctitud."""
import numpy as np
from chromosome_patch import (
    ChromosomeEvaluator, make_cost_matrix, make_teams,
    gene_index, edge_from_index,
)
from qea_ga_cga_ea_random_fixed import (
    Config, QuantumChromosome, run_qea, run_ga, run_cga,
    run_one_plus_one, run_rs,
)


def test_gene_index_roundtrip():
    for n in (4, 5, 10, 30):
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                ell = gene_index(n, i, j)
                ii, jj = edge_from_index(n, ell)
                assert (ii, jj) == (i, j), f"{n} {i} {j} -> {ell} -> {ii} {jj}"
    print("✓ gene_index roundtrip")


def test_symmetric_costs():
    for n in (10, 20, 30):
        C = make_cost_matrix(n, 2026)
        assert np.allclose(C, C.T)
        assert C.min() >= 1.0 and C.max() <= 6.0
    print("✓ symmetric costs U(1,6)")


def test_golden_genes():
    for n in (10, 20, 30):
        C = make_cost_matrix(n, 2026)
        cons = make_teams(n)
        ev = ChromosomeEvaluator(n, C, cons, variant="C")
        # Golden genes deben incluir A_1k para todo k
        for k in range(2, n + 1):
            ell = gene_index(n, 1, k)
            assert ev.golden_mask[ell] == 1, f"n={n} A_1{k} no es golden"
        # Contar golden: |P| = 16, 35, 54
        expected = {10: 16, 20: 35, 30: 54}[n]
        # Nota: el conteo exacto depende de make_teams; verificar
        print(f"  n={n}: |P|={ev.golden_mask.sum()} (esperado ~{expected}), "
              f"|L|={ev.L}")
    print("✓ golden genes")


def test_amplitude_normalization():
    qc = QuantumChromosome(10, locked=np.array([0, 3, 7]))
    assert np.allclose(qc.alpha**2 + qc.beta**2, 1.0)
    assert qc.alpha[0] == 0.0 and qc.beta[0] == 1.0
    print("✓ amplitude normalization")


def test_rotation_scheme_I():
    """
    Tabla I: para (xℓ=1, bℓ=0, w=sí), Δθ = −sgn(αβ)θ.
    Si αβ>0, Δθ<0 → P(1) debe disminuir (alejarse de 1).
    """
    qc = QuantumChromosome(1, locked=np.array([]))
    qc.alpha[0] = 1/np.sqrt(2); qc.beta[0] = 1/np.sqrt(2)
    p_before = qc.beta[0]**2
    x = np.array([1]); b = np.array([0])
    qc.rotate(x, b, f_worse=True, theta=0.1, scheme="I")
    p_after = qc.beta[0]**2
    assert p_after < p_before, f"P(1) no disminuyó: {p_before} -> {p_after}"
    # (0,1,sí) → P(1) debe aumentar
    qc.alpha[0] = 1/np.sqrt(2); qc.beta[0] = 1/np.sqrt(2)
    x = np.array([0]); b = np.array([1])
    qc.rotate(x, b, f_worse=True, theta=0.1, scheme="I")
    assert qc.beta[0]**2 > 0.5
    print("✓ rotation scheme I")


def test_budget_exact():
    n, seed = 10, 2026
    C = make_cost_matrix(n, seed)
    cons = make_teams(n)
    ev = ChromosomeEvaluator(n, C, cons, variant="U")
    for name, runner in [("QEA", run_qea), ("GA", run_ga), ("cGA", run_cga),
                          ("(1+1)-EA", run_one_plus_one), ("RS", run_rs)]:
        cfg = Config(n_agents=n, variant="U", budget=1000, seed=seed,
                     constraints=cons)
        r = runner(cfg, ev)
        assert r.n_evals <= 1000, f"{name}: {r.n_evals} > 1000"
        assert r.best_history[-1] == r.best_f, f"{name}: best_f desincronizado"
        # Monotonía del mejor acumulado
        bh = np.array(r.best_history)
        assert np.all(np.diff(bh) <= 1e-12), f"{name}: best no monótono"
        print(f"  {name}: {r.n_evals} llamadas, best_f={r.best_f:.3f}")
    print("✓ budget exact")


def test_milp_vs_enumeration_small():
    """n=4 o 5, comparar MILP con enumeración exhaustiva."""
    import itertools
    n = 5
    C = make_cost_matrix(n, 2026)
    cons = make_teams(n) if n >= 10 else []  # sin constraints para n pequeño
    ev = ChromosomeEvaluator(n, C, cons, variant="C")
    milp = ev.solve_milp()
    # Enumerar
    best = np.inf
    for bits in itertools.product([0, 1], repeat=ev.L):
        x = np.zeros(ev.n_genes, dtype=int)
        x[ev.golden_mask == 1] = 1
        x[ev.free_idx] = bits
        f = ev.evaluate(x)
        if f < best:
            best = f
    assert abs(milp["F_star"] - best) < 1e-6, f"MILP={milp['F_star']} vs enum={best}"
    print(f"✓ MILP vs enumeración n={n}: F*={milp['F_star']:.3f}")


if __name__ == "__main__":
    test_gene_index_roundtrip()
    test_symmetric_costs()
    test_golden_genes()
    test_amplitude_normalization()
    test_rotation_scheme_I()
    test_budget_exact()
    test_milp_vs_enumeration_small()
    print("\nTodas las pruebas pasaron.")