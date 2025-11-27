from pyomo.environ import (
    ConcreteModel, Set, Param, Var,
    NonNegativeReals, Binary,
    Constraint, Objective, Expression,
    minimize, maximize, SolverFactory, value
)
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
#  CARGA MATRIZ DE DISTANCIAS (10 nodos)
# ============================================================

# Archivo con 10 nodos (0 = depósito, 1..9 = localidades)
DIST_MATRIX_FILE = "asymmetric_from_symmetric_n10.csv"

df = pd.read_csv(DIST_MATRIX_FILE, header=None)

# Nodos = índices de la matriz [0,1,...,9]
nodes = list(df.index)
depot = 0
localities = [i for i in nodes if i != depot]

# Distancias (km) para todos los pares (i,j)
distance = {}
for i in nodes:
    for j in nodes:
        distance[(i, j)] = float(df.iloc[i, j])

# ============================================================
#  CALIDAD DE INSPECCIÓN (Cuadro 5)
# ============================================================

quality = {
    1: 85.0,
    2: 92.0,
    3: 78.0,
    4: 90.0,
    5: 82.0,
    6: 88.0,
    7: 95.0,
    8: 75.0,
    9: 84.0,
}

# ============================================================
#  RIESGO POR TRAMO (Cuadro 6)
# ============================================================

# Tramos con riesgo específico (el resto tendrán riesgo medio = 5)
risk_specific = {
    (0, 1): 3,
    (0, 2): 2,
    (0, 3): 4,
    (0, 4): 5,
    (0, 5): 6,
    (0, 6): 3,
    (0, 7): 2,
    (0, 8): 4,
    (0, 9): 5,
    (2, 8): 9,
    (2, 9): 8,
    (3, 4): 5,
    (4, 9): 7,
    (5, 6): 7,
    (8, 9): 7,
}
default_risk = 5  # riesgo medio para tramos no listados

# ============================================================
#  MODELO BASE
# ============================================================

def build_base_model():
    """
    Un solo equipo:
      - sale de 0,
      - visita un subconjunto de localidades 1..9,
      - vuelve a 0,
      - sin subrutas.
    Z1 = distancia total
    Z2 = calidad total (suma calidad de localidades visitadas)
    Z3 = riesgo total (suma riesgo de arcos recorridos)
    """
    m = ConcreteModel()

    # Conjuntos
    m.N = Set(initialize=nodes)        # todos los nodos (0..9)
    m.N0 = Set(initialize=localities)  # sólo localidades (1..9)

    # Parámetros
    def dist_init(m, i, j):
        if i == j:
            return 0.0
        # si falta (i,j), intenta (j,i) para asumir simetría
        return distance.get((i, j), distance.get((j, i), 0.0))

    def risk_init(m, i, j):
        if i == j:
            return 0.0
        # si falta (i,j), intenta (j,i); si no, riesgo medio
        return risk_specific.get((i, j), risk_specific.get((j, i), default_risk))

    def quality_init(m, i):
        # calidad solo en localidades, la base 0 vale 0
        return quality.get(i, 0.0)

    m.dist = Param(m.N, m.N, initialize=dist_init, mutable=True)
    m.risk = Param(m.N, m.N, initialize=risk_init, mutable=True)
    m.quality = Param(m.N, initialize=quality_init, mutable=True)

    # Variables
    m.x = Var(m.N, m.N, within=Binary)           # 1 si se recorre arco i->j
    m.y = Var(m.N0, within=Binary)              # 1 si la localidad i es visitada
    m.u = Var(m.N0, within=NonNegativeReals)    # variables MTZ (orden de visita)

    # ========================================================
    #  Restricciones
    # ========================================================

    # Prohibir bucles i->i
    def no_loops_rule(m, i):
        return m.x[i, i] == 0
    m.no_loops = Constraint(m.N, rule=no_loops_rule)

    # Grado de salida y entrada en localidades: 1 entra y 1 sale si se visita
    def out_degree_rule(m, i):
        return sum(m.x[i, j] for j in m.N if j != i) == m.y[i]
    m.out_degree = Constraint(m.N0, rule=out_degree_rule)

    def in_degree_rule(m, i):
        return sum(m.x[j, i] for j in m.N if j != i) == m.y[i]
    m.in_degree = Constraint(m.N0, rule=in_degree_rule)

    # En el depósito 0: exactamente 1 arco sale y 1 entra (ruta cerrada)
    def depot_out_rule(m):
        return sum(m.x[depot, j] for j in m.N0) == 1
    m.depot_out = Constraint(rule=depot_out_rule)

    def depot_in_rule(m):
        return sum(m.x[j, depot] for j in m.N0) == 1
    m.depot_in = Constraint(rule=depot_in_rule)

    # Al menos una localidad visitada
    def at_least_one_rule(m):
        return sum(m.y[i] for i in m.N0) >= 1
    m.at_least_one = Constraint(rule=at_least_one_rule)

    # MTZ para evitar subrutas (adaptado a visitas opcionales)
    n_local = len(localities)
    M = n_local

    def u_upper_rule(m, i):
        return m.u[i] <= M * m.y[i]
    m.u_upper = Constraint(m.N0, rule=u_upper_rule)

    def u_lower_rule(m, i):
        return m.u[i] >= m.y[i]
    m.u_lower = Constraint(m.N0, rule=u_lower_rule)

    def mtz_rule(m, i, j):
        if i == j:
            return Constraint.Skip
        # si x[i,j]=1 y j se visita, fuerza u[j] >= u[i] + 1
        return m.u[i] + 1 <= m.u[j] + M * (1 - m.x[i, j]) + M * (1 - m.y[j])
    m.mtz = Constraint(m.N0, m.N0, rule=mtz_rule)

    # ========================================================
    #  Expresiones de objetivos
    # ========================================================

    # Distancia total
    def z1_rule(m):
        return sum(m.dist[i, j] * m.x[i, j] for i in m.N for j in m.N)
    m.Z1 = Expression(rule=z1_rule)

    # Calidad total
    def z2_rule(m):
        return sum(m.quality[i] * m.y[i] for i in m.N0)
    m.Z2 = Expression(rule=z2_rule)

    # Riesgo total
    def z3_rule(m):
        return sum(m.risk[i, j] * m.x[i, j] for i in m.N for j in m.N)
    m.Z3 = Expression(rule=z3_rule)

    return m

# ============================================================
#  PROBLEMAS DE UN SOLO OBJETIVO
# ============================================================

def solve_single_objective(which, solver_name="glpk", tee=False):
    """
    which ∈ {"minZ1", "maxZ2", "minZ3"}
    """
    m = build_base_model()

    if which == "minZ1":
        m.obj = Objective(expr=m.Z1, sense=minimize)
    elif which == "maxZ2":
        m.obj = Objective(expr=m.Z2, sense=maximize)
    elif which == "minZ3":
        m.obj = Objective(expr=m.Z3, sense=minimize)
    else:
        raise ValueError("which debe ser 'minZ1', 'maxZ2' o 'minZ3'")

    solver = SolverFactory(solver_name)
    results = solver.solve(m, tee=tee)

    return m, value(m.Z1), value(m.Z2), value(m.Z3), results


def compute_extreme_values(solver_name="glpk"):
    """
    Calcula Z1_min, Z2_max, Z3_min, Z3_max
    """
    print("Calculando extremos del Problema 2...")

    _, z1_min, _, _, _ = solve_single_objective("minZ1", solver_name)
    _, _, z2_max, _, _ = solve_single_objective("maxZ2", solver_name)
    _, _, _, z3_min, _ = solve_single_objective("minZ3", solver_name)

    # Para Z3_max: maximizamos directamente Z3
    m_maxz3 = build_base_model()
    m_maxz3.obj = Objective(expr=m_maxz3.Z3, sense=maximize)
    solver = SolverFactory(solver_name)
    solver.solve(m_maxz3)
    z3_max = value(m_maxz3.Z3)

    print(f"  Z1_min ≈ {z1_min:.2f}")
    print(f"  Z2_max ≈ {z2_max:.2f}")
    print(f"  Z3_min ≈ {z3_min:.2f}")
    print(f"  Z3_max ≈ {z3_max:.2f}")

    return {
        "Z1_min": z1_min,
        "Z2_max": z2_max,
        "Z3_min": z3_min,
        "Z3_max": z3_max,
    }

# ============================================================
#  MÉTODO ε-CONSTRAINT
# ============================================================

def solve_epsilon_constraint(q_lb, r_ub, solver_name="glpk", tee=False):
    """
    Min Z1
    s.a. Z2 >= q_lb   (calidad mínima)
         Z3 <= r_ub   (riesgo máximo)
    """
    m = build_base_model()
    m.quality_lb = Constraint(expr=m.Z2 >= q_lb)
    m.risk_ub = Constraint(expr=m.Z3 <= r_ub)

    m.obj = Objective(expr=m.Z1, sense=minimize)

    solver = SolverFactory(solver_name)
    results = solver.solve(m, tee=tee)

    return m, value(m.Z1), value(m.Z2), value(m.Z3), results

# ============================================================
#  EXPERIMENTOS Y GRÁFICA
# ============================================================

def run_experiments(solver_name="glpk"):
    extremes = compute_extreme_values(solver_name)
    Z2_max = extremes["Z2_max"]
    Z3_min = extremes["Z3_min"]
    Z3_max = extremes["Z3_max"]

    # Diferentes combinaciones de calidad mínima y riesgo máximo
    configs = [
        (0.30, 0.30),
        (0.40, 0.40),
        (0.50, 0.50),
        (0.60, 0.60),
        (0.70, 0.70),
        (0.80, 0.80),
        (0.90, 0.90),
    ]

    solutions = []

    print("\n--- Método ε-constraint (Problema 2) ---")
    for q_frac, r_frac in configs:
        q_lb = q_frac * Z2_max
        r_ub = Z3_min + r_frac * (Z3_max - Z3_min)

        print(f"Resolviendo con Z2 >= {q_lb:.2f} "
              f"(≈{q_frac*100:.0f}% de Z2_max) y Z3 <= {r_ub:.2f} "
              f"(≈{r_frac*100:.0f}% entre Z3_min y Z3_max)...")

        m, z1, z2, z3, _ = solve_epsilon_constraint(q_lb, r_ub, solver_name)
        solutions.append({
            "q_frac": q_frac,
            "r_frac": r_frac,
            "Z1": z1,
            "Z2": z2,
            "Z3": z3,
        })
        print(f"  -> Z1 = {z1:.2f}, Z2 = {z2:.2f}, Z3 = {z3:.2f}")

    # Gráfica 3D (Z1, Z2, Z3)
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        xs = [s["Z1"] for s in solutions]
        ys = [s["Z2"] for s in solutions]
        zs = [s["Z3"] for s in solutions]

        ax.scatter(xs, ys, zs)
        ax.set_xlabel("Z1: Distancia total")
        ax.set_ylabel("Z2: Calidad total")
        ax.set_zlabel("Z3: Riesgo total")
        ax.set_title("Problema 2 - Frente de Pareto (ε-constraint)")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("No se pudo generar la gráfica 3D (quizá estás en un entorno sin interfaz gráfica).")
        print("Error:", e)

    return {"solutions": solutions, "extremes": extremes}


if __name__ == "__main__":
    run_experiments(solver_name="glpk")
