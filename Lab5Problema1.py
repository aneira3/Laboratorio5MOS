from pyomo.environ import (
    ConcreteModel, Set, Param, Var,
    NonNegativeReals, NonNegativeIntegers, Binary,
    Constraint, Objective, Expression,
    maximize, SolverFactory, value
)
import matplotlib.pyplot as plt

# =========================
#  DATOS
# =========================

resources = ["food", "med", "equip", "water", "blankets"]

weight_per_unit = {
    "food": 5.0,
    "med": 2.0,
    "equip": 0.3,
    "water": 6.0,
    "blankets": 3.0,
}

volume_per_unit = {
    "food": 3.0,
    "med": 1.0,
    "equip": 0.5,
    "water": 4.0,
    "blankets": 2.0,
}

impact_per_ton = {
    "food": 50.0,
    "med": 100.0,
    "equip": 120.0,
    "water": 60.0,
    "blankets": 40.0,
}

availability_units = {
    "food": 12,
    "med": 15,
    "equip": 40,
    "water": 15,
    "blankets": 20,
}

planes = ["P1", "P2", "P3", "P4"]

cap_weight = {"P1": 40.0, "P2": 50.0, "P3": 60.0, "P4": 45.0}
cap_volume = {"P1": 35.0, "P2": 40.0, "P3": 45.0, "P4": 38.0}

fixed_cost = {"P1": 15.0, "P2": 20.0, "P3": 25.0, "P4": 18.0}
var_cost   = {"P1": 0.020, "P2": 0.025, "P3": 0.030, "P4": 0.022}

zones = ["A", "B", "C", "D"]

distance_zone = {"A": 800, "B": 1200, "C": 1500, "D": 900}
multiplier_zone = {"A": 1.2, "B": 1.5, "C": 1.8, "D": 1.4}

trips = [1, 2]

# Necesidades mínimas (TON) cuadro 4
min_needs_ton = {
    ("food", "A"): 8.0,
    ("food", "B"): 12.0,
    ("food", "C"): 16.0,
    ("food", "D"): 10.0,

    ("water", "A"): 6.0,
    ("water", "B"): 9.0,
    ("water", "C"): 12.0,
    ("water", "D"): 8.0,

    ("med", "A"): 2.0,
    ("med", "B"): 3.0,
    ("med", "C"): 4.0,
    ("med", "D"): 2.0,

    ("equip", "A"): 0.6,
    ("equip", "B"): 0.9,
    ("equip", "C"): 1.2,
    ("equip", "D"): 0.6,

    ("blankets", "A"): 3.0,
    ("blankets", "B"): 5.0,
    ("blankets", "C"): 7.0,
    ("blankets", "D"): 4.0,
}

# =====================================================
#     Helper para crear solver (con límite de tiempo)
# =====================================================

def create_solver(solver_name="glpk", time_limit=60):
    solver = SolverFactory(solver_name)
    try:
        if solver_name == "glpk" and time_limit is not None:
            solver.options["tmlim"] = int(time_limit)
    except Exception:
        pass
    return solver

# =========================
#  MODELO BASE
# =========================

def build_base_model():
    m = ConcreteModel()

    # Conjuntos
    m.R = Set(initialize=resources)
    m.J = Set(initialize=planes)
    m.Z = Set(initialize=zones)
    m.V = Set(initialize=trips)

    # Parámetros
    m.weight = Param(m.R, initialize=weight_per_unit)
    m.volume = Param(m.R, initialize=volume_per_unit)
    m.impact = Param(m.R, initialize=impact_per_ton)
    m.availability = Param(m.R, initialize=availability_units)

    m.cap_weight = Param(m.J, initialize=cap_weight)
    m.cap_volume = Param(m.J, initialize=cap_volume)

    m.fixed_cost = Param(m.J, initialize=fixed_cost)
    m.var_cost   = Param(m.J, initialize=var_cost)

    m.distance   = Param(m.Z, initialize=distance_zone)
    m.multiplier = Param(m.Z, initialize=multiplier_zone)

    def min_need_init(m, r, z):
        return min_needs_ton.get((r, z), 0.0)
    m.min_need = Param(m.R, m.Z, initialize=min_need_init, default=0.0)

    # Variables
    m.x = Var(m.R, m.J, m.V, m.Z, within=NonNegativeReals)      # unidades
    m.units_equip = Var(m.J, m.V, m.Z, within=NonNegativeIntegers)

    m.y_zone = Var(m.J, m.V, m.Z, within=Binary)
    m.use_plane = Var(m.J, within=Binary)

    m.y_water_trip = Var(m.J, m.V, within=Binary)
    m.y_equip_trip = Var(m.J, m.V, within=Binary)

    # =========================
    #  Restricciones
    # =========================

    # Cada viaje a lo sumo a una zona
    def one_zone_per_trip_rule(m, j, v):
        return sum(m.y_zone[j, v, z] for z in m.Z) <= 1
    m.one_zone_per_trip = Constraint(m.J, m.V, rule=one_zone_per_trip_rule)

    # Si el viaje va a una zona, el avión se usa
    def link_use_plane_rule(m, j, v, z):
        return m.y_zone[j, v, z] <= m.use_plane[j]
    m.link_use_plane = Constraint(m.J, m.V, m.Z, rule=link_use_plane_rule)

    # Disponibilidad por recurso
    def availability_rule(m, r):
        return sum(m.x[r, j, v, z]
                   for j in m.J for v in m.V for z in m.Z) <= m.availability[r]
    m.availability_constr = Constraint(m.R, rule=availability_rule)

    # Capacidad de peso
    def weight_cap_rule(m, j, v):
        return sum(m.weight[r] * m.x[r, j, v, z]
                   for r in m.R for z in m.Z) <= m.cap_weight[j]
    m.weight_cap = Constraint(m.J, m.V, rule=weight_cap_rule)

    # Capacidad de volumen
    def volume_cap_rule(m, j, v):
        return sum(m.volume[r] * m.x[r, j, v, z]
                   for r in m.R for z in m.Z) <= m.cap_volume[j]
    m.volume_cap = Constraint(m.J, m.V, rule=volume_cap_rule)

    # Necesidades mínimas por zona y recurso (TON)
    def min_needs_rule(m, r, z):
        return sum(m.weight[r] * m.x[r, j, v, z]
                   for j in m.J for v in m.V) >= m.min_need[r, z]
    m.min_needs = Constraint(m.R, m.Z, rule=min_needs_rule)

    # Solo se puede enviar si el viaje está asignado a esa zona
    def link_x_zone_rule(m, r, j, v, z):
        return m.x[r, j, v, z] <= m.availability[r] * m.y_zone[j, v, z]
    m.link_x_zone = Constraint(m.R, m.J, m.V, m.Z, rule=link_x_zone_rule)

    # Medicinas no pueden ir en P1
    def med_plane1_rule(m, v, z):
        return m.x["med", "P1", v, z] == 0
    m.no_meds_plane1 = Constraint(m.V, m.Z, rule=med_plane1_rule)

    # Incompatibilidad agua / equipos en el mismo viaje
    def water_trip_upper_rule(m, j, v):
        return sum(m.x["water", j, v, z] for z in m.Z) \
               <= m.availability["water"] * m.y_water_trip[j, v]

    def equip_trip_upper_rule(m, j, v):
        return sum(m.x["equip", j, v, z] for z in m.Z) \
               <= m.availability["equip"] * m.y_equip_trip[j, v]

    def compatibility_rule(m, j, v):
        return m.y_water_trip[j, v] + m.y_equip_trip[j, v] <= 1

    m.water_trip_upper = Constraint(m.J, m.V, rule=water_trip_upper_rule)
    m.equip_trip_upper = Constraint(m.J, m.V, rule=equip_trip_upper_rule)
    m.compatibility    = Constraint(m.J, m.V, rule=compatibility_rule)

    # Indivisibilidad equipos médicos
    def link_equip_units_rule(m, j, v, z):
        return m.x["equip", j, v, z] == m.units_equip[j, v, z]
    m.link_equip_units = Constraint(m.J, m.V, m.Z, rule=link_equip_units_rule)

    # =========================
    #  Expresiones objetivo
    # =========================

    # Z1: impacto social total
    def z1_rule(m):
        return sum(
            m.impact[r] * m.weight[r] * m.x[r, j, v, z] * m.multiplier[z]
            for r in m.R for j in m.J for v in m.V for z in m.Z
        )
    m.Z1 = Expression(rule=z1_rule)

    # Z2: costo total
    def z2_rule(m):
        cost_fixed = sum(m.fixed_cost[j] * m.use_plane[j] for j in m.J)
        cost_var = sum(
            m.var_cost[j] * m.distance[z] * m.y_zone[j, v, z]
            for j in m.J for v in m.V for z in m.Z
        )
        return cost_fixed + cost_var
    m.Z2 = Expression(rule=z2_rule)

    return m

# =========================
#  MÉTODO DE LA SUMA PONDERADA
# =========================

def solve_weighted_sum(alpha, solver_name="glpk", tee=False):
    """
    Max Z = alpha * Z1 - (1 - alpha) * Z2

    - alpha en [0,1].
    - alpha = 0   -> min Z2 (costo mínimo).
    - alpha = 1   -> max Z1 (impacto máximo).
    """
    m = build_base_model()
    m.obj = Objective(expr=alpha * m.Z1 - (1 - alpha) * m.Z2, sense=maximize)

    solver = create_solver(solver_name)
    results = solver.solve(m, tee=tee)

    return m, value(m.Z1), value(m.Z2), results

def run_weighted_sum_experiments(alphas, solver_name="glpk"):
    """
    Ejecuta el método de la suma ponderada para una lista de alphas
    y grafica el frente aproximado en el plano (Z2, Z1).
    """
    solutions = []

    print("Método de la Suma Ponderada (solo):\n")
    for a in alphas:
        print(f"Resolviendo para alpha = {a:.2f} ...")
        m, z1, z2, _ = solve_weighted_sum(a, solver_name)
        solutions.append({"alpha": a, "Z1": z1, "Z2": z2})
        print(f"  -> Z1 = {z1:.2f}, Z2 = {z2:.2f}")

    # Gráfica Z2 vs Z1
    plt.figure()
    plt.scatter([s["Z2"] for s in solutions],
                [s["Z1"] for s in solutions])

    for s in solutions:
        plt.annotate(f"α={s['alpha']:.2f}", (s["Z2"], s["Z1"]),
                     textcoords="offset points", xytext=(5,5))

    plt.xlabel("Costo total Z2 (miles USD)")
    plt.ylabel("Impacto social Z1 (miles USD)")
    plt.title("Problema 1 - Método de la Suma Ponderada")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return solutions

if __name__ == "__main__":
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    run_weighted_sum_experiments(alphas, solver_name="glpk")
