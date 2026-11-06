"""
Configuración y argumentos comunes de los programas de los experimentos.

Uso (con el entorno virtual activado, desde la raíz del repositorio):
    python experiments/E1.py                          # número de réplicas por defecto
    python experiments/E1.py --R 3 --archivo prueba   # prueba rápida
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Configuración estándar de la memoria: t = 100, psi = 256, h_lim = psi - 1
ESTANDAR = {"iForest": {"tipo": "iforest", "psi": 256, "t": 100},
            "EIF": {"tipo": "eif", "psi": 256, "t": 100}}


def argumentos(R_defecto):
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=R_defecto, help="número de réplicas")
    p.add_argument("--n-jobs", type=int, default=-1, help="procesos en paralelo (-1: todos)")
    p.add_argument("--archivo", default=None, help="nombre del CSV de resultados")
    return p.parse_args()


def lanzar(experimento, escenarios, R_defecto, base, specs=ESTANDAR, **opciones):
    """Ejecuta las réplicas del experimento y guarda los resultados en results/."""
    from experimento import ejecutar
    args = argumentos(R_defecto)
    archivo = args.archivo or experimento
    n = ejecutar(experimento, escenarios, specs, args.R, base, archivo=archivo, n_jobs=args.n_jobs, **opciones)
    print(f"{experimento}: {n} réplicas nuevas -> results/{archivo}.csv")
