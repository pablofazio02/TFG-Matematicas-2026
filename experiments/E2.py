"""E2: normal bivariante con correlación alta (0.85), n = 1000, R = 100."""
from comun import lanzar
from escenarios import normal

BASE = 2000  # semilla de la réplica r: 2000 + r
ESCENARIOS = [normal("E2", 1000, [2.0, -1.0], [[2.0, 1.2], [1.2, 1.0]])]

if __name__ == "__main__":
    lanzar("E2", ESCENARIOS, R_defecto=100, base=BASE)
