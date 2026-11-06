"""E1: normal bivariante con correlación baja (0.245), n = 500, R = 100."""
from comun import lanzar
from escenarios import normal

BASE = 1000  # semilla de la réplica r: 1000 + r
ESCENARIOS = [normal("E1", 500, [1.0, -0.5], [[1.5, 0.3], [0.3, 1.0]])]

if __name__ == "__main__":
    lanzar("E1", ESCENARIOS, R_defecto=100, base=BASE)
