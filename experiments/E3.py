"""
E3: orientación de los datos. N_2(0, R_theta diag(lambda, 1) R_theta^T), n = 1000,
con lambda en {1, 4, 16} (razón entre los semiejes 1, 2 y 4) y theta en {0, 15, 30, 45}
grados. R = 100.

Todas las combinaciones comparten la semilla de cada réplica, de modo que sus muestras
son transformaciones lineales de una misma muestra N_2(0, I).
"""
from comun import lanzar
from escenarios import normal_rotada

BASE = 3000  # semilla de la réplica r: 3000 + r
ESCENARIOS = [normal_rotada(f"E3 lambda={lam} theta={th}", 1000, lam, th)
              for lam in (1, 4, 16) for th in (0, 15, 30, 45)]

if __name__ == "__main__":
    lanzar("E3", ESCENARIOS, R_defecto=100, base=BASE)
