"""
E4: elección de los parámetros en los escenarios de E1 y E2, con iForest y el EIF:
    psi   en {16, 32, 64, 128, 256, 512, n}
    t     en {10, 25, 50, 100, 250, 500, 1000} (los primeros árboles de un bosque de 1000)
    h_lim en {psi - 1, ceil(log2 psi)}
R = 50.
"""
from comun import lanzar
from escenarios import normal

BASE = 7000  # semilla de la réplica r: 7000 + r
ESCENARIOS = [normal("E4 (E1)", 500, [1.0, -0.5], [[1.5, 0.3], [0.3, 1.0]]),
              normal("E4 (E2)", 1000, [2.0, -1.0], [[2.0, 1.2], [1.2, 1.0]])]
T_LISTA = [10, 25, 50, 100, 250, 500, 1000]


def specs(n):
    """Una especificación por método y tamaño de submuestra (psi < n, y psi = n)."""
    s = {}
    for psi in (16, 32, 64, 128, 256, 512, "n"):
        if psi != "n" and psi >= n:
            continue
        for tipo, nombre in (("iforest", "iForest"), ("eif", "EIF")):
            s[f"{nombre} psi={psi}"] = {"tipo": tipo, "psi": psi, "t": 1000, "t_lista": T_LISTA,
                                         "hlims": ["completo", "log2"]}
    return s


if __name__ == "__main__":
    for esc in ESCENARIOS:
        lanzar("E4", [esc], R_defecto=50, base=BASE, specs=specs(esc.n))
