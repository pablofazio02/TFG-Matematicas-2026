"""
Construcción de un bosque (iForest o EIF) y cálculo de la longitud de ruta media
de cada observación (Capítulo 2, sección 1.1 de la memoria). Mayor longitud de ruta
media = observación más difícil de aislar, es decir, más central.

Una especificación (diccionario) fija:
    tipo      "iforest" o "eif" (el EIF, con el nivel de extensión completo d - 1)
    psi       tamaño de submuestra (entero o "n": toda la muestra)
    t         número de árboles del bosque
    t_lista   números de árboles que se evalúan (por defecto, [t]); se usan los
              primeros árboles del mismo bosque
    hlims     límites de altura: "completo" (psi - 1) o "log2" (ceil(log2 psi))

Los árboles usan el módulo `random`, cuya semilla fija quien llama a la función.
"""

import math

from isolation_forest import iforest, longitudes
from eif import eif, longitudes_eif


def variantes(X, spec):
    """
    Construye el bosque de `spec` sobre X y devuelve una lista de pares
    (meta, h_media), uno por cada combinación de t y h_lim, donde h_media[j] es
    la longitud de ruta media de X[j] con los t primeros árboles.
    """
    n, d = X.shape
    psi = n if spec.get("psi", 256) == "n" else min(spec.get("psi", 256), n)
    t_max = spec.get("t", 100)
    tipo = spec.get("tipo", "iforest")
    bosque = iforest(X, t=t_max, psi=psi) if tipo == "iforest" else eif(X, t=t_max, psi=psi, nivel=d - 1)

    resultados = []
    for hl in spec.get("hlims", ["completo"]):
        h_lim = psi - 1 if hl == "completo" else math.ceil(math.log2(psi))
        H = (longitudes if tipo == "iforest" else longitudes_eif)(X, bosque, h_lim)
        for t in spec.get("t_lista", [t_max]):
            meta = {"metodo": "iForest" if tipo == "iforest" else "EIF",
                    "psi": spec.get("psi", 256), "t": t, "hlim": hl}
            resultados.append((meta, H[:, :t].mean(axis=1)))
    return resultados
