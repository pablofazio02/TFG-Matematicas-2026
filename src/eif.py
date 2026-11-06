"""
Extended Isolation Forest (EIF) desde cero.

Referencia: S. Hariri, M. Carrasco Kind y R. J. Brunner (2021). Extended
Isolation Forest. IEEE Transactions on Knowledge and Data Engineering, 33(4),
1479-1489. Páginas citadas de la versión arXiv:1811.02141v3 (§3.3-3.5, Alg. 2-3).

En cada nodo interno (Alg. 2 de Hariri et al.):
    - normal n con componentes i.i.d. N(0, 1) (dirección uniforme en la esfera);
    - según el nivel de extensión k, se anulan d - k - 1 coordenadas de n
      elegidas al azar, de modo que quedan k + 1 coordenadas activas
      (k = d - 1: extensión completa; k = 0: un solo eje activo);
    - punto de paso p uniforme en el rango (la caja) de los datos del nodo;
    - x va a la izquierda si (x - p)·n <= 0 y a la derecha en otro caso (ec. 3).
Al evaluar (Alg. 3) la longitud de ruta es la misma que en iForest:
e + c(|D_T|) al llegar a un nodo externo o al límite de altura hlim.

Diferencias de esta implementación con el Alg. 2 de Hariri et al., que hay que
señalar en la memoria:
    1. Límite de altura en el entrenamiento. Hariri et al. usan
       ceil(log2 psi) (su Alg. 1). Aquí, por defecto (limite=None), el árbol se
       construye completo, igual que el iForest de la memoria (h_lim = psi - 1
       solo en la evaluación); se puede fijar `limite` para reproducir su
       versión.
    2. Parada adicional si todas las observaciones del nodo son iguales, como
       en el iTree de la memoria. Sin ella, con observaciones repetidas y
       árbol completo la construcción no terminaría. Con datos continuos no
       cambia nada (casi seguramente).
Como en Hariri et al., un hiperplano puede no separar los datos del nodo; en
ese caso un hijo queda vacío (nodo externo con |D_T| = 0, c(0) = 0).

La aleatoriedad usa el módulo `random`, de modo que random.seed(...) fija los
árboles.
"""

import random
import numpy as np

from isolation_forest import c, _recorrer


def eif(X, t=100, psi=256, nivel=None, limite=None):
    """
    Construye t árboles de EIF, cada uno sobre una submuestra de tamaño psi
    extraída de X sin reemplazamiento. nivel = nivel de extensión k en
    {0, ..., d-1} (por defecto d - 1, extensión completa); limite = límite de
    altura en el entrenamiento (None: árbol completo).
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    if nivel is None:
        nivel = d - 1
    if not 0 <= nivel <= d - 1:
        raise ValueError("El nivel de extensión debe estar entre 0 y d - 1.")
    if psi > len(X):
        raise ValueError("psi no puede ser mayor que el número de observaciones.")
    return [eif_arbol(X[random.sample(range(len(X)), psi)], nivel, limite) for _ in range(t)]


def eif_arbol(X, nivel, limite=None, e=0):
    """
    Alg. 2 de Hariri et al. (iTree con cortes por hiperplanos) sobre la
    submuestra X (array n x d), con altura actual e.
    """
    if len(X) <= 1 or (limite is not None and e >= limite) or np.all(X == X[0]):
        return {"tam": len(X)}

    d = X.shape[1]
    # Línea 4: normal con componentes N(0, 1).
    n = np.array([random.gauss(0.0, 1.0) for _ in range(d)])
    # Línea 6: se anulan d - nivel - 1 coordenadas elegidas al azar.
    n[random.sample(range(d), d - nivel - 1)] = 0.0
    # Línea 5: punto de paso uniforme en la caja de los datos del nodo.
    minimos, maximos = X.min(axis=0), X.max(axis=0)
    p = np.array([random.uniform(minimos[j], maximos[j]) for j in range(d)])
    # Líneas 7-8: (x - p)·n <= 0 a la izquierda.
    izquierda = (X - p) @ n <= 0
    return {"tam": len(X), "n": n, "p": p,
            "izq": eif_arbol(X[izquierda], nivel, limite, e + 1),
            "der": eif_arbol(X[~izquierda], nivel, limite, e + 1)}


def path_length_eif(x, T, hlim, e=0):
    """Alg. 3 de Hariri et al.: longitud de ruta de x en el árbol T."""
    if "n" not in T or e >= hlim:
        return e + c(T["tam"])
    if (np.asarray(x) - T["p"]) @ T["n"] <= 0:
        return path_length_eif(x, T["izq"], hlim, e + 1)
    return path_length_eif(x, T["der"], hlim, e + 1)


def aplanar_eif(T):
    """
    Convierte un árbol de EIF en arrays por nodo: normal n, umbral b = n·p
    (x va a la izquierda si n·x <= b), hijos y tamaño. El nodo 0 es la raíz.
    """
    normales, b, izq, der, tam = [], [], [], [], []
    d = None
    pila = [(T, None, None)]
    while pila:
        nodo, padre, lado = pila.pop()
        k = len(tam)
        if padre is not None:
            (izq if lado == "izq" else der)[padre] = k
        tam.append(nodo["tam"])
        izq.append(-1)
        der.append(-1)
        if "n" in nodo:
            d = len(nodo["n"])
            normales.append(nodo["n"])
            b.append(float(nodo["p"] @ nodo["n"]))
            pila.append((nodo["der"], k, "der"))
            pila.append((nodo["izq"], k, "izq"))
        else:
            normales.append(None)
            b.append(0.0)
    d = d or 1
    N = np.array([v if v is not None else np.zeros(d) for v in normales])
    return {"n": N, "b": np.array(b), "izq": np.array(izq),
            "der": np.array(der), "tam": np.array(tam)}


def longitudes_eif(X, bosque, hlim):
    """
    Matriz H (n x t) con las longitudes de ruta de cada observación en cada
    árbol de EIF. Equivale a evaluar path_length_eif.
    """
    X = np.asarray(X, dtype=float)
    H = np.empty((len(X), len(bosque)))
    for k, T in enumerate(bosque):
        A = aplanar_eif(T)
        H[:, k] = _recorrer(X, A, hlim,
                            lambda Xa, nd, A=A: np.einsum("ij,ij->i", Xa, A["n"][nd]) <= A["b"][nd])
    return H
