"""
Algoritmo Isolation Forest desde cero.

Referencia: F. T. Liu, K. M. Ting y Z.-H. Zhou (2012). Isolation-based anomaly
detection. ACM Transactions on Knowledge Discovery from Data, 6(1), art. 3.
En la memoria: subsecciones «Métricas y puntuación de anomalías» e «Isolation
Forest» del capítulo 1.

Notación (la de la memoria):
    X          conjunto de datos, n observaciones de R^d
    t          número de árboles (iTrees)
    psi        tamaño de la submuestra de cada árbol
    hlim       límite de altura en la evaluación (por defecto psi - 1)
    h(x)       longitud de ruta de x en un iTree, ecuación (eq:h-ajustada)
    c(psi)     longitud media de ruta, ecuación (eq:cpsi)
    s(x, psi)  puntuación de anomalía, ecuación (eq:score)

Las funciones recursivas (itree, path_length, anomaly_score) siguen literalmente
las definiciones de la memoria. `longitudes` calcula las mismas longitudes de
ruta para muchas observaciones a la vez (recorrido vectorizado); da exactamente
los mismos valores que `path_length`.

Toda la aleatoriedad de la construcción usa el módulo `random`, de modo que
`random.seed(...)` fija los árboles.
"""

import random
import numpy as np

CONST_EULER = 0.5772156649  # Constante de Euler-Mascheroni


# ---------------------------------------------------------------------------
# Ecuación (eq:cpsi): longitud de ruta promedio c(n)
# ---------------------------------------------------------------------------
def c(n):
    """
    Longitud media de una búsqueda fallida en un árbol binario de búsqueda
    construido con n elementos, con H(i) ~ ln(i) + constante de Euler.
    Admite un entero o un numpy.ndarray de enteros.
    """
    n = np.asarray(n, dtype=float)
    con_dos = np.where(n == 2, 1.0, 0.0)
    mayor = 2 * (np.log(np.maximum(n - 1, 1)) + CONST_EULER) - 2.0 * (n - 1) / np.maximum(n, 1)
    resultado = np.where(n > 2, mayor, con_dos)
    return float(resultado) if resultado.ndim == 0 else resultado


# ---------------------------------------------------------------------------
# Algoritmo 1: construcción del iForest
# ---------------------------------------------------------------------------
def iforest(X, t=100, psi=256):
    """
    Construye t iTrees, cada uno sobre una submuestra de tamaño psi extraída
    de X sin reemplazamiento (definición de bosque de aislamiento).
    """
    X = np.asarray(X, dtype=float)
    if psi > len(X):
        raise ValueError("psi no puede ser mayor que el número de observaciones.")
    return [itree(X[random.sample(range(len(X)), psi)]) for _ in range(t)]


# ---------------------------------------------------------------------------
# Algoritmo 2: construcción de un iTree
# ---------------------------------------------------------------------------
def itree(X):
    """
    Construye un iTree sobre la submuestra X (array n x d) dividiéndola
    recursivamente hasta que el nodo tiene una sola observación o todas sus
    observaciones son iguales (definición de iTree(D') en la memoria).
    """
    X = np.asarray(X, dtype=float)
    if len(X) <= 1:
        return {"tam": len(X)}

    minimos, maximos = X.min(axis=0), X.max(axis=0)
    # Atributos no constantes en el nodo: solo sobre ellos se puede cortar.
    Q = np.flatnonzero(minimos < maximos)
    if len(Q) == 0:
        return {"tam": len(X)}

    q = int(random.choice(Q))
    m, M = minimos[q], maximos[q]
    # p uniforme en (m, M]: random.random() está en [0, 1), así que
    # M - (M - m) * u está en (m, M]. Con p > m los dos hijos son no vacíos
    # (hipótesis de la Proposición sobre la estructura de un iTree). El bucle
    # solo descarta el caso, de probabilidad prácticamente nula, en que el
    # redondeo en coma flotante dé p = m.
    p = M - (M - m) * random.random()
    while p <= m:
        p = M - (M - m) * random.random()

    izquierda = X[:, q] < p
    return {"tam": len(X), "q": q, "p": p,
            "izq": itree(X[izquierda]), "der": itree(X[~izquierda])}


# ---------------------------------------------------------------------------
# Algoritmo 3: longitud de ruta h(x)
# ---------------------------------------------------------------------------
def path_length(x, T, hlim, e=0):
    """
    Recorre el árbol T con la observación x y devuelve h(x; T, hlim).

    e cuenta las aristas recorridas desde la raíz. Al llegar a un nodo externo,
    o al alcanzar el límite de altura hlim, se devuelve e + c(T.tam).
    """
    if "q" not in T or e >= hlim:
        return e + c(T["tam"])
    if x[T["q"]] < T["p"]:
        return path_length(x, T["izq"], hlim, e + 1)
    return path_length(x, T["der"], hlim, e + 1)


# ---------------------------------------------------------------------------
# Ecuación (eq:score): puntuación de anomalía s(x, psi)
# ---------------------------------------------------------------------------
def anomaly_score(x, bosque, psi, hlim=None):
    """
    s(x, psi) = 2^(-hbar_t(x) / c(psi)), con hbar_t(x) la media de h(x) sobre
    los t árboles del bosque.
    """
    if hlim is None:
        hlim = psi - 1
    E_h = sum(path_length(x, T, hlim) for T in bosque) / len(bosque)
    return 2 ** (-E_h / c(psi))


# ---------------------------------------------------------------------------
# Evaluación vectorizada
# ---------------------------------------------------------------------------
def aplanar(T):
    """
    Convierte un iTree (diccionarios anidados) en arrays indexados por nodo:
    atributo q (-1 en los nodos externos), valor de corte p, hijos izquierdo y
    derecho y tamaño |D_T|. El nodo 0 es la raíz.
    """
    q, p, izq, der, tam = [], [], [], [], []
    pila = [(T, None, None)]
    while pila:
        nodo, padre, lado = pila.pop()
        k = len(tam)
        if padre is not None:
            (izq if lado == "izq" else der)[padre] = k
        tam.append(nodo["tam"])
        izq.append(-1)
        der.append(-1)
        if "q" in nodo:
            q.append(nodo["q"])
            p.append(nodo["p"])
            pila.append((nodo["der"], k, "der"))
            pila.append((nodo["izq"], k, "izq"))
        else:
            q.append(-1)
            p.append(0.0)
    return {"q": np.array(q), "p": np.array(p), "izq": np.array(izq),
            "der": np.array(der), "tam": np.array(tam)}


def _recorrer(X, A, hlim, va_izquierda):
    """
    Recorre el árbol aplanado A con todas las filas de X a la vez y devuelve
    h(x; T, hlim) para cada una. va_izquierda(X_activos, nodos) indica, para
    cada observación activa, si baja al hijo izquierdo.
    """
    n = len(X)
    nodo = np.zeros(n, dtype=int)
    e = np.zeros(n, dtype=int)
    interno = A["izq"] >= 0
    activos = np.flatnonzero(interno[nodo] & (e < hlim))
    while len(activos):
        nd = nodo[activos]
        izquierda = va_izquierda(X[activos], nd)
        nodo[activos] = np.where(izquierda, A["izq"][nd], A["der"][nd])
        e[activos] += 1
        sigue = interno[nodo[activos]] & (e[activos] < hlim)
        activos = activos[sigue]
    return e + c(A["tam"][nodo])


def longitudes(X, bosque, hlim):
    """
    Matriz H (n x t) con H[j, k] = h(x_j; T_k, hlim) para un bosque de iTrees.
    Equivale a evaluar path_length para cada observación y cada árbol.
    """
    X = np.asarray(X, dtype=float)
    H = np.empty((len(X), len(bosque)))
    for k, T in enumerate(bosque):
        A = aplanar(T)
        H[:, k] = _recorrer(X, A, hlim,
                            lambda Xa, nd, A=A: Xa[np.arange(len(nd)), A["q"][nd]] < A["p"][nd])
    return H
