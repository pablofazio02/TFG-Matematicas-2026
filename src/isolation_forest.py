"""
Cálculo del algoritmo Isolation Forest desde cero.

Referencia: F. T. Liu, K. M. Ting y Z.-H. Zhou (2012). Isolation-based anomaly
detection. ACM Transactions on Knowledge Discovery from Data, 6(1), art. 3.

Notación:
    X          conjunto de datos, lista de n observaciones x = [x_1, ..., x_n]
    t          número de árboles (iTrees)
    psi        tamaño de la submuestra de cada árbol
    hlim       límite de altura en la evaluación (por defecto psi - 1)
    h(x)       longitud de ruta de x en un iTree
    c(psi)     longitud media de ruta, ecuación (1)
    s(x, psi)  puntuación de anomalía, ecuación (2)

"""

import random
import numpy as np

# ---------------------------------------------------------------------------
# Ecuación 1: longitud de ruta promedio esperada c(n)
# ---------------------------------------------------------------------------

def c(n):
    """
    Definición 4. Calcula la longitud de camino promedio no exitoso de un Binary Search Tree.
    Utilizado para normalizar la longitud de ruta.
    
    Sea H(i) el número armónico estimado como ln(i) + 0.5772156649 (constante de Euler).
    
    Parámetros:
    -----------
    n : int o float
        Tamaño de la muestra de datos en un nodo u hoja.
        
    Devuelve:
    --------
    float
        Valor de ajuste estadístico c(n).
    """
    CONST_EULER = 0.5772156649  # Constante de Euler-Mascheroni
    if n > 2:
        return 2 * (np.log(n - 1) + CONST_EULER) - 2.0 * (n - 1) / n
    elif n == 2:
        return 1.0
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Ecuación (2):  puntuación de anomalía s(x, psi)
# ---------------------------------------------------------------------------
def anomaly_score(x, bosque, psi, hlim=None):
    """
    Definición 5. Calcula la puntuación de anomalía s(x, psi) para una observación x.

    Sea E(h(x)) la media de h(x) sobre los t árboles del bosque.

    Parámetros:
    -----------
    x : numpy.ndarray
        Instancia individual a evaluar.
    bosque : list
        Lista de árboles (iTrees) construidos durante el entrenamiento.
    psi : int
        Tamaño de la submuestra utilizada para construir cada árbol.
    hlim : int, opcional
        Límite de altura para la evaluación de h(x). Por defecto es psi - 1.

    Devuelve:
    --------
    float
        Puntuación de anomalía s(x, psi) en el rango [0, 1].
    """
    if hlim is None:
        hlim = psi - 1

    E_h = sum(path_length(x, T, hlim) for T in bosque) / len(bosque)
    return 2 ** (-E_h / c(psi))


# ---------------------------------------------------------------------------
# Algoritmo 1: construcción del iForest
# ---------------------------------------------------------------------------
def iforest(X, t=100, psi=256):
    """
    Construye t iTrees, cada uno sobre una submuestra de tamaño psi extraída
    de X sin reemplazamiento.
    """
    if psi > len(X):
        raise ValueError("psi no puede ser mayor que el número de observaciones.")
    return [itree(random.sample(X, psi)) for _ in range(t)]

# ---------------------------------------------------------------------------
# Algoritmo 2: construcción de un iTree
# ---------------------------------------------------------------------------
def itree(X):
    """
    Construye un iTree sobre la submuestra X dividiéndola recursivamente hasta
    que no se puede dividir más, es decir, hasta que el nodo tiene una sola
    observación o todas sus observaciones son iguales.
    """

    # Atributos que toman más de un valor en el nodo. Solo sobre ellos se
    # puede elegir un punto de corte que separe los datos.
    Q = [q for q in range(len(X[0])) if min(x[q] for x in X) < max(x[q] for x in X)]

    # El nodo no se puede dividir si tiene una sola observación (|X| <= 1)
    # o si todas son iguales (Q vacío): es un nodo externo.
    if len(X) <= 1 or not Q:
        return {"tam": len(X)}

    # Se elige al azar un atributo q y un punto de corte p entre el mínimo y
    # el máximo de q en el nodo.
    q = random.choice(Q)
    minimo = min(x[q] for x in X)
    maximo = max(x[q] for x in X)
    p = random.uniform(minimo, maximo)

    X_izq = [x for x in X if x[q] < p]
    X_der = [x for x in X if x[q] >= p]

    return {"tam": len(X), "q": q, "p": p,
            "izq": itree(X_izq), "der": itree(X_der)}

# ---------------------------------------------------------------------------
# Algoritmo 3: longitud de ruta h(x)
# ---------------------------------------------------------------------------
def path_length(x, T, hlim, e=0):
    """
    Recorre el árbol T con la observación x y devuelve h(x).

    e cuenta las aristas recorridas desde la raíz. Al llegar a un nodo externo,
    o al alcanzar el límite de altura hlim, se devuelve e + c(T.tam): el
    término c(T.tam) estima la longitud de ruta que faltaría recorrer en el
    subárbol de T.tam observaciones que queda por debajo.

     Parámetros:
        -----------
        x : numpy.ndarray
            Instancia individual a evaluar.
        T : Node
            Nodo actual del iTree.
        hlim : int
            Límite de altura para la recursión.
        e : int
            Longitud del camino actual.
            
        Devuelve:
        --------
        float
            La longitud del camino h(x) estimada para la instancia.

    """
    if "q" not in T or e >= hlim:
        return e + c(T["tam"])

    if x[T["q"]] < T["p"]:
        return path_length(x, T["izq"], hlim, e + 1)
    return path_length(x, T["der"], hlim, e + 1)