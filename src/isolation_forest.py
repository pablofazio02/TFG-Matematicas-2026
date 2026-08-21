"""
Módulo: isolation_forest.py
Implementación formal de Isolation Forest basado en Liu et al. (2012)
Referencia: "Isolation-Based Anomaly Detection", ACM TKDD, Vol. 6, No. 1, 2012.
"""

__author__ = "Pablo Fazio Arrabal"
__institution__ = "Universidad de Málaga (UMA)"
__degree__ = "Doble Grado en Matemáticas e Ingeniería Informática"

import numpy as np

def c_factor(n):
    """
    Calcula la longitud de camino promedio no exitoso de un Binary Search Tree.
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
    if n > 2:
        CONST_EULER = 0.5772156649
        harmonic_number = np.log(n - 1) + CONST_EULER
        return 2.0 * harmonic_number - (2.0 * (n - 1) / n)
    elif n == 2:
        return 1.0
    else:
        return 0.0

class Node:
    """
    Estructura de datos para los nodos del Isolation Tree.
    Un iTree es un árbol binario propio donde cada nodo interno tiene exactamente 
    dos nodos hijos y los nodos externos son hojas.
    """

    def __init__(self, left=None, right=None, split_att=None, split_val=None, size=0, node_type='internal'):
        self.left = left                    # Rama izquierda (T_L)
        self.right = right                  # Rama derecha (T_R)
        self.split_att = split_att          # Atributo de corte aleatorio (q)
        self.split_val = split_val          # Valor de corte aleatorio (p)
        self.size = size                    # Número de instancias en el nodo (|X'|)
        self.node_type = node_type          # Tipo de nodo: 'internal' o 'external'

def iTree(X_prime):
    """
    Algoritmo 2: iTree(X') - Construcción recursiva de un árbol de aislamiento.
    
    El proceso de partición selecciona aleatoriamente un atributo q y un punto de 
    corte p entre el mínimo y máximo de dicho atributo q. La división continúa hasta 
    que el nodo tiene una sola instancia, todas las instancias son idénticas, 
    o se alcanza el límite de altura predefinido.
    
    Parámetros:
    -----------
    X_prime : numpy.ndarray
        Submuestra de datos actual en el nodo.
        
    Devuelve:
    --------
    Node
        El nodo raíz del subárbol de aislamientogenerado.
    """

    n_samples, n_features = X_prime.shape
    
    # Condición de parada: El nodo contiene 1 o menos instancias
    if n_samples <= 1:
        return Node(size=n_samples, node_type='external')
    
    # Condición de parada adicional: Todos los datos son idénticos 
    if np.all(X_prime == X_prime[0]):
        return Node(size=n_samples, node_type='external')
    
    # Seleccionamos aleatoriamente un atributo q
    q = np.random.randint(0, n_features)
    min_val = np.min(X_prime[:, q])
    max_val = np.max(X_prime[:, q])
    
    # Si todos los valores del atributo aleatorio son iguales, 
    # forzamos un nodo externo para evitar bucles infinitos.
    if min_val == max_val:
        return Node(size=n_samples, node_type='external')
    
    # Seleccionamos aleatoriamente un valor de partición p entre min_val y max_val
    p = np.random.uniform(min_val, max_val)
    
    # Particionamos los datos en dos subconjuntos basados en el valor de corte p
    left_mask = X_prime[:, q] < p

    X_l = X_prime[left_mask]
    X_r = X_prime[~left_mask]
    
    # Llamadas recursivas para construir los subárboles
    left_child = iTree(X_l)
    right_child = iTree(X_r)
    
    # Devolvemos nodo interno
    return Node(left=left_child, right=right_child, split_att=q, split_val=p, 
                size=n_samples, node_type='internal')


def PathLength(x, T, height_limit, e=0):
    """
    Algoritmo 3: PathLength(x, T, hlim, e) - Calcula la longitud de ruta de x.
    
    Evalúa la longitud del camino recorrido por una instancia x desde la raíz 
    hasta un nodo externo. Incorpora el factor de ajuste c(Size) si el árbol 
    termina antes debido al límite de altura.
    
    Parámetros:
    -----------
    x : numpy.ndarray
        Instancia individual a evaluar.
    T : Node
        Nodo actual del iTree.
    height_limit : int
        Límite de altura para la recursión.
    e : int
        Longitud del camino actual.
        
    Devuelve:
    --------
    float
        La longitud del camino h(x) estimada para la instancia.
    """

    if T.node_type == 'external' or e >= height_limit:
        return e + c_factor(T.size)
    
    a = T.split_att
    if x[a] < T.split_val:
        return PathLength(x, T.left, height_limit, e + 1)
    else:
        return PathLength(x, T.right, height_limit, e + 1)

class IsolationForest:
    """
    Algoritmo 1: iForest(X, t, psi) - Ensamblado completo del Isolation Forest.
    
    Combina múltiples árboles de aislamiento entrenados sobre submuestras aleatorias.
    """
    def __init__(self, n_estimators=100, subsampling_size=256):
        """
        Parámetros:
        -----------
        n_estimators : int (t)
            Número de árboles a construir. Valor por defecto: 100.
        subsampling_size : int (psi)
            Tamaño de la submuestra sin reemplazo. Valor por defecto: 256.
        """
        self.t = n_estimators
        self.psi = subsampling_size
        self.trees = []
        self.height_limit = 0
    
    def fit(self, X):
        """
        Construye el bosque de aislamiento (etapa de entrenamiento).
        """
        n_samples = X.shape[0]
        actual_psi = min(self.psi, n_samples)
        
        # El límite de altura crece logarítmicamente respecto al tamaño de muestra
        # hlim = ceil(log2(psi))
        self.height_limit = int(np.ceil(np.log2(max(actual_psi, 2))))
        self.trees = []
        
        for _ in range(self.t):
            # Muestreo aleatorio sin reemplazo
            indices = np.random.choice(n_samples, actual_psi, replace=False)
            X_prime = X[indices]
            
            # Construir iTree
            tree = iTree(X_prime)
            self.trees.append(tree)
            
        return self


    def anomaly_score(self, X_eval):

        """
        Etapa de Evaluación: Cálculo del anomaly score.
        
        s(x, psi) = 2^(- E(h(x)) / c(psi))
        
        - Si s -> 1, la instancia es altamente anómala.
        - Si s -> 0, la instancia es muy normal.
        - Si s -> 0.5, no hay distinción clara.
        
        Devuelve:
        --------
        scores : numpy.ndarray
            Puntuaciones de anomalía s(x) en el rango [0, 1].
        mean_path_length : numpy.ndarray
            La esperanza de la longitud de camino E(h(x)).
        """

        X_eval = np.atleast_2d(X_eval)
        n_eval = len(X_eval)
        paths = np.zeros((n_eval, self.t))
        
        # Propagar instancias a través de cada árbol
        for t_idx, tree in enumerate(self.trees):
            for i in range(n_eval):
                paths[i, t_idx] = PathLength(X_eval[i], tree, self.height_limit, e=0)
                
        # Calcular esperanza estadística de h(x)
        mean_path_length = np.mean(paths, axis=1)
        
        # Calcular c(psi) para normalizar la longitud de camino
        c_psi = c_factor(self.psi)
        
        # Computar score
        if c_psi == 0:
            scores = np.zeros(n_eval)
        else:
            scores = 2.0 ** (- (mean_path_length / c_psi))
            
        return scores, mean_path_length