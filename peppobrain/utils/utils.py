import numpy as np

def codificacion_one_hot(Y):
    """
    Codificación one-hot para convertir un valor numérico en un array que lo represente.
    Args:
        Y (numpy.ndarray): Array unidimensional de valores numéricos enteros.
    Returns:
        numpy.ndarray: Matriz con la codificación one-hot, donde cada fila representa un valor de Y.
    """
    num_clases = np.max(Y) + 1

    one_hot = np.eye(int(num_clases))[Y]  # Genera la codificación one-hot

    return one_hot


def decodificacion_one_hot(Y_one_hot):
    """Decodificacion one hot para combertir array codificado con one hot en un valor numerico"""
    Y = np.argmax(Y_one_hot, axis=0)
    return Y

def obtener_exactitud(predicciones, Y):
    """Operacion para calcular la exactitud de la ia durante la fase de entrenamiento"""
    return np.sum(predicciones == decodificacion_one_hot(Y)) / decodificacion_one_hot(Y).size