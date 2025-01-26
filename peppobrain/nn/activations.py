import numpy as np

class ReLU:
    def activar(self, Z):
        """Aplica la función ReLU a un arreglo."""
        return np.maximum(Z, 0)

    def derivar(self, Z):
        """Calcula la derivada de la función ReLU."""
        return Z > 0

class Sigmoid:
    def sigmoid(self, Z):
        """Aplica la función sigmoid a un arreglo."""
        return 1 / (1 + np.exp(-Z))

    def derivar(self, Z):
        """Calcula la derivada de la función sigmoid."""
        s = self.sigmoid(Z)
        return s * (1 - s)

class Softmax:
    def activar(self, Z):
        """Aplica la función softmax a un arreglo."""
        expZ = np.exp(Z - np.max(Z, axis=0))
        return expZ / np.sum(expZ, axis=0)

    def derivar(self, Z):
        """Calcula la derivada de la función softmax."""
        expZ = np.exp(Z - np.max(Z, axis=0))
        return expZ / np.sum(expZ, axis=0)