import numpy as np
from .. import utils

"""
====================================
MODELO
====================================
"""
class Model:
    def __init__(self, *args):
        self.capas = list(args)

    def __call__(self, X):
        X = X.T
        activaciones = [X]
        Zs = []
        
        for capa in self.capas:
            Z = capa.pesos.dot(X) + capa.sesgos
            X = capa.activar(Z)
            activaciones.append(X)
            Zs.append(Z)

        return activaciones[-1], activaciones, Zs

    def inference(self, X):
        A, _, _ = self(X)
        return A

    def retropropagacion(self, Y, activaciones, Zs, tasa_aprendizaje):
        m = Y.size/10
        Y_one_hot = Y
        dA_prev = activaciones[-1] - Y_one_hot

        for i in reversed(range(len(self.capas))):
            dZ = dA_prev * self.capas[i].derivar(Zs[i])
            dW = 1 / m * dZ.dot(activaciones[i].T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

            if i > 0:
                dA_prev = self.capas[i].pesos.T.dot(dZ)

            self.capas[i].pesos -= tasa_aprendizaje * dW
            self.capas[i].sesgos -= tasa_aprendizaje * db

    def entrenar(self, X, Y, tasa_aprendizaje, epochs):
        Y = Y.T
        for i in range(epochs):
            A, activaciones, Zs = self(X)
            self.retropropagacion(Y, activaciones, Zs, tasa_aprendizaje)

            if i % (epochs / 10) == 0:
                predicciones = np.argmax(A, axis=0)
                exactitud = utils.obtener_exactitud(predicciones, Y)
                print(f"Epochs: {i} // Exactitud: {exactitud * 100}%")