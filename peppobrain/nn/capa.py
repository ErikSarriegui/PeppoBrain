import numpy as np

class CapaLinear:
    def __init__(self, tamano_entrada, tamano_salida, funcion_activacion):
        self.tamano_salida = tamano_salida
        self.pesos = np.random.randn(tamano_salida, tamano_entrada) * np.sqrt(2. / tamano_entrada)
        self.sesgos = np.zeros((tamano_salida, 1))
        self.funcion_activacion = funcion_activacion

    def activar(self, Z):
        return self.funcion_activacion.activar(Z)

    def derivar(self, Z):
        return self.funcion_activacion.derivar(Z)