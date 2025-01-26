# 🧠 **PeppoBrain**



![pepe pensando muy fuerte](repo_assets/peppobrain_img.png)

## 🤔 **¿Qué es esto?**
Porque hasta Pepe merece una oportunidad para pensar. Una librería de aprendizaje automático hecha a partir de módulos matemáticos. No es la que mejor rendimiento tiene (Pepe no es muy listo) ni la que más soporte tiene (Pepe se siente solo).

## 🕴️ **Modo Serio**

## 🚀 **Inicio Rápido**
Puedes utilizar a Pepe en Colab. [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ErikSarriegui/PeppoBrain/blob/main/quickstart.ipynb)

```python
import peppobrain as pb

# Crea un modelo tan simple como los pensamientos de Pepe
modelo = pb.Model(
    pb.CapaLinear(784, 128, pb.ReLU()),
    pb.CapaLinear(128, 64, pb.ReLU()),
    pb.CapaLinear(64, 10, pb.Softmax())
)

# Entrénalo (los resultados pueden variar según el humor de Pepe)
modelo.entrenar(x, y, 0.3, 50)
```
## **📦 Instalación**
`pip install git+https://github.com/ErikSarriegui/PeppoBrain`

## **⚠️ Advertencia**
Ni Pepe ni nosotros garantizamos el correcto funcionamiento de las redes neuronales generadas con esta librería. Además, Pepe no ha aprendido a usar GPUs, por lo que su velocidad de pensar es muy lenta. Pepe está haciendo su mejor esfuerzo, por favor ten paciencia.

## **📝 Licencia (MIT)**
Haz lo que quieras con esto. Pepe está demasiado ocupado intentando entender la retropropagación como para preocuparse por licencias.

*Disclaimer: Ningún meme ni rana fue dañado durante el desarrollo de este proyecto, aunque algunas neuronas pueden haber quedado ligeramente confundidas.*
