import numpy as np
from random import randint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Minimización por mínimos cuadrados.
def minimos_cuadrados(x, y):
  # Objeto Linear Regression 
  model = LinearRegression()
  # Ajustar modelo lineal a los datos
  model.fit(x,y)
  # Obtener la pendiente de la línea ajustada
  m = model.coef_
  # Obtener el intercepto en y de la línea
  b = model.intercept_
  # Obtener predicciones para los valores originales de X
  predicciones = model.predict(x)
  # Fórmula
  print("Fórmula: y = {0}x + {1}", m, b)
  # Graficar el modelo original (negro) y las predicciones (rojo)
  plt.scatter(x, y,  color='black')
  plt.plot(x, predicciones, color='red',linewidth=3)
  plt.legend(['Función ajustada','Datos observados'])
  plt.title('Mínimos cuadrados')
  plt.show()

if __name__ == '__main__':
  X = np.matrix ([[i] for i in range (1,101)])
  Y = np.matrix ([[randint(1,100)] for _ in range (1,101)])

  minimos_cuadrados(X, Y)