import sys
import numpy as np
from random import randint,uniform
from sklearn.linear_model import LinearRegression,Lasso,LogisticRegressionCV
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def minimos_cuadrados(x, y):
  """
    Minimización por mínimos cuadrados.
    Ajusta un modelo lineal con coeficiente w = (w1) para 
    minimizar la suma de cuadrados entre las respuestas 
    observadas en el conjunto de datos.
    Parámetros
    ----------
    x : numpy.matrix
      Arreglo de números enteros [1..100].
    y : numpy.matrix
      Arreglo de números enteros aleatorios.
    Regresa
    ---------- 
    predicciones : numpy.ndarray
      Predicciones para los valores originales de X.
    m : numpy.ndarray
      Pendiente de la línea ajustada.
    b : numpy.ndarray
      Intercepto en y de la línea.
  """
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
  print("LS: Fórmula: y = {0}x + {1}".format(m[0,0], b[0]))
  return (predicciones,m,b)
  # Graficar el modelo original (negro) y las predicciones (rojo)
  #plt.plot(x,y,'bo')
   #plt.scatter(x, y,  color='black')
  #plt.plot(x, predicciones, color='red',linewidth=3)
  #plt.legend(['Datos observados','Función ajustada'])
  #plt.title('Mínimos cuadrados')
  #plt.show()

def lasso(x,y):
  """
    Lasso.
    Método que realiza la selección y la regularización
    de las variables del conjunto de datos, con el objetivo
    de mejorar la precisión de predicción.
    Parámetros
    ----------
    x : numpy.matrix
      Arreglo de números enteros [1..100].
    y : numpy.matrix
      Arreglo de números enteros aleatorios.
    Regresa
    ---------- 
    predicciones : numpy.ndarray
      Predicciones para los valores originales de X.
    m : numpy.ndarray
      Pendiente de la línea ajustada.
    b : numpy.ndarray
      Intercepto en y de la línea.
  """
  model = Lasso()
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print("LASSO: Fórmula: y = {0}x + {1}".format(m[0], b[0]))
  return (predicciones,m,b)

def visualizar_rl(x, y, model):
  """
    Traza el límite para la clasificación.
    Parámetros
    ----------
    x : numpy.matrix
      Arreglo de números enteros [1..100].
    y : numpy.matrix
      Arreglo de números enteros aleatorios.
    model : sklearn.linear_model.logistic.LogisticRegressionCV
      Modelo de regresión logística.
  """
  plot_clasificacion(lambda x: model.predict(x), x, y)

def plot_clasificacion(funcion_prediccion, x, y):
  """
    Grafica la clasificación.
    Parámetros
    ----------
    funcion_prediccion : function
      La función de predicción del modelo.
    x : numpy.matrix
      Arreglo de números enteros [1..100].
    y : numpy.matrix
      Arreglo de números enteros aleatorios.
    model : sklearn.linear_model.logistic.LogisticRegressionCV
      Modelo de regresión logística.
  """
  # Establecer valores mínimos y máximos
  x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
  y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
  h = 0.01
  # Generar una cuadrícula de puntos con distancia h entre ellos
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  # Predecir el valor de la función para toda la cuadrícula
  z = funcion_prediccion(np.c_[xx.ravel(), yy.ravel()])
  z = z.reshape(xx.shape)
  # Trazar el contorno y los ejemplos de entrenamiento
  plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
  plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
  plt.title("Regresion Logistica") 
  plt.show()   

def regresion_logistica(x,y):
  """
    Regresión logística.
    Clasificación de máxima entropía. Las probabilidades que
    describen los posibles resultados de un solo ensayo se 
    modelan utilizando la función logística.
    Parámetros
    ----------
    x : numpy.matrix
      Arreglo de números enteros [1..100].
    y : numpy.matrix
      Arreglo de números enteros aleatorios.
    Regresa
    ---------- 
    predicciones : numpy.ndarray
      Predicciones para los valores originales de X.
    m : numpy.ndarray
      Pendiente de la línea ajustada.
    b : numpy.ndarray
      Intercepto en y de la línea.
    model : sklearn.linear_model.logistic.LogisticRegressionCV
      Modelo de regresión logística
  """
  model = LogisticRegressionCV()
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print(type(model))
  print("Fórmula: z = {}x + {}y + {}".format(m[0,0],m[0,1], b[0]))
  return (predicciones,m,b,model)

if __name__ == '__main__':
  X = np.matrix ([[i] for i in range (1,101)])
  Y = np.matrix ([[uniform(0,1)] for _ in range (1,101)])

  plt.plot(X,Y,'b.')
  (predicciones_ls,m_ls,b_ls) = minimos_cuadrados(X, Y)
  (predicciones_lasso,m_lasso,b_lasso) = lasso(X,Y)
  # Minimos cuadrados
  plt.plot(X, predicciones_ls, color='red',linewidth=2)
  # Lasso
  plt.plot(X, predicciones_lasso, color='green',linewidth=2)
  plt.legend(['Datos observados','Mínimos cuadrados','LASSO'])
  plt.title('Regresión')
  plt.show()
  
  # Regresion logistica
  plt.title("Dataset Original")
  X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                              n_clusters_per_class=2)
  plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
              s=25, edgecolor='k')
  plt.legend('Datos')
  plt.show()
  # Clasificacion
  (predicciones_lr,m_lr,b_lr,model) = regresion_logistica(X1,Y1)
  visualizar_rl(X1,Y1,model)

  # Archivos .csv
  f = open ('coeficientes_ls.csv','w')
  for t in predicciones_ls:
    f.write("{},{},{}\n".format(m_ls[0,0],b_ls[0],t[0]))
  f.close()

  f = open ('coeficientes_lasso.csv','w')
  for t in predicciones_lasso:
    f.write("{},{},{}\n".format(m_ls[0,0],b_ls[0],t))
  f.close()

  f = open ('coeficientes_lr.csv','w')
  for t in predicciones_lr:
    f.write("{},{},{}\n".format(m_lr[0,0],b_lr[0],t))
  f.close()
