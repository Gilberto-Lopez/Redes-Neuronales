import numpy as np
from random import randint,uniform
from sklearn.linear_model import LinearRegression,Lasso,LogisticRegression
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
  # Modelo LASSO con constante de término de penalización
  # alpha igual a 20. Si alpha es pequeño y los valores en y
  # grandes, el resultado es una recta muy parecida a la obtenida
  # por mínimos cuadrados, si es muy grande trata de ajustar los
  # datos con una línea constante
  model = Lasso(alpha=20)
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print("LASSO: Fórmula: y = {0}x + {1}".format(m[0], b[0]))
  return (predicciones,m,b)

def regresion_logistica(x,y):
  """
    Regresión logística.
    Clasificación de máxima entropía. Las probabilidades que
    describen los posibles resultados de un solo ensayo se 
    modelan utilizando la función logística.
    Parámetros
    ----------
    x : numpy.matrix
      Arreglo de pares (x,y) que represetan puntos en el plano
      que deben ser clasificados.
    y : numpy.matrix
      Arreglo de números enteros que representan las clases
      a las que corresponde cada ejemplar en x.
    Regresa
    ---------- 
    predicciones : numpy.ndarray
      Predicciones para los valores originales de X.
    m : numpy.ndarray
      Pendiente de la línea ajustada.
    b : numpy.ndarray
      Intercepto en y de la línea.
  """
  model = LogisticRegression(solver='liblinear',C=3.)
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print("LR: Fórmula: y = {}x + {}".format(-m[0,0]/m[0,1],-b[0]/m[0,1]))
  return (predicciones,m,b)

if __name__ == '__main__':
  # Arreglo X de números 1..100
  X = np.matrix ([[i] for i in range (1,101)])
  # Arreglo Y de números aleatorios en 1..100
  Y = np.matrix ([[randint(0,100)] for _ in range (100)])

  # Regresión

  # Graficación de los puntos (x,y)
  plt.plot(X,Y,'b.')
  # Ajuste por mínimos cuadrados
  (predicciones_ls,m_ls,b_ls) = minimos_cuadrados(X, Y)
  # Ajuste por LASSO
  (predicciones_lasso,m_lasso,b_lasso) = lasso(X,Y)
  # Graficación de la línea ajustada de mínimos cuadrados
  plt.plot(X, predicciones_ls, color='red',linewidth=2)
  # Graficación de la línea ajustada de LASSO
  plt.plot(X, predicciones_lasso, color='green',linewidth=2)
  plt.legend(['Datos observados','Mínimos cuadrados','LASSO'])
  plt.title('Regresión')
  plt.show()
  
  # Regresion Logística (Clasificación)
  plt.title("Regresión Logística\nDataset Original")
  plt.gca().set_xlim([-5,5])
  plt.gca().set_ylim([-5,5])
  # Generación del conjunto de datos a clasificar
  # conformado por 2 clases
  X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=2)
  # Graficación de los datos originales
  plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
  # Ajuste de una línea para separar las clases del conjunto de datos
  # usando regresión logística
  (predicciones_lr,m_lr,b_lr) = regresion_logistica(X1,Y1)
  # Cálculo de la línea a graficar
  a = -m_lr[0,0]/m_lr[0,1]
  u = - b_lr[0]/m_lr[0,1]
  xx = np.linspace(-5,5)
  yy = a*xx + u
  # Graficación de la línea
  plt.plot(xx,yy,color = 'green',linewidth=2)
  plt.legend(['Línea de clasificación','Conjunto de datos'])
  plt.show()
  # Graficación de los datos clasificados
  plt.title("Regresión Logísitca\nPredicción")
  plt.gca().set_xlim([-5,5])
  plt.gca().set_ylim([-5,5])
  plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=predicciones_lr, s=25, edgecolor='k')
  plt.legend(['Datos clasificados'])
  plt.show()

  # Generación de archivos.csv

  # Mínimos cuadrados
  f = open ('coeficientes_ls.csv','w')
  for t in predicciones_ls:
    f.write("{},{},{}\n".format(m_ls[0,0],b_ls[0],t[0]))
  f.close()

  # LASSO
  f = open ('coeficientes_lasso.csv','w')
  for t in predicciones_lasso:
    f.write("{},{},{}\n".format(m_ls[0,0],b_ls[0],t))
  f.close()

  # Regresión logística
  f = open ('coeficientes_lr.csv','w')
  for t in predicciones_lr:
    f.write("{},{},{}\n".format(a,u,t))
  f.close()
