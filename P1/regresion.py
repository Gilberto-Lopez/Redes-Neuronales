import sys
import numpy as np
from random import randint,uniform
from sklearn.linear_model import LinearRegression,Lasso,LogisticRegression
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
  print("LS: Fórmula: y = {0}x + {1}".format(m[0,0], b[0]))
  return (predicciones,m,b)

def lasso(x,y):
  model = Lasso()
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print("LASSO: Fórmula: y = {0}x + {1}".format(m[0], b[0]))
  return (predicciones,m,b)

def regresion_logistica(x,y):
  model = LogisticRegression(solver='liblinear',C=3.)
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print("Fórmula: z = {}x + {}y + {}".format(m[0,0],m[0,1], b[0]))
  return (predicciones,m,b)

if __name__ == '__main__':
  X = np.matrix ([[i] for i in range (1,101)])
  Y = np.matrix ([[uniform(0,1)] for _ in range (1,101)])

  plt.plot(X,Y,'b.')
  (predicciones_ls,m_ls,b_ls) = minimos_cuadrados(X, Y)
  (predicciones_lasso,m_lasso,b_lasso) = lasso(X,Y)
  plt.plot(X, predicciones_ls, color='red',linewidth=2)
  plt.plot(X, predicciones_lasso, color='green',linewidth=2)
  plt.legend(['Datos observados','Mínimos cuadrados','LASSO'])
  plt.title('Regresión')
  plt.show()
  
  #regresion logistica
  plt.title("Dataset Original")
  X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                              n_clusters_per_class=2)
  plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
              s=25, edgecolor='k')
  plt.legend(['Conjunto de datos'])
  plt.show()
  (predicciones_lr,m_lr,b_lr) = regresion_logistica(X1,Y1)
  plt.title("Regresión Logísitca")
  plt.gca().set_ylim([-4,4])
  plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
              s=25, edgecolor='k')
  a = -m_lr[0,0]/m_lr[0,1]
  xx = np.linspace(-3.5,3.5)
  yy = a*xx - b_lr[0]/m_lr[0,1]
  plt.plot(xx,yy,color = 'green',linewidth=2)
  plt.legend(['Línea de calsificación','Datos'])
  plt.show()

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
