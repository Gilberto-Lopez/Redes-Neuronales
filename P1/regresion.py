import numpy as np
from random import randint,uniform
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

def lasso(x,y):
  model = Lasso()
  model.fit(x,y)
  m = model.coef_
  b = model.intercept_
  predicciones = model.predict(x)
  print("LASSO: Fórmula: y = {0}x + {1}".format(m[0], b[0]))
  return predicciones

if __name__ == '__main__':
  X = np.matrix ([[i] for i in range (1,101)])
  Y = np.matrix ([[uniform(0,1)] for _ in range (1,101)])

  plt.plot(X,Y,'bo')
  predicciones_lasso = lasso(X,Y)
  plt.plot(X, predicciones_lasso, color='red',linewidth=1)
  plt.legend(['Datos observados','LASSO'])
  plt.title('Regresión')
  plt.show()
