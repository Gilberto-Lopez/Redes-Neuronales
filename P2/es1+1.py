from random import randint,random
import matplotlib.pyplot as plt
import numpy as np

def funcion(x):
    """
    La función que se va a minimizar.
    Parámetros
    ----------
    x : float
        Valor de x a evaluar en la función.
    """
    return ((x-10)**4) - (50 * (x-10)**2) + (8*x) - 100

if __name__ == '__main__':
  iteraciones = 200
  x1 = 2.0
  x2 = 18.0
  c = 0.817
  f = funcion

  X = randint(x1,x2)
  p = f(X)
  xs = [X]
  ys = [p]
  S = 0
  F = 0
  ps = 0
  sig = 0.1
  for _ in range(iteraciones):
    X_ = X + sig*np.random.normal()
    p_ = f(X_)
    if p_ < p:
      X = X_
      p = p_
      xs.append(X)
      ys.append(p)
      S += 1
    else:
      F += 1
    #xs.append(X)
    ps = S / float(S+F)
    if ps < 1.0/5:
      sig = sig*c
    elif ps > 1.0/5:
      sig = sig/c

  puntos_x = np.arange(x1,x2,0.01)
  puntos_y = f(puntos_x)
  plt.plot(puntos_x,puntos_y,'r-')
  plt.scatter(xs,ys)
  plt.plot(xs,ys)
  plt.title('Estrategia evolucionaria 1+1.')
  plt.show()
