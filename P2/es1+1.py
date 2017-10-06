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
  # Número máximo de iteraciones
  iteraciones = 2000
  # Rango de interés
  x1 = 2.0
  x2 = 18.0
  # 
  c = 0.817
  # El número de variables para f es 1
  f = funcion

  # Punto inicial
  X = randint(x1,x2)
  # f(X), su fitness
  p = f(X)
  # Soluciones encontradas y su valor
  xs = [X]
  ys = [p]
  # Mutaciones exitosas (mejor resultado que la solución actual)
  S = 0
  # Mutaciones no exitosas (no es mejor que la solución actual)
  F = 0
  # Relación mutaciones exitosas sobre mutaciones totales
  ps = 0
  # Sigma la desviación estándar de la distribución normal
  # Para sigma = 1 tiende a quedarse atorado en mínimos locales.
  sig = 0.5
  for _ in range(iteraciones):
    # Generamos una nueva solución a partir de una mutación sobre la actual
    X_ = X + sig*np.random.normal()
    # Calculamos su fitness
    p_ = f(X_)
    if p_ < p:
      # La nueva solución es mejor
      X = X_
      p = p_
      xs.append(X)
      ys.append(p)
      S += 1
    else:
      F += 1
    # Actualizamos ps
    ps = S / float(S+F)
    # Ajustamos sigma
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
