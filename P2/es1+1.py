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
  iteraciones = 200
  # Rango de interés
  x1 = 2.0
  x2 = 18.0
  # Valor recomendado para el ajuste de sigma
  c = 0.817
  # El número n de variables para f es 1
  f = funcion

  # Punto inicial
  X = randint(x1,x2)
  # f(X), su fitness
  p = f(X)
  # Soluciones encontradas y su valor
  xs = [X]
  ys = [p]

  # Mutaciones realizadas (1 para exitosas, 0 para no exitosas)
  # Las mutaciones exitosas tienen mejor fitness que la solución actual
  M = []
  S,F = 0,0
  # Relación mutaciones exitosas sobre mutaciones totales
  ps = 0
  # Sigma la desviación estándar de la distribución normal
  # Para sigma pequeña tiende a quedarse atorado en mínimos locales.
  sig = 10
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
      M.append(1)
    else:
      M.append(0)
    # Actualizamos ps cada n iteraciones
    # Nos fijamos en las últimas (a lo más) 10n mutaciones
    M_10 = M[-10:]
    ps = sum(M_10)/float(len(M_10))
    # Ajustamos sigma
    # 1/5 success rule: Se maximiza la velocidad del progreso en la optimización
    # cuando ~1/5 de las mutaciones resultan existosas
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
