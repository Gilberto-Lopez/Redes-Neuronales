#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python 2.7

from random import uniform
import math
import matplotlib.pyplot as plt
import numpy as np

class Perceptron (object):
  """Clase Perceptron para modelar un perceptón simple.
  """

  # Función signo. Función de activación.
  SIGNUM = lambda x: 1 if x > 0 else -1

  def __init__ (self, n, activacion = SIGNUM, tasa_aprendizaje = 0.1, error = 0.1):
    """Inicializa un perceptrón con una cantidad fija de pesos para las
    entradas, establecidos como números aleatorios en el intervalo
    [-1,1], y un umbral para el sesgo del perceptrón en ese mismo
    intervalo. También recibe la función de activación para el preceptrón,
    la tasa de aprendizaje (en el intervalo (0,1)) y el umbral de error
    para el entrenamiento (no negativo).
    Parámetros:
    -----------
    n : Int
      El número de entradas del perceptrón.
    activacion : Float -> Int
      La función de activación. SIGNUM por defecto.
    tasa_aprendizaje : Float
      La tasa de aprendizaje. Número positivo.
    error : float
      Umbral de error para el entrenamiento.
    """
    self.n = n
    self.pesos = [uniform (-1, 1) for _ in range (n)]
    self.theta = uniform (-1, 1)
    self.f = activacion
    self.alpha = tasa_aprendizaje
    self.error = abs (error)

  def predice (self, entradas):
    """Calcula la salida del perceptrón para un ejemplar dado.
    La salida del perceptrón está dada por:
      Y = f(Sum i (w_i.x_i) - theta)
    donde x_i es el i-ésimo valor de entrada, w_i el i-ésimo peso para x_i,
    theta el umbral y f es la función de activación del perceptrón.
    Parámetros:
    -----------
    entradas : [Float]
      Los valores de entrada para el perceptrón (ejemplar).
    Regresa:
    -----------
    c : Int
      La clasificación que predice el perceptrón para el ejemplar.
    """
    if len (entradas) != self.n:
      raise Exception ('Número de entradas incorrecto.')
    suma_ponderada = sum ([w_i * x_i for (w_i, x_i) in zip (self.pesos, entradas)])
    return self.f (suma_ponderada - self.theta)

  def __entrena (self, ejemplar, salida_esperada):
    # Método auxiliar para entrena(), calcula el error del perceptrón
    # con un ejemplar dado y actualiza los pesos del perceptrón.
    salida_perceptron = self.predice (ejemplar)

    # *** El proceso de entrenamiento se muestra en pantalla.
    print('Ejemplar: {}\tSalida: {}\tSalida esperada: {}'.format(ejemplar, salida_perceptron, salida_esperada))

    error = salida_esperada - salida_perceptron
    if error != 0:
      self.theta = self.theta - self.alpha * error
      self.pesos = [w_i + self.alpha * x_i * error for (w_i, x_i) in zip (self.pesos, ejemplar)]

      # *** El proceso de entrenamiento se muestra en pantalla.
      print('Pesos actualizados:\t{}'.format([self.theta] + self.pesos))
    return abs (error)

  def entrena (self, conjunto, salidas, iteraciones = 1000):
    """Proceso de entrenamiento de perceptrón. Itera sobre el conjunto de
    ejemplares de entrada el número de iteraciones dado o hasta que el
    error de la salida del perceptrón se haya minimizado por debajo del
    umbral de error permitido. El máximo de iteraciones permitidas es
    1000 por defecto.
    Parámetros:
    -----------
    conjunto : [[Float]]
      El conjunto de ejemplares para el entrenamiento.
    salidas : [Int]
      El conjunto de salidas esperadas para cada ejemplar.
    iteraciones : Int
      El número máximo de iteraciones permitidas.
    Regresa:
    -----------
    w : [Float]
      La lista de pesos resultantes del entrenamiento
    e : Float
      El error resultante al clasificar el conjunto después del entrenamiento.
    """
    # Valores absolutos de los errores de la iteración
    m = len (conjunto)
    errores = [0 for _ in range (m)]
    for _ in range (iteraciones):
      for i in range (m):
        errores[i] = self.__entrena (conjunto[i], salidas[i])
      d = sum (errores) / float(m)
      print('*** Error total en la iteración: {}'.format(d))
      if d <= self.error:
        break
    return ([self.theta]+self.pesos, d)

if __name__ == '__main__':
  D = [[1.4,54.1],
    [2.7,76.2],
    [7.7,14.4],
    [3.5,54.4],
    [8.3,0.35],
    [1.9,76.5],
    [7.8,34.3],
    [9.9,12.4],
    [3.8,52.8],
    [9.3,28.0]]
  y = [1,1,-1,1,-1,1,-1,-1,1,-1]

  ### PERCEPTRÓN SIMPLE
  
  print('Perceptrón Simple:\n')
  params = {'n':2, 'tasa_aprendizaje':0.003}
  p = Perceptron (**params)
  print('Inicio\n\tPesos: {}'.format([p.theta]+p.pesos))
  (w,e) = p.entrena (conjunto = D, salidas = y)
  print('Final\n\tPesos: {}\tError:{}'.format(w,e))

  a = -w[1]/w[2]
  b = -w[0]/w[2]
  print('Ecuación del plano: y = {}x + {}'.format(a,b))
  plt.scatter([s[0] for s in D],[s[1] for s in D],c=y)
  xx = np.linspace(1,10)
  yy = a*xx -b
  plt.plot(xx,yy,color='green',linewidth=1)
  plt.title('Perceptrón Simple')
  plt.xlabel('pH')
  plt.ylabel('Concentración de Fe')
  plt.show()
