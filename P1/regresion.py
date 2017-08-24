import numpy as np
from random import randint

def taxicab (X):
  """Calcula la norma taxicab o l-1 de un vector.
  ||X||_1 = Sum (|X_i|)
  Parámetros:
  -----------
  X : Numpy Matrix
  Regresa:
  --------
  ||X||_1
  """
  return sum (abs (X))

if __name__ == '__main__':
  X = np.matrix ([[i] for i in range (1,101)])
  Y = np.matrix ([randint (100) for _ in range (100)])