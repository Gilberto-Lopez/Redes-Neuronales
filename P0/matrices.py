# Para generar números aleatorios.
from random import randint

def imprime_matriz (X,n):
  """Imprime una matriz cuadrada en pantalla.
  Parámetros
  ----------
  X : list (list (int))
     Matriz a imprimir.
  n : int
     Tamaño de la matriz.
  """
  for row in X:
    print (("{}\t"*n).format(*row))

# Tamaño
n = randint (2,7)
# Matriz A
A = []
# Matriz B
B = []

for _ in range (n):
  A.append ([randint (-10,10) for _ in range (n)])
  B.append ([randint (-10,10) for _ in range (n)])

print ("A :")
imprime_matriz (A,n)
print ("\nB :")
imprime_matriz (B,n)

def prod_matrices (X,Y,n):
  """Multiplica dos matrices cuadradas del mismo orden.
  Parámetros
  ----------
  X : list (list (int))
  Y : list (list (int))
     Matrices a multiplicar.
  n : int
     Tamaño de las matrices
  Regresa
  -------
  Z : list (list (int))
     Z = X.Y
  """
  Z = [[0]*n for _ in range (n)]
  for i in range (n):
    for j in range (n):
      Zij = 0
      for k in range (n):
        Zij += A[i][k] * B[k][j]
      Z[i][j] = Zij
  return Z

AB = prod_matrices (A,B,n)
print ("\nA.B :")
imprime_matriz (AB,n)

