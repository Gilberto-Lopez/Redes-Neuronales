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

# Delta de Kronecker
delta = lambda n, m: 1 if n == m else 0
# Identidad de tamaño n
In = [[delta(i,j) for j in range (n)] for i in range (n)]
# Matriz aumentada (A|In)
AI = [RA + RI for (RA,RI) in zip(A,In)]

for i in range (n):
  for j in range (n):
    Aji = AI[j][i]
    if j != i and Aji != 0:
      if AI[i][i] == 0:
        AI[i] = [xi + xn for (xi,xn) in zip (AI[i],AI[n-1])]
      Aii = AI[i][i]
      AI[j] = [-Aii/Aji * x for x in AI[j]]
      AI[j] = [xj + xi for (xj,xi) in zip (AI[j],AI[i])]
for i in range (n):
  aii = AI[i][i]
  AI[i] = [aii * x for x in AI[i]]

A_inv = [X[n:] for X in AI]
imprime_matriz(A_inv,n)
imprime_matriz(prod_matrices(A_inv,A,n),n)