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

def prod_matrices (X,Y,n):
  """Multiplica dos matrices cuadradas del mismo orden.
  Parámetros
  ----------
  X : list (list (int))
  Y : list (list (int))
     Matrices a multiplicar.
  n : int
     Tamaño de las matrices.
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

def inv_matriz (X,n):
  """Invierte una matriz cuadrada de tamaño n.
  Parámetros
  ----------
  X : list (list (int))
     Matriz a invertir.
  n : int
     Tamaño de la matriz.
  Regresa
  -------
  X' : list (list (int))
      X' = X^{-1}
  """
  # Delta de Kronecker
  delta = lambda s, t: 1 if s == t else 0
  # Identidad de tamaño n
  In = [[delta(i,j) for j in range (n)] for i in range (n)]
  # Matriz aumentada (X|In)
  XI = [RX + RI for (RX,RI) in zip(X,In)]

  for i in range (n): # Columna
    for j in range (n): # Fila
      Xji = XI[j][i]
      if j != i and Xji != 0:
        if XI[i][i] == 0:
          XI[i] = [xi + xn for (xi,xn) in zip (XI[i],XI[n-1])]
        Xii = XI[i][i]
        XI[j] = [-r * Xii/Xji for r in XI[j]]
        XI[j] = [xj + xi for (xj,xi) in zip (XI[j],XI[i])]
  for i in range (n):
    xii = XI[i][i]
    XI[i] = [xii * r for r in XI[i]]
  return [R[n:] for R in XI]

# 2, Punto 1.

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

# 2, Punto 2.

AB = prod_matrices (A,B,n)
print ("\nA.B :")
imprime_matriz (AB,n)

# 2, Punto 3.

A_inv = inv_matriz (A,n)
print ("\nA^{-1} :")
imprime_matriz (A_inv, n)

print ("\n A.A^{-1} :")
imprime_matriz (prod_matrices(A,A_inv,n),n)
print ("\n A^{-1}.A :")
imprime_matriz (prod_matrices(A_inv,A,n),n)

# 2, Punto 4.

Z = [[randint (0,1) for _ in range (n)] for _ in range (n)]
for i in range (n):
  for j in range (n):
    if Z[i][j] == 0:
      Z[i][j] = False
    else:
      Z[i][j] = True
C = [[0 for _ in range (n)] for _ in range (n)]
for i in range (n):
  for j in range (n):
    if C[i][j]:
      C[i][j] = A[i][j]
    else:
      C[i][j] = B[i][j]
print ("\nC :")
imprime_matriz (C,n)
for i in range (n):
  for j in range (n):
    if C[i][j] < -5:
      C[i][j] = 0
print ("\nC :")
imprime_matriz (C,n)
