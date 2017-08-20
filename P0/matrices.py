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

def menor (X,i,j):
  """Obtiene el menor_{i,j} de la matriz X.
  Parámetros
  ----------
  X : list (list (int))
     Matriz.
  i : int
  j : int
     El menor a obtener.
  Regresa
  -------
  M : list (list (int))
     El menor_{i,j}
  """
  return [S[:j] + S[j+1:] for S in (X[:i]+X[i+1:])]

def determinante (X,n):
  """Calcula el determinante de la matriz X.
  Parámetros
  ----------
  X : list (list (int))
     Matriz.
  n : int
     Tamaño de la matriz.
  Regresa
  -------
  d : int
     El determinante de la matriz.
  """
  if n == 2:
    return X[0][0]*X[1][1] - X[1][0]*X[0][1]
  m = -1
  det = 0
  for i in range (n):
    m *= -1
    if X[0][i] != 0:
      det += m*X[0][i]*determinante (menor (X,0,i), n-1)
  return det

def adj (X,n):
  """Calcula la matriz de adjuntos de X.
  Parámetros
  ----------
  X : list (list (int))
     Matriz.
  n : int
     Tamaño de la matriz.
  Regresa
  -------
  Z : list (list (int))
     La matriz de ajuntos de X.
  """
  Adj = [[0 for _ in range (n)] for _ in range (n)]
  for i in range (n):
    for j in range (n):
      s = (-1) ** (i+j % 2)
      Adj[j][i] = s*determinante (menor (X,i,j), n-1)
  return Adj

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
  Y : list (list (int))
      Y = X^{-1}
  """
  det = determinante (X,n)
  adjunta = adj (X,n)
  for i in range (n):
    for j in range (n):
      adjunta[i][j] /= det
  return adjunta

#  # Delta de Kronecker
#  delta = lambda s, t: 1 if s == t else 0
#  # Identidad de tamaño n
#  In = [[delta(i,j) for j in range (n)] for i in range (n)]
#  # Matriz aumentada (X|In)
#  XI = [RX + RI for (RX,RI) in zip(X,In)]
#  print ("\n")
#  imprime_matriz(XI,2*n)
#
#  for i in range (n): # Columna
#    for j in range (n): # Fila
#      Xji = XI[j][i]
#      if j != i and Xji != 0:
#        k = i+1
#        while X[i][i] == 0 and k < n:
#          XI[i] = [xi + xk for (xi,xk) in zip (XI[i],XI[k])]
#          k += 1
#        Xii = XI[i][i]
#        if Xii == 0:
#          raise Exception ("Matriz singular.")
#        XI[j] = [-Xii*r/Xji for r in XI[j]]
#        XI[j] = [rj + ri for (rj,ri) in zip(XI[j],XI[i])]
#  for i in range (n):
#    xii = XI[i][i]
#    XI[i] = [r/xii for r in XI[i]]
#  imprime_matriz (XI,2*n)
#  print("\n\n--------------------")
#  return [R[n:] for R in XI]

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
