# Para vectores y matrices
import numpy as np
# Para graficar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Para generar números aleatorios.
from random import randint

### 1. Operaciones con vectores.

def determinante(a):
  """
  Regresa el determinante de a.
  Parámetros
  ----------
  a : array(array(float))
    Matriz para calcular el determinante.
  Regresa
  -------
  det(a) : float
  """
  return np.linalg.det(a)

def producto_punto(u, v):
  """
  Regresa el producto punto entre dos vectores.
  Parámetros
  ----------
  u : array(float)
    Primer vector.
  v : array(float)
    Segundo vector.
  Regresa
  -------
  u.v : float
  """
  return np.vdot(u,v)

def ortonormalizar():
  # Dimensión del espacio.
  n = 3
  a = []
  # Pide en la línea de comandos los 3 vectores que serán ortonormalizados.
  print("Ingresa los valores de cada vector, separados por un espacio, p. e. >v1: x0 x1 x2")
  for i in range(n):
    v = input("v"+str(i)+": ").split()
    v = [float(num) for num in v]
    a.append(v)
  a = np.array(a)
  # Determinante para ver si es linealmente independiente.
  det = determinante(a)
  print("Determinante: " + str(det))
  
  # Producto punto entre todos los vectores para verificar si son 
  # ortogonales.
  i = 0
  ortogonal = True
  while(i < len(a)):
    j = i+1
    while (j < len(a)):
      pp = producto_punto(a[i], a[j])
      print(str(a[i])+"."+str(a[j])+"="+str(pp))
      if pp != 0:
        print("El conjunto no es ortogonal")
        ortogonal = False
      j+=1
    i+=1

  if (det != 0 and ortogonal == True):
    print("El conjunto es linealmente independiente y es ortogonal.")

  # Norma de cada vector para ver si es una base ortogonal
  norma0 = np.linalg.norm(a[0])
  norma1 = np.linalg.norm(a[1])
  norma2 = np.linalg.norm(a[2])
  print("Norma v0: " + str(norma0))
  print("Norma v1: " + str(norma1))
  print("Norma v2: " + str(norma2))

  # Base ortonormal.
  ortonormalizados = []
  if (norma0 == norma1 and norma1 == norma2):
    print("Son vectores unitarios.")
    ortonormalizados = a
  else:
    print("No son vectores unitarios.")
    # Obtener primer vector unitario u0.
    u0 = a[0]/norma0
    print("Vector unitario u0: " + str(u0))
    ortonormalizados.append(u0)

    # Otener vector u1 ortogonal a u0.
    u1 = a[1]-((producto_punto(a[1], u0))*u0)

    # Normalizar u1.
    u1 = u1/np.linalg.norm(u1)
    print("Vector unitario u1: " + str(u1))
    ortonormalizados.append(u1)

    # Obtener un vector u2 ortogonal a u0 y u1.
    u2 = a[2]-((producto_punto(a[2], u0))*u0)-((producto_punto(a[2], u1))*u1)

    # Normalizar u2.
    u2 = u2/np.linalg.norm(u2)
    print("Vector unitario u2: " + str(u2))
    ortonormalizados.append(u2)

  # Graficar vectores.
  fig = plt.figure()
  ax1 = fig.add_subplot(111, projection='3d')

  graficar(ax1, a, 'red')
  graficar(ax1, ortonormalizados, 'blue')

  ax1.set_xlabel('eje x')
  ax1.set_ylabel('eje y')
  ax1.set_zlabel('eje z')

  plt.show()

def graficar(ax1, a, col):
  """
  Grafica los vectores.
  Parámetros
  ----------
  ax1: matplotlib.axes._subplots.Axes3DSubplot
    Eje de la gráfica.
  a: array(array(float))
    Vectores a graficar.
  col: str
    Color para el trazado.
  """
  for i in range(len(a)):
    v = a[i]
    x = [0, v[0]]
    y = [0, v[1]]
    z = [0, v[2]]
    ax1.plot(x, y, z, color=col)

### 2. Operaciones con matrices.

def imprime_matriz (X):
  """Imprime una matriz en pantalla.
  Parámetros
  ----------
  X : list (list (int))
     Matriz a imprimir.
  """
  for row in X:
    print (("{}\t"*len(row)).format(*row))

def prod_matrices (X,Y):
  """Multiplica dos matrices de n.m y m.p.
  Parámetros
  ----------
  X : list (list (int))
  Y : list (list (int))
     Matrices a multiplicar.
  Regresa
  -------
  Z : list (list (int))
     Z = X.Y
  """
  n1 = len (X)
  n2 = len (X[0])
  m1 = len (Y)
  m2 = len (Y[0])
  if n2 != m1:
    raise Exception ("Las matrices no se pueden multiplicar.")
  Z = [[0]*m2 for _ in range (n1)]
  for i in range (n1):
    for j in range (m2):
      Zij = 0
      for k in range (m1):
        Zij += X[i][k] * Y[k][j]
      Z[i][j] = Zij
  return Z

def inv_matriz (X):
  """Invierte la matriz cuadrada X.
  La invesrión se hace con el método de Gauss-Jordan.
  Parámetros
  ----------
  X : list (list (int))
     Matriz a invertir.
  Regresa
  -------
  Y : list (list (int))
      Y = X^{-1}
  """
  n = len (X)
  if n != len (X[0]):
    raise Exception ("Matriz no cuadrada, no es invertible.")

  # Delta de Kronecker
  delta = lambda s, t: 1 if s == t else 0
  # Identidad de tamaño n
  In = [[delta(i,j) for j in range (n)] for i in range (n)]
  # XI es la matriz aumentada (X|In)
  XI = [RX + RI for (RX,RI) in zip(X,In)]
  # Ponemos 0s duera de la diagonal de X.
  for i in range (n): # Columna
    for j in range (n): # Fila
      Xji = XI[j][i]
      if j != i and Xji != 0:
        k = i+1
        while X[i][i] == 0 and k < n:
          XI[i] = [xi + xk for (xi,xk) in zip (XI[i],XI[k])]
          k += 1
        Xii = XI[i][i]
        if Xii == 0:
          raise Exception ("Matriz singular.")
        XI[j] = [-Xii*(r/Xji) for r in XI[j]]
        XI[j] = [rj + ri for (rj,ri) in zip(XI[j],XI[i])]
  # Ponemos 1s en la diagonal de  XI.
  for i in range (n):
    xii = XI[i][i]
    XI[i] = [r/xii for r in XI[i]]
  # XI ahora es (In|X^{-1})
  return [R[n:] for R in XI]

if __name__ == '__main__':
  # 1, Operaciones con vectores
  ortonormalizar()
  # Cierre la ventana con la graficación de los vectores
  # para proceder en la ejecución del programa.

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

  print ("\nA :")
  imprime_matriz (A)
  print ("\nB :")
  imprime_matriz (B)

  # 2, Punto 2.

  AB = prod_matrices (A,B)
  print ("\nA.B :")
  imprime_matriz (AB)

  # 2, Punto 3.

  A_inv = inv_matriz (A)
  print ("\nA^{-1} :")
  imprime_matriz (A_inv)

  print ("\n A.A^{-1} :")
  imprime_matriz (prod_matrices(A,A_inv))
  print ("\n A^{-1}.A :")
  imprime_matriz (prod_matrices(A_inv,A))

  # 2, Punto 4.

  # Matriz de 0s y 1s.
  Z = [[randint (0,1) for _ in range (n)] for _ in range (n)]
  for i in range (n):
    for j in range (n):
      if Z[i][j] == 0:
        Z[i][j] = False
      else:
        Z[i][j] = True
  # Valores True y False
  C = [[0 for _ in range (n)] for _ in range (n)]
  for i in range (n):
    for j in range (n):
      if C[i][j]:
        C[i][j] = A[i][j]
      else:
        C[i][j] = B[i][j]
  print ("\nC :")
  imprime_matriz (C)
  # Cambio de los valores.
  for i in range (n):
    for j in range (n):
      if C[i][j] < -5:
        C[i][j] = 0
  print ("\nC :")
  imprime_matriz (C)
