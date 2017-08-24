import numpy as np
import random

### Descenso por el gradiente para mínimos cuadrados.

# w es el vector con los pesos w = [w0 ... wn]^t
# X son los datos: X = [X^1 ... X^m]^t donde X^i es un ejemplar
# de dimensión n (aumentado), ie, X^i = [1 X^i_1 ... X^i_n].
# y es el vector con los resultados y = [y^1 ... y^m]^t donde
# y^i es el resultado del ejemplar X^i

def h_ls (w,X):
  return np.dot(X,w)

# J(w) = 0.5||Xw-y||^2

def J_ls (w,X,y,h):
  s = h(w,X)-y
  return np.dot(np.transpose(s),s)/2

# grad(J)(w) = X^t(Xw-y)

def GJ_ls (w,X,y,h):
  s = h(w,X)-y
  return np.dot(np.transpose (X),s)

def descenso_gradiente (J,GJ,X,y,w,h,e=0.1,alpha=0.2,it=1000):
  q = len(w)
  P = np.copy(w)
  k = 0
  while J(P,X,y,h) >= e and k < it:
    k += 1
    P = P - alpha*GJ(P,X,y,h)
  return P

if __name__ == '__main__':
  X = np.matrix([[1,1],[1,2],[1,3],[1,4]])
  y = np.matrix([[2],[3],[5],[7]])
  w = np.matrix([[random.uniform(-0.5,0.5)],[random.uniform(-0.5,0.5)]])
  descenso_gradiente(J_ls,GJ_ls,X,y,w,h_ls)
