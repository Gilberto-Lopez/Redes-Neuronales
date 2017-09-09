import numpy as np
import random

# w es el vector con los pesos w = [w0 ... wn]^t.
# X son los datos: X = [X^1 ... X^m]^t donde X^i es un ejemplar
# de dimensión n (aumentado), ie, X^i = [1 X^i_1 ... X^i_n]^t.
# y es el vector con los resultados y = [y^1 ... y^m]^t donde
# y^i es el resultado del ejemplar X^i.

### Descenso por el gradiente para mínimos cuadrados.

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

### Descenso por el gradiente para regresión logística
# w es el vector con los pesos w = [w0 ... wn c]^t.

def J_lr (w,X,y,h):
  C_ = 1 # parámetro
  c = w[-1,0]
  m = len(X)
  s = 0
  for i in range(m):
    s += np.log(1+np.exp(-y[i,0]*(c+np.dot(X[i,:],w[:-1,:]))))
  return np.dot(np.transpose(w),w)/2 + C_*s

def GJ_lr (w,X,y,h):
  C_ = 1 # parámetro
  c = w[-1,0]
  p = np.copy(w)
  n = len(w)
  m = len(X)
  q = np.matrix([0]*m).transpose()
  for i in range(m):
    q[i,0] = y[i,0]/(1+np.exp(y[i,0]*(c+np.dot(X[i,:],w[:-1,:]))))
  p[-1,0] = C_*sum(q)
  for j in range(n-1):
    p[j,0] = w[j,0] + C_*np.dot(np.transpose(X[:,j]),q)
  return p

def descenso_gradiente (J,GJ,X,y,w,h,e=0.1,alpha=0.02,it=1000):
  P = np.copy(w)
  k = 0
  while J(P,X,y,h) >= e and k < it:
    print("---------------")
    print(P)
    print(J(P,X,y,h))
    k += 1
    P = P - alpha*GJ(P,X,y,h)
  return P

if __name__ == '__main__':
  X = np.matrix([[1,1],[1,2],[1,3],[1,4]])
  y = np.matrix([[2],[3],[5],[7]])
  #w = np.matrix([[0],[0]])
  #w_f = descenso_gradiente(J_ls,GJ_ls,X,y,w,h_ls)
  #print(w_f)
  w =  np.matrix([[0],[0],[0]])
  w_g = descenso_gradiente(J_lr,GJ_lr,X,y,w,None)
  print(w_g)