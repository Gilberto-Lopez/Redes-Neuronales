import numpy as np

### Descenso por el gradiente para mínimos cuadrados.

# w es el vector con los pesos w = [w0 ... wn]^t
# X son los datos: X = [X^1 ... X^m]^t donde X^i es un ejemplar
# de dimensión n (aumentado), ie, X^i = [1 X^i_1 ... X^i_n].
# y es el vector con los resultados y = [y^1 ... y^m]^t donde
# y^i es el resultado del ejemplar X^i

def h (w,X):
  return np.dot (X,w)

def J (w,X,y):
  return (np.linalg.norm (h (w,X) - y)**2) /2

def descenso_gradiente (J,NJ,X,y,w,h,e=0.01,alpha=0.2,it=1000)
  # Punto inicial: origen
  P = np.copy (w)
  j = 0
  while J(w,X,y) >= e and j < it:
    j += 1
    H = h (w,X) - y
    P = P - alpha * np.dot(np.transpose(X[:,j]),H)
  return P
