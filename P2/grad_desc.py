# Para calcular n-derivada de una función en un punto
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random

def funcion(x):
    """
    La función que se va a minimizar.
    Parámetros
    ----------
    x : float
        Valor de x a evaluar en la función.
    """
    return ((x-10)**4) - (50 * (x-10)**2) + (8*x) - 100

def descenso_gradiente(alpha, iteraciones, precision, x0):
    """
    Descenso del gradiente.
    Encuentra el mínimo de una función.
    Parámetros
    ----------
    alpha : float
        La constante de aprendizaje.
    iteraciones : int
        El número de iteraciones.
    precision : float
        Toleracia para F(x).
    x0 : int
        Punto inicial para el descenso.
    """
    # Guardar los puntos x.
    xs = []
    # Guardar los puntos f(x).
    ys = []
    y0 = funcion(x0)
    xs.append(x0)
    ys.append(funcion(x0))

    cond = precision + 10.0 # empezar con cond mayor que precision
    iteracion = 0 
    tmp_y = y0
    
    while cond > precision and iteracion < iteraciones:
        x0 = x0 - alpha * misc.derivative(funcion, x0)
        y0 = funcion(x0)
        iteracion += 1
        cond = abs(tmp_y - y0)
        tmp_y = y0
        #print(x0,y0,cond)
        xs.append(x0)
        ys.append(y0)
    # Graficar los puntos (x,f(x))
    plt.scatter(xs, ys)
    plt.plot(xs,ys)
    print("Mínimo local en (x,y): ({}, {})".format(x0, y0))

if __name__ == '__main__':
    # Graficar la función 
    x1 = 2.0
    x2 = 18.0
    x = np.arange(x1, x2, 0.01)
    y = funcion(x)
    plt.plot(x, y,'r-')

    alpha = 0.01 # constante de aprendizaje
    iteraciones = 200 # número máximo de iteraciones
    precision = 0.0001 # condición para terminar
    x0 = random.randint(x1,x2)    # x0 punto inicial
    descenso_gradiente(alpha,iteraciones,precision,x0)

    plt.title("Descenso del gradiente")
    plt.grid()
    plt.show()