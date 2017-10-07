# Para calcular n-derivada de una función en un punto
from scipy import misc
from random import randint,random
import matplotlib.pyplot as plt
import numpy as np

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

class Individuo (object):
  """Representa un individuo, una solución del problema. La unidad básica de los
  algoritmos genéticos.
  """

  def __init__ (self, tam):
    """Crea un Individuo cuya representación en bits tiene tamaño TAM.
    Parámetros
    ----------
    tam : Int
      El tamaño de la representación.
    """
    # El cromosoma
    self.crom = [randint(0,1) for _ in range(tam)]
    # El fitness del individuo
    self.fit = 0

  def _valor (self):
    # El valor que representa el individuo
    # Se implementa por motivos prácticos
    val = 0
    tam = len(self.crom)
    for i in range(tam):
      val += self.crom[i]*(2**(tam-i-1))
    return val

  @staticmethod
  def mutacion (individuo, p):
    """Genera una mutación en INDIVIDUO con probabilidad P.
    Parámetros
    ----------
    individuo : Individuo
      El individuo que sufrirá la mutación.
    p : float
      La probabilidad de mutación.
    Regresa
    ----------
    n : Individuo
      El individuo resultante de generar una mutación en INDIVIDUO.
    """
    n = Individuo (len(individuo.crom))
    n.crom = individuo.crom[:]
    for i in range(len(individuo.crom)):
      if random() <= p:
        n.crom[i] = 1 - n.crom[i]
    return n

  @staticmethod
  def cruce (individuo1, individuo2):
    """Cruza dos individuos para generar dos nuevos.
    Parámetros
    ----------
    individuo1, individuo2 : Individuo
      Los individuos a cruzar.
    Regresa
    ----------
    hijo1, hijo2 : Individuo
      Los individuos resultantes del cruce.
    """
    tam = len(individuo1.crom)
    s = randint(0, tam-1)
    hijo1 = Individuo(tam)
    hijo2 = Individuo(tam)
    hijo1.crom = individuo1.crom[:s] + individuo2.crom[s:]
    hijo2.crom = individuo2.crom[:s] + individuo1.crom[s:]
    return (hijo1,hijo2)

  def __str__ (self):
    # Representación del individuo.
    return ('[ '
      +('{} '*len(self.crom)).format(*self.crom)
      +'] Valor: {}\tFitness: {}'.format(self._valor(),self.fit))

  def __repr__ (self):
    # Representación del individuo.
    return self.__str__()

class Poblacion (object):
  """Representa una población de individuos, una generación de la ejecución del
  algoritmo.
  """

  def __init__ (self, tam_ind, tam, fitness, vacia = False):
    """Crea una nueva población con una cantidad fija de individuos (con un
    tamaño en su representación fija), con una función de fitness fija. La
    población puede estar vacía dependiendo el flag VACIA.
    Parámetros:
    -----------
    tam_ind : Int
      El tamaño en la representación (cromosoma) de los individuos.
    tam : Int
      El tamaño máximo de la población.
    fitness : Individuo -> Int
      La función de fitness para evaluar a los individuos.
    vacia : Bool
      Flag que determina si la población debe ser vacía (True). Por defecto
      su valor es False.
    """
    self._ind = tam_ind
    self._tam = tam
    # Población
    self.pob = [Individuo(tam_ind) for _ in range(tam)] if not vacia else []
    self.fit_f = fitness
    # Total fitness
    # Se tiene aquí por motivos prácticos
    self._total_fit = 0

  def calcula_fitness (self):
    """Calcula el fitness de todos los individuos en la población.
    """
    for individuo in self.pob:
      s = self.fit_f(individuo)
      individuo.fit = s
      # Aprovechamos para calcular el fitness total
      self._total_fit += s

  def seleccion (self):
    """Operador de selección. Regresa un individuo de la población
    aleatoriamente. La probabilidad de selección depende del fitness de los
    individuos, individuos mejor adaptados tiene mayor probabilidad de ser
    seleccionados.
    Regresa:
    -----------
    individuo : Individuo
    """
    while True:
      for ind in self.pob:
        if random() <= ind.fit / float(self._total_fit):
          return ind

  def agrega (self, individuo):
    """Agrega un nuevo individuo a la población actual.
    La población no debe exceder el máximo de individuos y la representación
    del nuevo debe coincidar con los demás miembros de la población.
    Parámetros:
    -----------
    individuo : Individuo
      El nuevo individuo.
    """
    if len(self.pob) < self._tam and len(individuo.crom) == self._ind:
      self.pob.append(individuo)

  @staticmethod
  def mejor_individuo (poblacion):
    """Devuelve al mejor individuo de la población dada, esto es, el individuo
    con el fitness más alto.
    Parámetros
    ----------
    poblacion : Poblacion
      La población.
    Regresa
    ----------
    m : Individuo
      El mejor individuo de la población.
    """
    b = -1
    m = None
    for ind in poblacion.pob:
      if ind.fit > b:
        b = ind.fit
        m = ind
    return m

  def __str__ (self):
    # Representación de la población.
    return ('{}\n'*self._tam).format(*self.pob)

  def __repr__ (self):
    # Representación de la población.
    return self.__str__()  

def estrategia_evolucionaria(sigma, iteraciones, c, x0):
  """
  """
  # Punto inicial
  X = x0
  # funcion(X), su fitness
  p = funcion(X)
  # Soluciones encontradas y su valor
  xs = [X]
  ys = [p]
  # Mutaciones realizadas (1 para exitosas, 0 para no exitosas)
  # Las mutaciones exitosas tienen mejor fitness que la solución actual
  M = []
  # Relación mutaciones exitosas sobre mutaciones totales
  ps = 0
  # Sigma la desviación estándar de la distribución normal
  sig = sigma
  for _ in range(iteraciones):
    # Generamos una nueva solución a partir de una mutación sobre la actual
    X_ = X + sig*np.random.normal()
    # Calculamos su fitness
    p_ = funcion(X_)
    if p_ < p:
      # La nueva solución es mejor
      X = X_
      p = p_
      xs.append(X)
      ys.append(p)
      M.append(1)
    else:
      M.append(0)
    # Actualizamos ps cada n iteraciones donde n = 1 es el número de variables
    # de funcion
    # Nos fijamos en las últimas (a lo más) 10n mutaciones
    M_10 = M[-10:]
    ps = sum(M_10)/float(len(M_10))
    # Ajustamos sigma
    # 1/5 success rule: Se maximiza la velocidad del progreso en la optimización
    # cuando ~1/5 de las mutaciones resultan existosas
    if ps < 1.0/5:
      sig = sig*c
    elif ps > 1.0/5:
      sig = sig/c
  # Graficación de los puntos (x,f(x))
  plt.scatter(xs,ys)
  plt.plot(xs,ys)
  print("Mínimo local en (x,y): ({}, {})".format(xs[-1], ys[-1]))


# ENTRADA DEL PROGRAMA


if __name__ == '__main__':
# 1. DESCENSO POR EL GRADIENTE


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


# 2. ALGORITMO GENÉTICO CLÁSICO


  # Parámetros del problema
  # Tamaño de la rep. de individuo
  tam_ind = 5
  # Tamaño de la población
  tam_pob = 6
  # Probabilidad de cruce
  #pc = 0.8
  # Probabilidad de mutación
  pm = 0.5
  # Generaiones a ejecutar el algoritmo
  generaciones = [5,10,50,100]
  # Función a maximizar
  f = lambda x: -funcion(x)
  # Función de fitness
  # f es un polinomio con valores máximo positivos, nos restringimos a los
  # valores no negativos, 0.01 para evitar que el fitness total llegue a ser 0
  fitness = lambda ind: max(0.01,f(ind._valor()))

  # Creamos la población inicial y calculamos el fitness de sus individuos
  for g in generaciones:
    # Solo nos interesa graficar en el intervalo [2,18]
    plt.gca().set_ylim([0,700])
    plt.gca().set_xlim([2,18])
    # Los mejores individuos en cada generación
    xs = []
    # Población inicial
    P0 = Poblacion(tam_ind, tam_pob, fitness)
    P0.calcula_fitness()
    P = P0
    for _ in range(g):
      # Siguiente generación
      Q = Poblacion(tam_ind, tam_pob, fitness, vacia=True)
      # Elitismo(1): conservamos al mejor individuo
      mejor = Poblacion.mejor_individuo (P)
      xs.append(mejor._valor())
      Q.agrega(mejor)
      # En cada cruce se crean dos individuos, así que se necesitan tam_pob//2
      # iteraciones.
      for _ in range(tam_pob//2):
        padre1 = P.seleccion()
        padre2 = P.seleccion()
        # Los dos nuevos individuos
        (hijo1,hijo2) = Individuo.cruce(padre1, padre2)
        hijo1 = Individuo.mutacion(hijo1, pm)
        hijo2 = Individuo.mutacion(hijo2, pm)
        #Q.agrega(max([hijo1,hijo2],key=lambda x: x.fit))
        Q.agrega(hijo1)
        Q.agrega(hijo2)
      P = Q
      P.calcula_fitness()
    print('Generaciones:\t{}'.format(g))
    print('Mejor Individuo:\t{}'.format(Poblacion.mejor_individuo (P)))
    
    ys = [f(s) for s in xs]
    plt.plot(x, f(x), 'r-')
    plt.scatter(xs, ys)
    plt.plot(xs, ys)
    plt.title('Algoritmo genético clásico.\nMáximo # de generaciones: {}'.format(g))
    plt.legend(['f (x)','Soluciones'])
    plt.show()


# ESTRATEGIA EVOLUCIONARIA (1+1)


  # Graficar la función
  plt.plot(x, y,'r-')

  # Valor recomendado para el ajuste de sigma
  c = 0.817
  # Sigma la desviación estándar de la distribución normal
  # Para sigma pequeña tiende a quedarse atorado en mínimos locales.
  sigma = 10
  # El número de iteraciones y el punto inicial x0 son los mismos que los usados
  # en el descenso por el gradiente.
  estrategia_evolucionaria(sigma, iteraciones, c, x0)

  plt.title('Estrategia evolucionaria 1+1.')
  plt.grid()
  plt.show()
