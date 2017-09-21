from random import randint,random

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

  def valor (self):
    # El valor que representa el individuo
    # Se implementa por motivos prácticos
    val = 0
    tam = len(self.crom)
    for i in range(tam):
      val += self.crom[i]*(2**(tam-i-1))
    return val

  def fitness (self):
    """Regresa el fitness o aptitud del individuo.
    Regresa
    ----------
    fitness : int
    """
    return self.fit

  @staticmethod
  def mutacion (individuo, p):
    """Genera una mutación en INDIVIDUO con probabilidad P.
    Parámetros
    ----------
    individuo : Individuo
      El individuo que sufrirá la mutación.
    p : float
      La probabilidad de mutación.
    """
    for i in range(len(individuo.crom)):
      if random() <= p:
        individuo.crom[i] = 1 - individuo.crom[i]

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
    for i in range(s):
      hijo1.crom[i] = individuo1.crom[i]
      hijo2.crom[i] = individuo2.crom[i]
    for i in range(s,tam):
      hijo1.crom[i] = individuo2.crom[i]
      hijo2.crom[i] = individuo1.crom[i]
    return (hijo1,hijo2)

  def __str__ (self):
    # Representación del individuo.
    return ('[ '
      +('{} '*len(self.crom)).format(*self.crom)
      +'] Valor: {}\tFitness: {}'.format(self.valor(),self.fit))

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
      # Aprovechamos para calculat el total fitness
      self._total_fit += s

  def seleccion (self):
    """
    """
    #n = randint(0,self._total_fit)
    return self.pob[randint(0,len(self.pob)-1)]
    return None

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

  def __str__ (self):
    # Representación de la población.
    return ('{}\n'*self._tam).format(*self.pob)

  def __repr__ (self):
    # Representación de la población.
    return self.__str__()  

if __name__ == '__main__':
  # Parámetros del problema
  # Tamaño de la rep. de individuo
  tam_ind = 5
  # Tamaño de la población
  tam_pob = 6
  # Probabilidad de cruce
  pc = 0.8
  # Probabilidad de mutación
  pm = 0.5
  # Generaiones a ejecutar el algoritmo
  generaciones = [5,10,50,100]
  # Función a maximizar
  f = lambda x: 100 - (x - 10)**4 + 50*(x - 10)**2 - 8*x
  # Función de fitness
  fitness = lambda ind: f(ind.valor())

  # Creamos la población inicial y calculamos el fitness de sus individuos
  P0 = Poblacion(tam_ind, tam_pob, fitness)
  P0.calcula_fitness()
  P = P0
  for i in range(1,generaciones[-1]+1):
    Q = Poblacion(tam_ind, tam_pob, fitness, vacia=True)
    for _ in range(tam_pob//2):
      padre1 = P.seleccion()
      padre2 = P.seleccion()
      (hijo1,hijo2) = Individuo.cruce(padre1, padre2)
      Individuo.mutacion(hijo1, pm)
      Individuo.mutacion(hijo2, pm)
      Q.agrega(hijo1)
      Q.agrega(hijo2)
    P = Q
    P.calcula_fitness()
    if i in generaciones:
      print('Generación: {}\n\tPoblación:\n{}'.format(i,P))
  print('Población resultante:\n{}'.format(P))
