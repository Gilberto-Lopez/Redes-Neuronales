from random import randint,random

class Individuo (object):
  """Representa un individuo, una solución del problema. La unidad básica de los
  algoritmos genéticos.
  """

  def __init__ (self, tam = 5):
    """Crea un Individuo cuya representación en bits tiene tamaño TAM.
    Parámetros
    ----------
    tam : Int
      El tamaño de la representación.
    """
    # El cromosoma
    self.crom = [randint(0,1) for _ in range(tam)]

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
    return self.crom.__str__()

  def __repr__ (self):
    # Representación del individuo.
    return self.__str__()

class Poblacion (object):
  """Representa una población de individuos, una generación de la ejecución del
  algoritmo.
  """

  def __init__ (self, tam = 6, pc = 0.8, pm = 0.5):
    self.pob = [Individuo() for _ in range(tam)]
    self.pc = pc
    self.pm = pm
