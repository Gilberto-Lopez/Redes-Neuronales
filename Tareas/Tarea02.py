import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles

def ejercicio1 ():
  ### EJERCICIO 1:
  ### SEPARACIÓN DE PATRONES

  # XOR
  X = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]])
  Y = np.array([0,1,1,0])
  xx, yy = np.meshgrid(np.arange(-.05, 1.05, .02), np.arange(-.05, 1.05, .02))
  nn = MLPClassifier(solver='lbfgs',activation='logistic',
    hidden_layer_sizes=(3,),alpha=0.001)
  nn.fit(X,Y)
  Z = nn.predict(np.c_[xx.ravel(),yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
  plt.scatter(X[:,0], X[:,1], c=Y, edgecolors='w')
  plt.title('Red neuronal\nSeparación de patrones XOR')
  plt.show()

def ejercicio2 ():
  ### EJERCICIO 2:
  ### SOBREENTRENAMIENTO
  
  # Dataset original
  X, Y = make_gaussian_quantiles (n_samples = 200, n_features = 4, n_classes = 3)
  # Ignoramos atributos, lo que provoca que los datos se vean (más) 'traslapados'
  X = np.column_stack ((X[:,0],X[:,2]))
  plt.scatter (X[:,0],X[:,1],marker='o',c=Y,edgecolor='k')
  plt.title('Dataset original')
  plt.show()
  # Obtenemos los intervalos donde yacen los puntos del dataset original (X)
  x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
  y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
  # Separamos el dataset para obtener un conjunto de entrenamiento
  x_train, x_test, y_train, y_test = train_test_split (X,Y,test_size=0.3)
  plt.scatter (x_train[:,0],x_train[:,1],marker='o',c=y_train,edgecolor='k')
  plt.title('Dataset de entrenamiento (70 %)')
  plt.show()
  # Creamos un grid para los decision plot de las redes neuronales
  xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
  # Creamos redes neuronales con una capa oculta y
  # distinto número de neuronas en esta capa.
  sizel = [(6,),
           (16,),
           (25,),
           (50,),
           (100,),
           (25,18),
           (50,50),
           (100,100)]
  for size in sizel:
    params = {'solver':'lbfgs',
    'activation':'tanh',
    'hidden_layer_sizes':size}
    nn = MLPClassifier (**params)
    # Entrenamos
    nn.fit(x_train,y_train)
    # Vemos que tan precisa es la red neuronal
    score = nn.score(x_test,y_test)
    # Decision plot de la red
    Z = nn.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Clasificación del conjunto de entrenamiento
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(x_train[:,0], x_train[:,1], c=y_train, edgecolors='k')
    plt.title('Red neuronal con {} neuronas\nConjunto de entrenamiento'
      .format(nn.hidden_layer_sizes))
    plt.show()
    # Clasificación del conjunto de verificación con el score asociado
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(x_test[:,0], x_test[:,1], c=y_test, edgecolors='k')
    plt.title('Red neuronal con {} neuronas\nConjunto de verificación'
      .format(nn.hidden_layer_sizes))
    plt.text(x_max - .3, y_min + .3, 'Score: {0:.2f}'.format(score).lstrip('0'),
      size=10, horizontalalignment='right')
    plt.show()

def ejercicio4 ():
  ### EJERCICIO 4:
  ### APROXIMACIÓN DE FUNCIONES

  # Función a aproximar
  f = lambda x: 0.2+0.4*x**2+0.3*x*np.sin(15*x)+0.05*np.cos(50*x)
  xx = np.linspace (0,1,500)
  yy = f (xx)
  plt.plot (xx,yy)
  plt.title ('Función a aproximar')
  plt.legend (['f(x)'])
  plt.show ()
  xx = xx.reshape(-1,1)
  # Parámetros de la red
  params = {'solver':'lbfgs',
    # nonconstant, bounded, and monotonically-increasing continuous function
    # tanh cumple lo que pide el teorema de aproximación universal
    'activation':'tanh',
    'hidden_layer_sizes':(50,)}
  # MLPRegressor usa la función identidad como activación de la neurona de salida
  nn = MLPRegressor (**params)
  nn.fit(xx,yy)
  # Graficamos la función y la predicción de la red
  plt.plot(xx,yy)
  plt.plot(xx,nn.predict(xx))
  plt.legend ('Aproximación de la función')
  plt.legend (['f(x)','Red Neuronal'])
  plt.show()

def ejercicio5 (e):
  ### EJERCICIO 5:
  ### ANÁLISIS DE LAS MATRICES DE PESOS

  if e == 1:
    ### Ejercicio 1
    ### SEPARACIÓN DE PATRONES

    # XOR
    X = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
    Y = np.array([0,1,1,0])
    xx, yy = np.meshgrid(np.arange(-.05, 1.05, .02), np.arange(-.05, 1.05, .02))
    nn = MLPClassifier(solver='lbfgs',activation='logistic',
      hidden_layer_sizes=(3,),alpha=0.001)
    # 1 iteración para ver los pesos de la red tras un fit.
    nn.max_iter = 1
    nn.fit(X,Y)
    # Obtenemos la lista de matrices de pesos
    print('Antes del entrenamiento')
    W = nn.coefs_
    for w in W:
      print(w)
      print('svd: {}\n'.format(np.linalg.svd(w,compute_uv=False)))
    # 200 iteraciones (valor default)  
    nn.max_iter = 200
    nn.fit(X,Y)
    # Obtenemos la lista de matrices de pesos después de haber entrenado la red
    print('Después del entrenamiento')
    W = nn.coefs_
    for w in W:
      print(w)
      print('svd: {}\n'.format(np.linalg.svd(w,compute_uv=False)))
  elif e == 4:
    ### EJERCICIO 4:
    ### APROXIMACIÓN DE FUNCIONES

    f = lambda x: 0.2+0.4*x**2+0.3*x*np.sin(15*x)+0.05*np.cos(50*x)
    xx = np.linspace (0,1,500)
    yy = f (xx)
    xx = xx.reshape(-1,1)
    params = {'solver':'lbfgs',
      'activation':'tanh',
      'hidden_layer_sizes':(50,)}
    nn = MLPRegressor (**params)
    # 1 iteración para ver los pesos de la red tras un fit.
    nn.max_iter = 1
    nn.fit(xx,yy)
    # Obtenemos la lista de matrices de pesos
    print('Antes del entrenamiento')
    W = nn.coefs_
    for w in W:
      print(w)
      print('svd: {}\n'.format(np.linalg.svd(w,compute_uv=False)))
    # 200 iteraciones (valor default)  
    nn.max_iter = 200
    nn.fit(xx,yy)
    # Obtenemos la lista de matrices de pesos después de haber entrenado la red
    print('Después del entrenamiento')
    W = nn.coefs_
    for w in W:
      print(w)
      print('svd: {}\n'.format(np.linalg.svd(w,compute_uv=False)))
  else:
    pass

if __name__ == '__main__':
  ejercicio1 ()
  ejercicio5 (1)
  ejercicio2 ()
  ejercicio4 ()
  ejercicio5 (4)
