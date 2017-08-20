# Para vectores y matrices
import numpy as np
# Para graficar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 1, Operaciones con vectores
ortonormalizar()