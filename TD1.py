#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math as m

#Question 1.1 
# Déclaration vecteur
vec = np.array([0.25, -0.13, 0.61, 1])

print("Vec = " + str(vec))
# Déclaration matrice
mat1 = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

mat2 = np.ones((4,4))

print(mat1)
print(mat2)

#Question 1.2
# Multiplication vecteur matrice
def multi(vecteur, matrice) :
	# Vecteur au carré
	vecteur_carre = vecteur ** 2

	# Multiplication vecteur - matrice
	return np.dot(vecteur_carre, matrice)

print(multi(vec,mat1))
print(multi(vec,mat2))
	
c = multi(vec,mat2)


#Question 1.3 
# fonction polynome
def polynom(matrice) :
	# Pour selectionner la colonne i : matrice[,:i]
	return np.array([3*matrice[:,0]+2*matrice[:,1]-5*matrice[:,2]]+ matrice[:,3])

print polynom(mat2)
	

#Question 2.1
# différence entre deux vecteurs
# Root mean square error
def rmse(vecteur1, vecteur2) :
	diff = vecteur1 - vecteur2
	# différence de deux vecteurs au carré
	return diff ** 2

v1 = np.array([1,2,3,4])
v2 = np.array([6,5,10,23])
print(rmse(v2,v1))


#Question 2.2
# Tangeante hyperbolique
def tanh(x) :
	return (1-m.exp(-2*x))/(1+m.exp(-2*x))

print(tanh(1))

# Fonction sigmoide
def sigmoid(x) : 
	return 1/(1 + m.exp(-x)) 	 

print(sigmoid(0))

# Dérivée de tangeante hyperbolique
def tanh_prime(x):
	return 1 - tanh(x) ** 2 

print(tanh_prime(1))

# Dérivée de sigmoide
def sigmoid_prime(x):
	return sigmoid(x)*(1 - sigmoid(x))


print(sigmoid_prime(0))

#Question 2.3 
# fonction de regression linéaire
def linearregression(X, Y, epsilon, nbiteration) : 
	W = np.random.rand(1, 4)
	for i in range(nbiteration) :
		W = W + 2 * X*(X * W -Y) * epsilon
	return W

print("linearregression = " + str(linearregression(vec, mat1, 0.0000001, 1000)))


# Question 3
mat=np.loadtxt(open("winequality-red.csv","rb"),delimiter=";",skiprows=1)
print(mat)



 

	

		 

