#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################
#
# Auteurs : CISSE Chafik, DALBERGUE Clément, KECHCHANY Hajar
# Version : 1.0.0
# Contexte : Projet étudiant machine learning
# Objectifs : 
# Comprendre les algorithmes de regression linéaire
# + et les pratiquer avec des données 
#
######################################
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
	return np.sqrt(np.mean(diff ** 2))
	#return np.mean(diff ** 2)

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
data=np.loadtxt(open("winequality-red.csv","rb"),delimiter=";",skiprows=1)
# Prendre la dernière colone
Y=data[:,-1]

# prendre les 10 premières colones
X=data[:,0:11]

def linearregression2(X, Y, epsilon, nbiteration) : 
	W = np.random.rand(X.shape[1])
	for i in range(nbiteration) :
		#W = W + 2 * X*(X * W -Y) * epsilon
		#W = W + (np.dot((np.dot(X,Y)-Y), (2*X)))*epsilon
		W = W - 2*epsilon*np.dot(X.transpose(),np.dot(X, W) - Y)
		# On calcule la RMSE pour chaque valeur calculée
		print(rmse(np.dot(X,W),Y))
	# On retourne notre vecteur de poid W calculé
	return W


#  
w = linearregression2(X, Y, 0.00000001, 10000)

# test du modele
vin = np.array([7.3,0.65,0,1.2,0.065,15,21,0.9946,3.39,0.47,10])
"""
On pourrait tester le modèle avec une matrice de vin bien plus grande
Ici, nous avons choisi de prédire la note d'un vin choisi au hasard dans le fichier
wine.
"""

# Definition de la fonction de prediction
def pred(w, Y):
	Ypred = np.dot(vin,w)
	return Ypred


print pred(w, vin)







 

	

		 

