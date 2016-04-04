#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import math as m


matrice = np.array([[5,3,3,5],[6,3,11,5],[2,8,5,6], [5,3,3,2]])*1.0

def standardize(mat):
	for col in range(0,mat.shape[1]):
		mincol = min(mat[:,col])
		maxcol = max(mat[:,col])
		mat.astype(float)
		mat[:,col] = (mat[:,col] - mincol) / (maxcol - mincol)*1.0
	return mat


print("matrice : \n" + str(matrice))
stdmat = standardize(matrice)
print("\nMatrice standardisée : \n" + str(stdmat))

# Spliter une matrice en deux
def split(mat, taux):
	nbligtrain = mat.shape[1] * taux
	train = mat[:nbligtrain:]
	test = mat[nbligtrain:mat.shape[1]:]
	

print("\nmatrice splittée :\n")
split(matrice, 0.75)

def accuracy(A, B):
	return np.mean(A == B)


vec1 = np.array([2,3,3,4])
vec2 = np.array([2,3,3,5])

print(accuracy(vec1, vec2))

def knnReg(X, k, vec):
	# X matrice sans les notes
	# vec vecteur a predire
	# k nombre de k les plus proches
	# Calcul de la distance entre le vecteur a prédire et les autres vecteurs
	# Création d'un vecteur qui répertorie les distances
	distance = np.sum((X[:,0:X.shape[1]-1]-vec)**2,axis=1)
	# Range les index dans l'ordre croissant des résultats de la distance
	a = np.argsort(distance)
	# On garde les k premiers index
	ind = a[:k]
	# On retourne la moyenne des notes des vins les plus proches (correspondant aux index)
	return np.mean(X[ind][:,-1])


"""
Test manuel à executer pour comprendre le principe
matrice = np.array([[5,3,3,5,5],[6,3,11,5,6],[2,8,5,6,3], [5,3,3,2,2]])*1.0
vec1 = np.array([2,3,3,4])
print "\n\n\n"
print matrice
print vec1
knnReg(matrice, 2, vec1)
"""

# KNN final
data = np.genfromtxt("winequality-red.csv", delimiter=";",skip_header=1)
noteapred = np.array([6, 0.31, 0.47, 3.6, 0.067, 18,42, 0.99549, 3.39, 0.66, 11])

print "\n\nNote prédie pour le vin :"
print knnReg(data, 100, noteapred)