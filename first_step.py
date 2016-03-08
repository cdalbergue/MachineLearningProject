# on importe numpy
import numpy as np

# np est un alias plus court pour numpy 

#hello world
print("hello world!")

#creer un vecteur
vec=np.array([1,2,3,4])

#creer une matrice
mat=np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12],
              [13,14,15,16]])

#afficher le vecteur et la matrice
print(mat)

print(vec)

#multiplication scalaire matrice

mat= 3*mat

#multiplication vecteur matrice

res=np.dot(vec,mat)

#somme des tous les elements du vecteur res

somme=sum(res)

#creer une matrice de nombres aleatoires
nbligne=5
nbcolonne=3

mat=np.random.rand(nbligne,nbcolonne)

#si on veut des valeurs entieres
mat=np.random.randint(nbligne,nbcolonne)

#matrice de 1 ou de 0

m1 = np.ones((10,2))  # matrice de 1, argument = nuplet avec les dimensions
                       # ATTENTION np.ones(10,2) ne marche pas

m0 = np.zeros((10,2))  # matrice de 0, argument = nuplet avec les dimensions
                       # ATTENTION np.ones(10,2) ne marche pas

#afficher la dimension de la matrice
print(mat.shape) #resultat sous forme nbligne nbcolonne

#selectionner une ligne dans la matrice

ligne=mat[0,:] #la premiere ligne

#selectionner une colonne dans la matrice

colonne=mat[:,2] #la troisieme colonne


#vecteur des valeurs de 0 à 9

vec=np.arange(10)

#boucle sur python

for i in range(10):
    print i    

#concatener des chaines de caracteres

ch1="hello"
ch2=" 42"
ch3=ch1+" numero "+ch2

#longueur d un vecteur

long=len(vec)

#ou

long= vec.shape[0]


#definir une fonction

def carre(x):
    return x**2




#maximum d un vecteur
def maximum(vec)
    maxi=vec[0]
    for i in range(len(vec)):
      if vec[i]>maxi:
          maxi=vec[i]
    return maxi

#!!! attention !!!
"""
numpy dispose d une fonction max beaucoup plus rapide
maxi=np.max(vec)
de maniere generale toujours eviter les boucles en python quand c est possible

"""


#lecture ecriture de matrice

"""
mat=np.loadtxt("fichier.txt")

np.savetxt("fichier.txt", mat)

"""





















