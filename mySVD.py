#!/usr/bin/python2.7
import numpy as np
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
from rsvd import RSVD, rating_t, MovieLensDataset

dataset = MovieLensDataset.loadDat('data_movilens1m/ratings.dat')
ratings=dataset.ratings()

# make sure that the ratings a properly shuffled
np.random.shuffle(ratings)

# create train, validation and test sets.
n = int(ratings.shape[0]*0.8)
train = ratings[:n]
test = ratings[n:]
v = int(train.shape[0]*0.9)
val = train[v:]
train = train[:v]


dims = (dataset.movieIDs().shape[0], dataset.userIDs().shape[0])

# boucle pour iterer sur reg
factor = 40

i = 0
lambdas = []
errors = []
for lambda_f in np.arange(0.00, 0.03, 0.001): 
# lambda_f ne doit pas depasser 1
# masEpochs = 1000
	model = RSVD.train(factor, train, dims, probeArray=val, maxEpochs = 1000, regularization=lambda_f)

	sqerr=0.0
	for movieID,userID,rating in test:
   		 err = rating - model(movieID,userID)
   		 sqerr += err * err
	sqerr /= test.shape[0]


	print "-------------------------------------------------"
	print "Pour lambda = ",lambda_f, " Test RMSE: ", np.sqrt(sqerr)
	print "-------------------------------------------------"
	lambdas.append(lambda_f)
	errors.append(np.sqrt(sqerr))

# print the lamdas and errors vectors
print lambdas
print errors

# get minimal error and its corresponding lamda
min_err = min(errors)
id_min_err = errors.index(min(errors))
best_lambda = lambdas[errors.index(min(errors))]
print "minimum trouve pour l erreur", min_err
print "correspond a lambda =", best_lambda


#print " lambda | erreur "
#print "________________"
#i =0
#while i < len(lambdas) & i < len(errors):
	#print i
	#print lambdas[i], " | " , errors[i]


#plot errors /lambdas
plt.plot(lambdas, errors)
plt.ylabel('erreur')
plt.xlabel('lambda')
plt.show()

#model.u c est une matrice avec un vecteur par film
#model.v c est une matrice avec un vecteur par utilisateur

dist = sc.pdist(model.u)
dissim = sc.squareform(dist)
# on obtient une matrice de dissimilarite




#calcul de norme, retrouver le film associe

# extraire le nombre d iterations
# visualiser avec un modele pourri
# caraceristique films (note moyenne, genre, nombre de personne l'ayant noté, type des gens l'ayant noté)
#selecetion des films a visu en focntion du nombre de note, faible, pas
# 5OO objet cest bien a visu
# restreindre lapprentisage aux gens dun certain age/sexe




# TODO
# interface graphique pour choisir nb facteur, reg, learnRate?
# reussir a obtenir un modele pour lequel err = 0
# possibilite d afficher diverses visualisation (par le biais de l interface graphique) -> plot ?
# train sur jester jokes

