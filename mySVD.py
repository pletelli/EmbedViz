#!/usr/bin/python2.7
import numpy as np
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
alphas = []
errors = []
for alpha in np.arange(0, 1, 0.005): 
# alpha ne doit pas depasser 1
# masEpochs = 1000
	model = RSVD.train(factor, train, dims, probeArray=val, maxEpochs = 500, regularization=alpha)

	sqerr=0.0
	for movieID,userID,rating in test:
   		 err = rating - model(movieID,userID)
   		 sqerr += err * err
	sqerr /= test.shape[0]


	print "-------------------------------------------------"
	print "Pour alpha = ",alpha, " Test RMSE: ", np.sqrt(sqerr)
	print "-------------------------------------------------"
	alphas.append(alpha)
	errors.append(sqerr)


print alphas
print errors
#print " alpha | erreur "
#print "________________"
#i =0
#while i < len(alphas) & i < len(errors):
	#print i
	#print alphas[i], " | " , errors[i]

plt.plot(alphas, errors)
plt.ylabel('erreur')
plt.xlabel('alpha')
plt.show()
# extraire le nombre d iterations
# 




# TODO
# interface graphique pour choisir nb facteur, reg, learnRate?
# reussir a obtenir un modele pour lequel err = 0
# possibilite d afficher diverses visualisation (par le biais de l interface graphique) -> plot ?
# train sur jester jokes

