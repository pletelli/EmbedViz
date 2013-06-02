#!/usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as plt
from rsvd import RSVD, rating_t, MovieLensDataset



ratingsDataset = MovieLensDataset.loadDat('data_movilens1m/ratings.dat')

ratings=ratingsDataset.ratings()

# make sure that the ratings a properly shuffled
np.random.shuffle(ratings)

# create train, validation and test sets.
n = int(ratings.shape[0]*0.8)
train = ratings[:n]
test = ratings[n:]
v = int(train.shape[0]*0.9)
val = train[v:]
train = train[:v]


dims = (ratingsDataset.movieIDs().shape[0], ratingsDataset.userIDs().shape[0])
factor = 40

lambdas = []
errors = []
# lambda_f ne doit pas depasser 1
# maxEpochs = 1000
for lambda_f in np.arange(0.0, 0.05, 0.0005): 
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


#plot errors /lambdas
plt.plot(lambdas, errors)
plt.ylabel('erreur')
plt.xlabel('lambda')
plt.show()


