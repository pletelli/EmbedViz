#!/usr/bin/python2.7
from rsvd import RSVD,MovieLensDataset
import matplotlib.pyplot as plt
import numpy as np

def findFactorsAndErrors(ratingsDataset):

    ratings=ratingsDataset.ratings()


    # create train, validation and test sets.
    n = int(ratings.shape[0]*0.8)
    train = ratings[:n]
    test = ratings[n:]
    v = int(train.shape[0]*0.9)
    val = train[v:]
    train = train[:v]


    dims = (ratingsDataset.movieIDs().shape[0], ratingsDataset.userIDs().shape[0])

    
    factors = []
    errors = []
    # lambda_f ne doit pas depasser 1


    # default values
    #probeArray=None
    #maxEpochs=100
    #minImprovement=0.000001
    #learnRate=0.001
    #regularization=0.011
    #randomize=False
    #randomNoise=0.005
    for factor in range(1, 100):
        model = RSVD.train(factor, train, dims, probeArray=val, maxEpochs = 1000, regularization=0.011)

        sqerr=0.0
        for movieID,userID,rating in test:
             err = rating - model(movieID,userID)
             sqerr += err * err
        sqerr /= test.shape[0]

        factors.append(factor)
        errors.append(np.sqrt(sqerr))

    # returns a dict, do result['best_factor'] to get the corresponding value
    return {'factors':factors, 'errors':errors}


ratingsDataset = MovieLensDataset.loadDat('data_movilens1m/ratings.dat')
result = findFactorsAndErrors(ratingsDataset)

errors = result['errors']
factors = result['factors']

  # get minimal error and its corresponding lamda
min_err = min(errors)
id_min_err = errors.index(min(errors))
best_factor = factors[id_min_err]
plt.plot(factors, errors)
plt.ylabel('erreur')
plt.xlabel('factors')
plt.show()

print best_factor
print min_err

# nous choisissons le facteur : 27