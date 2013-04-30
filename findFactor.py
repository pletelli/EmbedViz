#!/usr/bin/python2.7
import functions
from rsvd import MovieLensDataset
import matplotlib.pyplot as plt

ratingsDataset = MovieLensDataset.loadDat('data_movilens1m/ratings.dat')
result =functions.findBestFactor(ratingsDataset)

errors = result['errors']
factors = result['factors']

min_err = min(errors)
id_min_err = errors.index(min(errors))
best_factor = factors[id_min_err]
plt.plot(factors, errors)
plt.ylabel('erreur')
plt.xlabel('factors')
plt.show()

print best_factor
print min_err

# nous choisissons le facteur : 