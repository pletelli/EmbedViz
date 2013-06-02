#!/usr/bin/python2.7
# coding: utf-8
import numpy as np
#
from rsvd import RSVD, rating_t, MovieLensDataset
# 
import functions

from os import remove

#ratingsDataset = MovieLensDataset.loadDat('data_movilens1m/ratings.dat')

f=open('data_movilens1m/ratings.dat')
try:
    rows=[tuple(map(int,l.rstrip().split("::"))) for l in f.readlines()]
    n=len(rows)
    # define rating array (itemID,userID,rating)
    ratings=np.empty((n,),dtype=rating_t)
    for i,row in enumerate(rows):
        ratings[i]=(row[1],row[0]-1,row[2])
    movieIDs=np.unique(ratings['f0'])
    userIDs=np.unique(ratings['f1'])
    movieIDs.sort()
    userIDs.sort()
    #map movieIDs
    for i,rec in enumerate(ratings):
        ratings[i]['f0']=movieIDs.searchsorted(rec['f0'])+1
    # correspondance entre les ids du fichier movies.dat et les ids utilisés dans l'objet ratings
    original_movieIDs=movieIDs      
    movieIDs=np.unique(ratings['f0'])
    movieIDs.sort()
    ratingsDataset = MovieLensDataset(movieIDs,userIDs,ratings)
finally:
    f.close()

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


dims = (ratingsDataset.movieIDs().shape[0], ratingsDataset.r)

factor = 27 #minimum d'erreur pour regularization=0.011
model = RSVD.train(factor, train, dims, probeArray=val, maxEpochs = 1000, regularization=0.013)



movieMean=np.empty((model.u.shape[0],2)) #subRatings doit etre du meme type que model
for row in ratings:
    movieMean[row[0]-1][0] = movieMean[row[0]-1][0]+ row[2]
    movieMean[row[0]-1][1] = movieMean[row[0]-1][1]+ 1
for i in range(len(movieMean)):
    movieMean[i][0] = np.round(movieMean[i][0]/movieMean[i][1])

userMean=np.empty((model.v.shape[0],2)) #subRatings doit etre du meme type que model
for row in ratings:
    userMean[row[1]][0] = userMean[row[1]][0]+ row[2]
    userMean[row[1]][1] = userMean[row[1]][1]+ 1
for i in range(len(userMean)):
    userMean[i][0] = np.round(userMean[i][0]/userMean[i][1])

#suppression des anciens models
remove('objects/model')
remove('objects/u.arr')
remove('objects/v.arr')
# écriture de l'objet model dans un fichier
model.save('objects/')
original_movieIDs.dump('objects/original_movieIDs')
ratings.dump('objects/ratings')
movieMean.dump('objects/movieMean')
userMean.dump('objects/userMean')

# ANALYSE
rmse = functions.computeRMSE(model, test)

# pour retrouver l'id d'origine d'un film à partir de son id dans ratings 
#id_dorigine = original_movieIDs[id_dans_ratings]-1
# id_dans_ratings = original_movieIDs.searchsorted(id_dorigine) +1

# mais dans les matrices de similarités ou dans model.u , si l'on remplace id_dans_ratings par 
# id_dans_matrice, il n'y a plus besoin de soustraire ou d'aditionner 1