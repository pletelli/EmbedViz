# coding: utf-8
from os.path import exists
from rsvd import RSVD, rating_t, MovieLensDataset

import numpy as np
import scipy.spatial.distance as dist
from collections import Counter
import re

# one way to test a model : RMSE
def computeRMSE(model,test):
    """ 
    """
    sqerr=0.0
    for movieID,userID,rating in test:
             err = rating - model(movieID,userID)
             sqerr += err * err
    sqerr /= test.shape[0]

    print "-------------------------------------------------"
    print " Test RMSE: ", np.sqrt(sqerr)
    print "-------------------------------------------------"
    return np.sqrt(sqerr)



def getSimilaritiesFromModel(model):
    # on obtient une matrice de dissimilarite entre films si model = model.u
    res = dist.pdist(model) # dist gathers the upper dist of the squareform matrix
    #return dist.squareform(dist)
    return res

# trouve parmis l'ensemble des films les 2 les moins semblables
def getLessSimilarOfAll(moviesSimilarityVector):
   #i,j = np.unravel_index(moviesSimilarityVector.argmax(), moviesSimilarityVector.shape)
    i,j = getIndexes(moviesSimilarityVector.argmax(),moviesSimilarityVector.shape[0])
    return (i, j)

# trouve le film le moins semblable au film donne
def getLessSimilarOfOne(moviesSimilarityVector, movieID):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    i = moviesSimilarityMatrix[movieID].argmax()
    return i

# trouve les n films les moins semblables au film donne
def getNLessSimilarOfOne(n, moviesSimilarityVector,movieID):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    ii = np.argsort(moviesSimilarityMatrix[movieID])[-n:] # positions
    return ii

# trouve les n films les moins semblables parmis l'ensemble
def getNLessSimilarOfAll(n, moviesSimilarityVector):
    allIds = np.zeros((n,2), int)
    di = []
    copie = np.copy(moviesSimilarityVector)
    for i in xrange(n):
        ids = np.argmax(copie)
        di.append(np.amax(copie))
        allIds[i] = getIndexes(ids,moviesSimilarityVector.shape[0])
        copie[ids] = -1;

    return allIds, di

# trouve parmis l'ensemble des films les 2 les plus semblables
def getMostSimilarOfAll(moviesSimilarityVector):
    i,j = getIndexes(moviesSimilarityVector.argmin(),moviesSimilarityVector.shape[0])
    return (i, j)

# trouve le film le plus semblable au film donne
def getMostSimilarOfOne(moviesSimilarityVector, movieID):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    i = moviesSimilarityMatrix[movieID].argmin()
    return i

# trouve les n films les plus semblables au film donne
def getNMostSimilarOfOne(n, moviesSimilarityVector,movieID):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    ii = np.argsort(moviesSimilarityMatrix[movieID])[1:n+1] # positions # le premier est toujours 0 donc on ne le prend pas
    di = np.sort(moviesSimilarityMatrix[movieID])[1:n+1]
    return ii,di

# trouve les n films les plus semblables parmis l'ensemble
def getNMostSimilarOfAll(n, moviesSimilarityVector):
    allIds = np.zeros((n,2), int)
    di = []
    copie = np.copy(moviesSimilarityVector)
    for i in xrange(n):
        ids = np.argmin(copie)
        di.append(np.amin(copie))
        allIds[i] = getIndexes(ids,moviesSimilarityVector.shape[0])
        copie[ids] = 1000;

    return allIds, di

#trouve le film qui ne ressemble a aucun autre
def getWeirdestOfAll(n, moviesSimilarityVector):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    reduced = [sum(moviesSimilarityMatrix[i]) for i in range(moviesSimilarityMatrix.shape[0])]
    i = np.argmax(reduced)
    return i

#trouve le film qui ressemble a tous les autre
def getNormalestOfAll(n, moviesSimilarityVector):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    reduced = [sum(moviesSimilarityMatrix[i]) for i in range(moviesSimilarityMatrix.shape[0])]
    i = np.argmin(reduced)
    return i

#trouve les n films qui ne ressemblent a aucun autre
def getNWeirdestOfAll(n, moviesSimilarityVector):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    reduced = [sum(moviesSimilarityMatrix[i]) for i in range(moviesSimilarityMatrix.shape[0])]
    i = np.argsort(reduced)[-n:]
    return i

#trouve les n films qui ressemblent a tous les autre
def getNNormalestOfAll(n, moviesSimilarityVector):
    moviesSimilarityMatrix = dist.squareform(moviesSimilarityVector)
    reduced = [sum(moviesSimilarityMatrix[i]) for i in range(moviesSimilarityMatrix.shape[0])]
    i = np.argsort(reduced)[:n]
    return i


# ecrire un rapport 
def getGlobalStatistics(moviesSimilarityVector,original_movieIDs, moviesInformations):
    # les 20 films les plus similaires
    print "------------------------------------"
    print "  les 20 films les plus similaires"
    print "------------------------------------"
    simil,di = getNMostSimilarOfAll(20, moviesSimilarityVector)
    for i,row in enumerate(simil):
        print i, "avec ", round(di[i],3)," : ", getMovieInformations(moviesInformations,original_movieIDs[row[0]])[0].decode('utf-8'),getMovieInformations(moviesInformations,original_movieIDs[row[0]])[1].decode('utf-8')
        print "       ", getMovieInformations(moviesInformations,original_movieIDs[row[1]])[0].decode('utf-8'),getMovieInformations(moviesInformations,original_movieIDs[row[1]])[1].decode('utf-8')


    # les 20 films les plus dissimilaires
    print "--------------------------------------"
    print "  les 20 films les plus dissimilaires"
    print "--------------------------------------"
    simil,di = getNLessSimilarOfAll(20, moviesSimilarityVector)
    for i,row in enumerate(simil):
        print i, "avec ", round(di[i],4)," : ", getMovieInformations(moviesInformations,original_movieIDs[row[0]])[0].decode('utf-8'),getMovieInformations(moviesInformations,original_movieIDs[row[0]])[1].decode('utf-8')
        print "       ", getMovieInformations(moviesInformations,original_movieIDs[row[1]])[0].decode('utf-8'),getMovieInformations(moviesInformations,original_movieIDs[row[1]])[1].decode('utf-8')

    # les 5 films les plus "normaux"
    print "----------------------------------"
    print '  les 5 films les plus "normaux"'
    print "----------------------------------"
    simil = getNNormalestOfAll(5, moviesSimilarityVector)
    for i in range(simil.shape[0]):
        print i, " : ", getMovieInformations(moviesInformations,original_movieIDs[simil[i]])


    # les 5 films les moins "normaux"
    print "----------------------------------"
    print '  les 5 films les moins "normaux"'
    print "----------------------------------"
    simil = getNWeirdestOfAll(5, moviesSimilarityVector)
    for i in range(simil.shape[0]):
        print i, " : ", getMovieInformations(moviesInformations,original_movieIDs[simil[i]])




def loadMoviesInformations(file):
    """Loads the names of the movies via movies.dat file. 
    """
    if not exists(file):
        raise ValueError("%s file does not exist" % file)
    f=open(file)
    try:
        rows=[tuple(l.rstrip().split("::")) for l in f.readlines()]
        moviesInfo = {a:(b,c) for a,b,c in rows}

        return moviesInfo
    finally:
        f.close()  


#id :: gender :: age :: occupation :: zipcode
def loadUsersInformations(file):
    """Loads the names of the movies via movies.dat file. 
    """
    if not exists(file):
        raise ValueError("%s file does not exist" % file)
    f=open(file)
    try:
        rows=[tuple(l.rstrip().split("::")) for l in f.readlines()]
        usersInfo = {a:(b,c,d,e) for a,b,c,d,e in rows}

        return usersInfo
    finally:
        f.close() 

def getMovieInformations(table, movieID):
    """Get the name and genre of the given movieID from the given table.
    """
    return table[str(movieID)] 

def getMovieName(table, movieID):
    """Get the name of the movie.
    """
    return re.sub('\(.*?\)','', getMovieInformations(table, movieID)[0]).strip()


def getMovieGenres(table, movieID):
    """Loads the genre of the given movieID.
    """
    return getMovieInformations(table, movieID)[1]

def getGenres(table):
    genres = [(key,list(value[1].rstrip().split("|"))) for (key,value) in table.iteritems()]
    genres1 = dict()
    for (key, value) in genres :
        genres1[key] = value[0] 
    return genres1


def getMoviesNames(table,movieIDs) :
    names = list()
    for i, movieID in enumerate(movieIDs):
        name = getMovieName(table, movieID)
        names.append(name)
    return names



# get the indexes in the square dist matrix given an index for the condensed dist matrix
def getIndexes(index, n):
    n= np.ceil(np.sqrt(2* n))
    ti= np.triu_indices(n, 1)
    return ti[0][index], ti[1][index]

# get the movies that have at least n rates
def getNMostRatedMovies(ratings,n=600):
    timesRated = Counter(ratings['f0'])
    return list(movie for movie,rateNb in timesRated.iteritems() if rateNb>=n)

# get the movies that have at least n rates
def getNMostRatingUsers(ratings,n=100):
    timesRated = Counter(ratings['f1'])
    return list(user for user,rateNb in timesRated.iteritems() if rateNb>=n)


# get the movies that have been rated n
def getRatedMovies(ratings,n=5):
    j = 0
    result=np.empty((len(mostRated),),dtype=rating_t)
    for i,row in enumerate(ratings):
        if row['f2'] == n:
            result[j]=row
            j= j + 1
    return result
##this functions is bad =)
