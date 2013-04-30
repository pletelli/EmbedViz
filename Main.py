#!/usr/bin/python2.7
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as hier
from scipy.cluster.vq import kmeans, whiten, vq
import scipy.spatial.distance as dist
# Utilisation de Pylab et de mplot3d integre a matplotlib
import mpl_toolkits.mplot3d.axes3d as p3
#
from rsvd import RSVD, rating_t, MovieLensDataset
# 
import functions
import changeEncoding

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

    original_movieIDs=movieIDs
        
    movieIDs=np.unique(ratings['f0'])
    movieIDs.sort()
    ratingsDataset = MovieLensDataset(movieIDs,userIDs,ratings)
finally:
    f.close()

#changeEncoding.convert_to_utf8('data_movilens1m/movies.dat')

moviesInformations = functions.loadMoviesInformations('data_movilens1m/movies.dat')

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

factor = 27 #minimum d'erreur pour regularization=0.011
model = RSVD.train(factor, train, dims, probeArray=val, maxEpochs = 1000, regularization=0.011)

rmse = functions.computeRMSE(model, test)


#model.u c est une matrice avec un vecteur par film
#model.v c est une matrice avec un vecteur par utilisateur
dissim = functions.getSimilaritiesFromModel(model.u)

#trouver les films avec le plus de dissimilarite
functions.getGlobalStatistics(dissim,original_movieIDs, moviesInformations)

k = functions.getLessSimilarOfOne(dissim,1)
print k

########
#PCA
########
# Preparation de la methode PCA pour une projection sur 3 dimensions
pca = PCA(n_components=3)


# PCA pour l'ensemble des films

# Calcul de la projection a partir des donnees
pca.fit(model.u)
# Application de la projection aux donnees
newMovies = pca.transform(model.u)

# Creation de la figure dans laquelle nous allons representer le nuage de point
fig=plt.figure()
ax = p3.Axes3D(fig)

ax.scatter3D(newMovies[:,0],newMovies[:,1],newMovies[:,2], label="coucou")
fig.add_axes(ax)
plt.show()

# PCA pour les films les plus notes
mostRated = functions.getNMostRatedMovies(ratings,2000)

j = 0
subModel=np.empty((len(mostRated),model.u.shape[1])) #subRatings doit etre du meme type que model
for i,row in enumerate(model.u):
    if i in mostRated:
        subModel[j]=row
        j= j +1


pca.fit(subModel)
# Application de la projection aux donnees
newMovies = pca.transform(subModel)

# Creation de la figure dans laquelle nous allons representer le nuage de point
fig=plt.figure()
ax = p3.Axes3D(fig)

ax.scatter3D(newMovies[:,0],newMovies[:,1],newMovies[:,2])

fig.add_axes(ax)
plt.show()

pca2 = PCA(n_components=2)
pca2.fit(subModel)
# Application de la projection aux donnees
newMovies = pca2.transform(subModel)

# Percentage of variance explained for each components
print "explained variance ratio (first two components): ", pca2.explained_variance_ratio_

#plot with labels
labels = functions.getMoviesNames(moviesInformations,original_movieIDs[mostRated])

plt.figure()
plt.scatter(newMovies[:, 0], newMovies[:, 1])
plt.legend()
plt.title('PCA of movies from model')
for label, x, y in zip(labels, newMovies[:, 0], newMovies[:, 1]):
    plt.annotate(
        label.decode('utf-8'), 
        xy = (x, y), xytext = (30, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-'))

plt.show()



########
# Hierarchical clustering
########


dissim2 = functions.getSimilaritiesFromModel(subModel)
functions.getGlobalStatistics(dissim2,original_movieIDs, moviesInformations)
linkageMatrix = hier.linkage(dist.squareform(dissim2), method='single')
dendro = hier.dendrogram(linkageMatrix, leaf_rotation=90)
plt.show()

########
# K-Means
########

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(subModel,5)

# assign each sample to a cluster
idx,_ = vq(subModel,centroids)

# some plotting using numpy's logical indexing
plt.plot(subModel[idx==0,0],subModel[idx==0,1],'oc',
     subModel[idx==1,0],subModel[idx==1,1],'or',
     subModel[idx==2,0],subModel[idx==2,1],'ob',
     subModel[idx==3,0],subModel[idx==3,1],'om',
     subModel[idx==4,0],subModel[idx==4,1],'oy')



plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.show()

labels = functions.getMoviesNames(moviesInformations,original_movieIDs[mostRated])

plt.figure()
plt.scatter(newMovies[:, 0], newMovies[:, 1])
for label, x, y in zip(labels, newMovies[:, 0], newMovies[:, 1]):
    plt.annotate(
        label.decode('utf-8'), 
        xy = (x, y), xytext = (30, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-'))




# extraire le nombre d iterations
# visualiser avec un modele pourri
# caraceristique films (note moyenne, genre, nombre de personne l'ayant note, type des gens l'ayant note)
#selecetion des films a visu en focntion du nombre de note, faible, pas
# 5OO objet cest bien a visu
# restreindre lapprentisage aux gens dun certain age/sexe




# TODO
# interface graphique pour choisir nb facteur, reg, learnRate?
# reussir a obtenir un modele pour lequel err = 0
# possibilite d afficher diverses visualisation (par le biais de l interface graphique) -> plot ?
# train sur jester jokes

