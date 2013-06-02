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
import math
import networkx as nx

#load model.u and model.v
#model.u c est une matrice avec un vecteur par film
#model.v c est une matrice avec un vecteur par utilisateur

class AnalysisObj(object):
    """AnalysisObj interface
    """

    def __init__(self):
        self._model= RSVD.load("objects/")
        self._original_movieIDs=np.load("objects/original_movieIDs")
        self._ratings= np.load("objects/ratings")
        self._moviesInformations = functions.loadMoviesInformations('data_movilens1m/movies.dat')
        self._usersInformations = functions.loadUsersInformations('data_movilens1m/users.dat')
        self._dissim = functions.getSimilaritiesFromModel(self._model.u)
        self._movieMean=np.load("objects/movieMean")
        self._userMean=np.load("objects/userMean")

        # PCA pour les films les plus notes
        self._mostRated = functions.getNMostRatedMovies(self._ratings,2000)

        j = 0
        self._subModel=np.empty((len(self._mostRated),self._model.u.shape[1])) #subRatings doit etre du meme type que model
        self._subMovieMean = np.empty((len(self._mostRated),2))
        for i,row in enumerate(self._model.u):
            if i in self._mostRated:
                self._subMovieMean[j] = self._movieMean[i]
                self._subModel[j]=row
                j= j +1

        # PCA pour les films les plus notes
        self._mostRating = functions.getNMostRatingUsers(self._ratings)
        j = 0
        self._subModel2=np.empty((len(self._mostRating),self._model.v.shape[1])) #subRatings doit etre du meme type que model
        self._subMovieMean2 = np.empty((len(self._mostRating),2))
        for i,row in enumerate(self._model.v):
            if i in self._mostRating:
                self._subMovieMean2[j] = self._userMean[i]
                self._subModel2[j]=row
                j= j +1
       


    def model(self):
        return self._model

    def original_movieIDs(self):
        return self._original_movieIDs

    def ratings(self):
        return self._ratings

    def moviesInformations(self):
        return self._moviesInformations

    def dissim(self):
        return self._dissim

    def mostRated(self):
        return self._mostRated

    def subModel(self):
        return self._subModel

    def movieMean(self):
        return self._movieMean


    def afficher1(self):
        """trouver les films avec le plus de dissimilarites, similarites
        """
        functions.getGlobalStatistics(self._dissim,self._original_movieIDs, self._moviesInformations)

    def afficher2(self):
        # Star Wars - Return of the Jedi real_id = 1210
        #il faut trouver le computed_id
        print "Star Wars - Return of the Jedi id = 1210"
        computed_id = self._original_movieIDs.searchsorted(1210)
        simil, di = functions.getNMostSimilarOfOne(20, self._dissim, computed_id)
        for i,row in enumerate(simil):
            print i, " avec ", round(di[i],4), " : ", functions.getMovieInformations(self._moviesInformations,self._original_movieIDs[row])


        # James Bond - Goldeneye id = 10
        print "James Bond - Goldeneye id = 10"
        computed_id = self._original_movieIDs.searchsorted(10)
        simil, di = functions.getNMostSimilarOfOne(20, self._dissim, computed_id)
        for i,row in enumerate(simil):
            print i," avec ", round(di[i],4),  " : ", functions.getMovieInformations(self._moviesInformations,self._original_movieIDs[row])

        # Indiana Jones and the lats crusade id = 1291
        print "Indiana Jones and the last crusade id = 1291"
        computed_id = self._original_movieIDs.searchsorted(1291)
        simil, di = functions.getNMostSimilarOfOne(20, self._dissim, computed_id)
        for i,row in enumerate(simil):
            print i," avec ", round(di[i],4),  " : ", functions.getMovieInformations(self._moviesInformations,self._original_movieIDs[row])


    def pca_movies(self, color):
                 # Preparation de la methode PCA pour une projection sur 3 dimensions
        pca = PCA(n_components=3)
        # PCA pour l'ensemble des films
        # Calcul de la projection a partir des donnees
        pca.fit(self._model.u)
        # Application de la projection aux donnees
        newMovies = pca.transform(self._model.u)
        # Creation de la figure dans laquelle nous allons representer le nuage de point
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if color == 0:
            # colorer selon la moyenne donnée au film
            moyennes = [row[0] for row in self._movieMean]
            moyennes = [0 if math.isnan(x) else x for x in moyennes]
            colors = ['blue','green','cyan','yellow','magenta','red']
            categories = np.unique(moyennes)
            colordict = dict(zip(categories, colors)) 
            listColors = [colordict[x] for x in moyennes]
        elif color ==1:
            # colorer selon le genre du film
            genres = functions.getGenres(self._moviesInformations)
            categories = ['Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
            colors = ['IndianRed','Red','Pink','PaleVioletRed','LightSalmon','Orange','Gold','Violet','Purple','DarkSlateBlue','GreenYellow','DarkOliveGreen','MediumAquamarine','DarkCyan', 'CornflowerBlue','Navy','MistyRose','Peru','Maroon']
            colordict = dict(zip(categories, colors)) 

        #faire la correspondance entre les ids
        genres2 = list()
        for i in range (0, self._model.u.shape[0]):
            genres2.append(genres[str(self._original_movieIDs[i])])

        listColors = [colordict[x] for x in genres2]

        ax.scatter(newMovies[:,0],newMovies[:,1],newMovies[:,2], c=listColors)
        #fig.add_axes(ax)
        plt.title('3D PCA of movies from model')
        plt.show()



        pca.fit(self._subModel)
        # Application de la projection aux donnees
        newMovies = pca.transform(self._subModel)

        # Creation de la figure dans laquelle nous allons representer le nuage de point
        fig=plt.figure()
        ax = p3.Axes3D(fig)
        
        j=0
        listColors2 =list()
        for i,x in enumerate(listColors) :
            if (i in self._mostRated):
                listColors2.append(x)
                j=j+1
        ax.scatter3D(newMovies[:,0],newMovies[:,1],newMovies[:,2],c=listColors2)

        fig.add_axes(ax)
        plt.show()

        pca2 = PCA(n_components=2)
        pca2.fit(self._subModel)
        # Application de la projection aux donnees
        newMovies = pca2.transform(self._subModel)

        # Percentage of variance explained for each components
        print "explained variance ratio (first two components): ", pca2.explained_variance_ratio_

        #plot with labels
        labels = functions.getMoviesNames(self._moviesInformations,self._original_movieIDs[self._mostRated])

        plt.figure()
        plt.scatter(newMovies[:, 0], newMovies[:, 1],c=listColors2)
        plt.legend()
        plt.title('2D PCA of most rated movies from model')
        for label, x, y in zip(labels, newMovies[:, 0], newMovies[:, 1]):
            plt.annotate(
                label.decode('utf-8'), 
                xy = (x, y), xytext = (30, 10),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                arrowprops = dict(arrowstyle = '-'))

        plt.show()

    def pca_users(self, color):

        "argument color can take value 0,1,2 "
        "respectively for color following gender, ages or occupation"

        #correspondance entre les utilisateurs du fichiers users.dat et ceux de la matrice model.v
        # il n'y a pas d'utilisateurs manquants =)
        # donc model.v[0] correspond à l'utilisateur à l'id 1

        liste = list()
        for i in range(0,self._model.v.shape[0]):
                item = self._usersInformations[str(i+1)][color]
                liste.append(item)
        if color == 0:
            # colorer les points selon le genre
            categories = ['M','F']
            colors = ['blue','red']
        elif color == 1:
            # colorer les points selon l'age des utilisateurs
            categories = ['1','18','25','35','45','50','56']
            colors = ['blue','green','cyan','yellow','magenta','red','black']
        elif color == 2:
            # colorer les points selon la profession des utilisateurs
            categories = [str(x) for x in range(0,21)]
            colors = ['blue','green','cyan','yellow','magenta','red','black','blue','green','cyan','yellow','magenta','red','black','blue','green','cyan','yellow','magenta','red','black']
        


        # Preparation de la methode PCA pour une projection sur 3 dimensions
        pca = PCA(n_components=3)
        # PCA pour l'ensemble des users
        # Calcul de la projection a partir des donnees
        pca.fit(self._model.v)
        # Application de la projection aux donnees
        newMovies = pca.transform(self._model.v)
        # Creation de la figure dans laquelle nous allons representer le nuage de point
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        
        colordict = dict(zip(categories, colors)) 
        listColors = [colordict[x] for x in liste]
            
        ax.scatter(newMovies[:,0],newMovies[:,1],newMovies[:,2], c=listColors)
        #fig.add_axes(ax)
        plt.title('3D PCA of users from model')
        plt.show()

    


########
# Hierarchical clustering
########
    def hclust(self):

        dissim2 = functions.getSimilaritiesFromModel(self._subModel)
        linkageMatrix = hier.linkage(dist.squareform(dissim2), method='single')
        labels = functions.getMoviesNames(self._moviesInformations,self._original_movieIDs[self._mostRated])
        dendro = hier.dendrogram(linkageMatrix, labels=labels, leaf_rotation=90)
        plt.show()

########
# K-Means
########
    def kmeans(self):
        #plot thanks to pca
        pca2 = PCA(n_components=2)

        # computing K-Means with K = 2 (2 clusters)
        centroids,dis = kmeans(self._subModel,5)
        # assign each sample to a cluster
        idx,_ = vq(self._subModel,centroids)
        pca2.fit(self._subModel)
        newMovies = pca2.transform(self._subModel)
        center = pca2.transform(centroids)
        # some plotting using numpy's logical indexing
        plt.plot(newMovies[idx==0,0],newMovies[idx==0,1],'oc',
            newMovies[idx==1,0],newMovies[idx==1,1],'or',
            newMovies[idx==2,0],newMovies[idx==2,1],'ob',
            newMovies[idx==3,0],newMovies[idx==3,1],'om',
            newMovies[idx==4,0],newMovies[idx==4,1],'oy')
        plt.plot(center[:,0],center[:,1],'sg',markersize=8)
        plt.title(dis)
        plt.show()


centroids,dis = kmeans(a._model.u,10)
# assign each sample to a cluster
idx,_ = vq(a._model.u,centroids)

pca2.fit(a._model.u)
# Application de la projection aux donnees
newMovies = pca2.transform(a._model.u)
center = pca2.transform(centroids)
# some plotting using numpy's logical indexing
plt.plot(newMovies[idx==0,0],newMovies[idx==0,1],'oc',
     newMovies[idx==1,0],newMovies[idx==1,1],'or',
     newMovies[idx==2,0],newMovies[idx==2,1],'ob',
     newMovies[idx==3,0],newMovies[idx==3,1],'om',
     newMovies[idx==4,0],newMovies[idx==4,1],'oy',
     newMovies[idx==5,0],newMovies[idx==5,1],'ok',
     newMovies[idx==6,0],newMovies[idx==6,1],color="#aff666", marker="o",
     newMovies[idx==7,0],newMovies[idx==7,1],color="#efe986", marker="o",
     newMovies[idx==8,0],newMovies[idx==8,1],color="#b34ee", marker="o",
     newMovies[idx==9,0],newMovies[idx==9,1],color="#bbbccc", marker="o")
plt.plot(center[:,0],center[:,1],'sg',markersize=8)
plt.title(dis)
#plot with labels
labels = functions.getMoviesNames(a._moviesInformations,a._original_movieIDs)
for label, x, y in zip(labels, newMovies[:, 0], newMovies[:, 1]):
    plt.annotate(
        label.decode('utf-8'), 
        xy = (x, y), xytext = (30, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-'))

plt.show()
print dis


    def bestK(self):
        all_dis = dict()
        for k in range(1,100):
            centroids,dis = kmeans(self._model.u,k)
            all_dis[k] = dis
        plt.plot(all_dis.keys(),all_dis.values(), 'ro')
        plt.title("Methode du coude")
        plt.show()
        # coude présent pour k = 10

    def graph(self) :
        dissim2 = functions.getSimilaritiesFromModel(self._subModel)
        maxi = np.amax(dissim2) #-1
        A = dist.squareform(maxi-dissim2)
        G = nx.from_numpy_matrix(A)
        movieList = functions.getMoviesNames(self._moviesInformations,self._original_movieIDs[self._mostRated])
        G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),movieList)))    

        #G = nx.to_agraph(G)

        #G.node_attr.update(color="red", style="filled")
        #G.edge_attr.update(color="blue", width="2.0")

        nx.draw(G, edge_color = "blue", font_weight="bold")
        plt.show()


    def createErrorMatrix(self):
        "Une ligne par film, une colonne par utilisateur"

        error_list=list()
        error_movie=dict()
        error_user=dict()
        for movieID,userID,rating in self._ratings:
            err = abs(rating - self._model(movieID,userID))
            error_list.append([movieID-1, userID,err])
            if movieID-1 not in error_movie:
                error_movie[movieID-1] = list()

            error_movie[movieID-1].append(err)
            if userID not in error_user:
                error_user[userID] = list()

            error_user[userID].append(err)

        movie_mean_err={i:sum(v)/len(v) for i,v in error_movie.items()}
        user_mean_err={i:sum(v)/len(v) for i,v in error_user.items()}



# extraire le nombre d iterations
# visualiser avec un modele pourri
# caraceristique films (note moyenne, genre, nombre de personne l'ayant note, type des gens l'ayant note)
#selecetion des films a visu en focntion du nombre de note, faible, pas
# 5OO objet cest bien a visu
# restreindre lapprentisage aux gens dun certain age/sexe


### ERREUR QUADRATIQUE MOYENNE
#rmse = functions.computeRMSE(model, test)

### ESTIMATION DE LA MOYENNE DES K ERREURS QUADRATIQUES MOYENNES POUR PLUS DE ROBUSTESSE

### DIRE COMBIEN SONT MAL ESTIMEES, COMBIEN SONT BIEN ESTIMEES

### QUELS SONT LES FILMS QUI, MEME EN AYANT SERVI A LAPPRENTISSAGE DU MODEL, RESTENT MAL ESTIMES, CE SONT DES FILMS DUN CERTAIN PROFIL
### PLOT NOTE REELLE, NOTE CALCULEE PAR FILM, EST CE QUE DES GROUPES SE DISTINGUENT


### FAIRE LINVERSE : AU LIEU DE DEMANDER AU MODEL DE TROUVER UNE NOTE POUR UN FILM ET UN UTILISATEUR DONNE, 
### DEMANDER AU MODEL DE DONNER UN/PLUSIEURS UTILISATEURS POUR UN FILM ET UNE NOTE DONNEE


### TROUVER LES UTILISATEURS QUI ONT DES PREFERENCES SIMILAIRES



# TODO
# interface graphique pour choisir nb facteur, reg, learnRate?
# reussir a obtenir un modele pour lequel err = 0
# possibilite d afficher diverses visualisation (par le biais de l interface graphique) -> plot ?
# train sur jester jokes

