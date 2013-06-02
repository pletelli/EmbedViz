#!/usr/bin/python2.7
# coding: utf-8

import functions
import numpy as np
import operator


ratings= np.load("objects/ratings")
n= 3706 # nb de films
u= 6040 # nb utilisateurs

#notes triées par utilisateur
ratings.sort(order=['f1'])


#préparer la structure
common = np.empty((n,n),int)
for i in range (1,6041) : 
    # récupère la liste de tous les films ayant obtenu une note par l'utilisateur i
    for j,row in enumerate(ratings) :
        if(row[1]==i):
            break
    ind_deb = j
    for j,row in enumerate(ratings) :
        if (j >= ind_deb) and (row[1]!=i):
            break
    ind_fin = j
    dico = dict()
    for a in range (ind_deb, ind_fin) : 
        dico[str(ratings[a][0])] = ratings[a][2]
    #dico est un un dico film, note
    dico_trie = sorted(dico.iteritems(), reverse=True, key=operator.itemgetter(1))
    # on ne récupère que les notes = 5
    subdico= dict((k,v) for k,v in dico_trie if v == 5)
    films = subdico.keys()
    for k in films:
        films=films[1:]
        for l in films:
                common[int(k)-1][int(l)-1]=common[int(k)-1][int(l)-1]+1

for i in range(0,3706):
    for j in range(i+1,3706):
        if common[i,j] !=0:
            common[j,i] = common[j,i]+common[i,j]
            common[i,j]=0

np.savetxt("common1.csv", common, delimiter=",")
common.dump("objects/common1")

# write a csv with labels
f = open('labels.csv','w')
for i in range(0,3706):
    string = str(i+1)+","+'"'+ str(functions.getMovieName(a._moviesInformations,a._original_movieIDs[i]))+'"\n'
    f.write(string)

