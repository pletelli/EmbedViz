#!/usr/bin/python2.7
# coding: utf-8

import functions
import numpy as np
import operator


ratings= np.load("objects/ratings")
n= 3706 # nb de films
u= 6040 # nb utilisateurs

#notes triées par utilisateur
ratings.sort(order=['f0'])


#préparer la structure
common = np.empty((u,u),int)
for i in range (1,n) : 
    print i
    # récupère la liste de tous les utilisateurs ayant mis une note au film i
    for j,row in enumerate(ratings) :
        if(row[0]==i):
            break
    ind_deb = j
    for j,row in enumerate(ratings) :
        if (j >= ind_deb) and (row[0]!=i):
            break
    ind_fin = j
    dico = dict()
    for a in range (ind_deb, ind_fin) : 
        dico[str(ratings[a][1])] = ratings[a][2]
    #dico est un un dico user, note
    dico_trie = sorted(dico.iteritems(), reverse=True, key=operator.itemgetter(1))
    # on ne récupère que les notes = 5
    subdico= dict((k,v) for k,v in dico_trie if v == 5)
    users = subdico.keys()
    for k in users:
        users=users[1:]
        for l in users:
                common[int(k)-1][int(l)-1]=common[int(k)-1][int(l)-1]+1

for i in range(0,u):
    for j in range(i+1,u):
        if common[i,j] !=0:
            common[j,i] = common[j,i]+common[i,j]
            common[i,j]=0

np.savetxt("common_user.csv", common, delimiter=",")
common.dump("objects/common_user")

