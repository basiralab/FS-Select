import numpy as np


def binary_intersection(a,b,k):
    ''' A function that return the percentage of common elements to both a and b'''
    sum_=0
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            sum_+=np.sum((a[x,y]==b[x,y]))
    #percentage=(sum_*100)/k
    percentage=(sum_*100)/b.size
    return(percentage)

def binary_matrice(FS_k,k):
    n=len(FS_k)
    m=np.ones((n,n))
    for i in range(n):
        for j in range(i+1,n):
            m[i,j]=binary_intersection(FS_k[i],FS_k[j],k)
            m[j,i]=m[i,j]
   # for i in range(n):
    #    m[i,i]=intersection(FS_k[i],FS_k[i])/100
    return(m)