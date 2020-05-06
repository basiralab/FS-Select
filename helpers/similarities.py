import numpy as np
from statistics import mean
import math
from helpers.fs_methods import *

pool_FS=[reliefF,lap_score,ll_l21,ls_l21,UDFS,fisher_score,chi_square,gini_index,SPEC]
# Overlapping matrices

def intersection(a,b):
    ''' A function that return the percentage of common elements to both a and b'''
    sum_=0
    for x in range(a.shape[0]):
       # for y in range(a.shape[1]):
        sum_+=np.sum((a[x]==b[x]))
    #percentage=(sum_*100)/a.size
    percentage=(sum_)/a.size
    return(percentage)

def FS_to_FS_similarity(FS_k):
    n=len(FS_k)
    m=np.ones((n,n))
    for i in range(n):
        for j in range(i+1,n):
            m[i,j]=intersection(FS_k[i],FS_k[j])
            m[j,i]=m[i,j]
    for i in range(n):
        m[i,i]=intersection(FS_k[i],FS_k[i])#/100
    return(m)

# Accuracy similarity matrix 

def cost(ai,aj):
    if abs(ai-aj)<0.05:
        return(20)
    #sigma=10
    #return(math.exp(-abs(ai-aj)/sigma))
    else:
        return(1/(abs(ai-aj)))
def matrix_acc(list_):
    n=len(list_)
    m=np.ones((n,n))
    for i in range(n-1):
        for j in range(i+1,n):
            m[i,j]=cost(list_[i],list_[j])
            m[j,i]=m[i,j]
    for i in range(n):
        m[i,i]=20
    
    return(m)

# Stability Construction

def kuncheva_stability(featidx,d):
    q,k = featidx.shape
    #q = size(featidx,1);
    #q=num_methods
    r = np.zeros((q,q))
    for n in range(1,q-1):
        for m in range(n,q-1):
            r[n,m] = len(np.intersect1d(featidx[n,0],featidx[n,1])) + len(np.intersect1d(featidx[n,0],featidx[n + 1,1])) + len(np.intersect1d(featidx[n,0],featidx[n-1,1]))
    A = np.true_divide(np.subtract(r, (k**2/d)),(k-(k**2/d)))
    S = 2*sum(sum(A))/(q*(q-1))
    return(S)

def get_ranking_matrix(a,k):
    #The columns are the respective rankings of pool_FS
    #a=dataframe_ranking_5fold.loc[10]
    #a=a.as_matrix(columns=None)
    a=np.array(a)
    b=np.zeros((k,len(pool_FS)))
    for i in range(len(a)):
        b[:,i]=a[i]
    return(b)

def normalize_stability(matrix):
    n,m=matrix.shape
    norm_mat = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            norm_mat[i,j] = 100*(matrix[i,j] - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    return(norm_mat)

def get_stability(ranking_matrix):
    
    stab_mat=np.zeros((len(pool_FS),len(pool_FS)))
    for i in range(len(pool_FS)):
        for j in range(len(pool_FS)):
            if i!=j:
                stab_mat[i,j] = kuncheva_stability(np.transpose(np.vstack((ranking_matrix[:,i],ranking_matrix[:,j]))),595)
    stab_mat=np.abs(stab_mat)
    stab_mat=normalize_stability(stab_mat)
    return(stab_mat)

def rowStability(row):
    matrix=get_ranking_matrix(row,row.name)
    return(get_stability(matrix))


