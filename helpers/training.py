import scipy.io as sio
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
from helpers.fs_methods import *

#kf5=KFold(n_splits=5,shuffle=True)
#kf10=KFold(n_splits=5,shuffle=True)
#loo=LeaveOneOut()

# Number of selected features (top k ranked features)
#top K selected features varying from 10 to 100 (with a step size of 10 features)
num_fea = [i for i in range(10,110,10)]
clf = svm.LinearSVC()    # linear SVM
correct=0
accuracy=[]

def normalize_ranks(X,feature_rankings,cv):
    p=cv.get_n_splits(X)
    mat_temp=np.zeros((595,2))
    mat_temp[:,0]=[i for i in range(595)]
    mat_temp[:,1]=feature_rankings.sum(axis=0)/p
    labels=['new_ranks','normalized_ranks']
    df_ranks=pd.DataFrame(mat_temp,columns=labels)
    df_ranks=df_ranks.sort_values(by ='normalized_ranks')
    newranks=list(df_ranks.new_ranks.values.astype('int'))
    return(newranks)
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#import statistics
#X_train=X
#y_train=y

def training(cv,k,FS_method,X_train,y_train):
    accuracy=[]
    featureranking=[]
    featureweight=[]
    normalized_rank_=[]
    correct=0
    selected_features_=[]
    for train_index, test_index in cv.split(X_train):
        X_train1, X_test1 = X_train[train_index], X_train[test_index]
        y_train1, y_test1 = y_train[train_index], y_train[test_index]
        if FS_method==reliefF:
            idx,score=relief_FS(X_train1,y_train1)
        #elif FS_method==MIFS: ##ERROR
        #    idx,score=MIFS_FS(k,X_train,y_train)
        elif FS_method==lap_score:
            #ok=True
            idx,score=lap_score_FS(X_train1)
        elif FS_method==ll_l21:
            #ok=True
            idx,score=ll_l21_FS(X_train1,y_train1)
        elif FS_method==UDFS:
            #ok=True
            idx,score=UDFS_FS(X_train1)
        elif FS_method==fisher_score:
            #ok=True
            idx,score=fisher_score_FS(X_train1,y_train1)
        elif FS_method==chi_square:
            #ok=True
            idx,score=chi_square_FS(X_train1,y_train1)
        elif FS_method==gini_index:
            #ok=True
            idx,score=gini_index_FS(X_train1,y_train1)
        #elif FS_method==FCBF:
        #    idx=FCBF_FS(X_train,y_train,k)
       # elif FS_method==BorutaPy:
            #ok=False
       #     idx=boruta_FS(X_train,y_train) 
        #elif FS_method==trace_ratio:
        #    idx=trace_ratio_FS(X,train_index,y_train)
        elif FS_method==SPEC:
           # ok=True
            idx,score=spec_FS(X_train1)
        #elif FS_method==CIFE:#Takes too long 
        #    idx=CIFE_FS(X_train,y_train)
        #elif FS_method==alpha_investing:#Error
        #    idx=alpha_investing_FS(X_train,y_train)
        #elif FS_method==CMIM:# Same ranking of the original features 
         #   n,m=X_train.shape
         #   idx=CMIM_FS(X_train,y_train,m)
        elif FS_method==ls_l21:
            #ok=True
            idx,score=ls_l21_FS(X_train1,y_train1)
        #elif FS_method==MCFS:
         #   idx=MCFS_FS(X_train,k)
        selected_features = X_train1[:, idx[0:k]]
        selected_features_test=X_test1[:,idx[0:k]]
        #selected_features_.extend(selected_features)
        #normalized_rank=sum(score[0:k])/77
        #normalized_rank_.append(normalized_rank)
        featureranking.extend([idx])
        featureweight.extend([score[0:k]])
        # train a classification model with the selected features on the training dataset
        clf.fit(selected_features, y_train1)  # predict the class labels of test data
        y_predict = clf.predict(selected_features_test)
        # obtain the classification accuracy on the test data
        acc = accuracy_score(y_test1, y_predict)
        #accuracy.append(acc)
        correct = correct + acc

        # output the average classification accuracy over all folds
    #featureranking=idx[0:k]
    #featureweight=score[0:k]
    newranks=normalize_ranks(X_train,np.array(featureranking),cv)
    accuracy=float(correct)/cv.get_n_splits(X_train)
    #return(np.array(newranks[0:k]),accuracy)
   # if ok==True:
    return(np.array(newranks)[0:k],accuracy,np.array(featureweight))
    #else:
     #   return(np.array(newranks)[0:k],accuracy)
    #return(np.array(newranks[0:k]),accuracy)#,np.array(featureweight))
    #return(featureranking,accuracy,featureweight)
    #return(accuracy,np.array(normalized_rank_),featureranking,featureweight)

def name_dataframe(X,cv_method):
    n,m=X.shape

    name1='df_accuracy_'
    name2='df_ranking_'
    name3='df_weights_'
    added_str='fold.pkl'
    num_splits =cv_method.get_n_splits(X)
    if num_splits!=n:
        name1=name1+str(num_splits)+added_str
        name2=name2+str(num_splits)+added_str
        name3=name3+str(num_splits)+added_str
    else:
        name1=name1+'LOO'
        name2=name2+'LOO'
        name3=name3+'LOO'
    return(name1,name2,name3)

def fs_training(cv_method,X,y,num_fea):
    pool_FS=[reliefF,lap_score]
    labels=['reliefF','lap_score']
    #pool_FS=[reliefF,lap_score,ll_l21,ls_l21,UDFS,fisher_score,chi_square,gini_index,SPEC]

    #labels=['reliefF','lap_score','ll_l21','ls_l21','UDFS','fisher_score','chi_square','gini_index','SPEC']#,'Boratapy']

    dataframe_ranking=pd.DataFrame(index=num_fea,columns=labels)
    dataframe_weights=pd.DataFrame(index=num_fea,columns=labels)
    dataframe_accuracies=pd.DataFrame(index=num_fea,columns=labels)

    #matrix_=np.zeros((50,589*3))
    for i in range(len(pool_FS)):
        for k in num_fea:
            ranking__,acc__,weight__=training(cv_method,k,pool_FS[i],X,y)
            #ranking__,acc__=training(kf5,k,pool_FS[i],X,y)
            #ranking__,acc__,=training(kf5,k,pool_FS[i])
            print('FS method :',labels[i],'num_fea = ',k)
            dataframe_ranking[labels[i]][k]=ranking__
            dataframe_weights[labels[i]][k]=weight__
            dataframe_accuracies[labels[i]][k]=acc__ 

    #dataframe_ranking_5fold=dataframe_ranking.copy()
    #dataframe_weights_5fold=dataframe_weights.copy()
    #dataframe_accuracies_5fold=dataframe_accuracies.copy()

    #name1,name2,name3=name_dataframe(X,cv_method)
    return(dataframe_ranking,dataframe_accuracies,dataframe_weights)


def store_training_keymetrics(output_dir,cv_method,X,y,num_fea):
    ''' INPUT

          output_dir: is the directory where to store the keymetrics (ranking, weights and accuracies dataframes)
          cv_method : 5_fold , 10_fold or LOO   

        OUTPUT

          stored pickle dataframes in the given directory
    '''

    # Initialize empty dataframes
    '''pool_FS=[reliefF,lap_score,ll_l21,ls_l21,UDFS,fisher_score,chi_square,gini_index,SPEC]

    labels=['reliefF','lap_score','ll_l21','ls_l21','UDFS','fisher_score','chi_square','gini_index','SPEC']#,'Boratapy']
    dataframe_ranking=pd.DataFrame(index=num_fea,columns=labels)
    dataframe_weights=pd.DataFrame(index=num_fea,columns=labels)
    dataframe_accuracies=pd.DataFrame(index=num_fea,columns=labels)

    #matrix_=np.zeros((50,589*3))
    for i in range(len(pool_FS)):
        for k in num_fea:
            ranking__,acc__,weight__=training(cv_method,k,pool_FS[i],X,y)
            #ranking__,acc__=training(kf5,k,pool_FS[i],X,y)
            #ranking__,acc__,=training(kf5,k,pool_FS[i])
            print('FS method :',labels[i],'num_fea = ',k)
            dataframe_ranking[labels[i]][k]=ranking__
            dataframe_weights[labels[i]][k]=weight__
            dataframe_accuracies[labels[i]][k]=acc__ '''

    dataframe_ranking,dataframe_accuracies,dataframe_weights = fs_training(cv_method,X,y,num_fea)
    name1,name2,name3=name_dataframe(X,cv_method)
    dataframe_accuracies.to_pickle(output_dir+name1)
    dataframe_ranking.to_pickle(output_dir+name2)
    dataframe_weights.to_pickle(output_dir+name3)