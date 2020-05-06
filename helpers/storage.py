import pandas as pd
from helpers.fs_methods import *
from helpers.training import *
import os

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

def store(output_dir,cv_method):
    ''' INPUT

          output_dir: is the directory where to store the keymetrics (ranking, weights and accuracies dataframes)
          cv_method : 5_fold , 10_fold or LOO   

        OUTPUT

          stored pickle dataframes in the given directory
    '''

    # Initialize empty dataframes
    pool_FS=[reliefF]#,lap_score,ll_l21,ls_l21,UDFS,fisher_score,chi_square,gini_index,SPEC]

    labels=['reliefF']#,'lap_score','ll_l21','ls_l21','UDFS','fisher_score','chi_square','gini_index','SPEC']#,'Boratapy']
    dataframe_ranking=pd.DataFrame(index=num_fea,columns=labels)
    dataframe_weights=pd.DataFrame(index=num_fea,columns=labels)
    dataframe_accuracies=pd.DataFrame(index=num_fea,columns=labels)

    #matrix_=np.zeros((50,589*3))
    for i in range(len(pool_FS)):
        for k in num_fea:
            ranking__,acc__,weight__=training(cv_method,k,pool_FS[i],X,y)
            #ranking__,acc__=training(kf5,k,pool_FS[i],X,y)
            #ranking__,acc__,=training(kf5,k,pool_FS[i])
            dataframe_ranking[labels[i]][k]=ranking__
            dataframe_weights[labels[i]][k]=weight__
            dataframe_accuracies[labels[i]][k]=acc__ 

    #dataframe_ranking_5fold=dataframe_ranking.copy()
    #dataframe_weights_5fold=dataframe_weights.copy()
    #dataframe_accuracies_5fold=dataframe_accuracies.copy()

    name1,name2,name3=name_dataframe(X,cv_method)
    
    dataframe_accuracies.to_pickle(output_dir+name1)
    dataframe_ranking.to_pickle(output_dir+name2)
    dataframe_weights.to_pickle(output_dir+name3)