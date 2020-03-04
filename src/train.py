
import os
import sys
import time
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from enum import Enum
from functools import wraps

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler,SMOTE,SVMSMOTE
from imblearn.keras import BalancedBatchGenerator
from imblearn import pipeline 

from trace import Trace
from model import create_model,is_batch_training
from visualization import plot_learning_curve
from utils import make_filename


class Resample(Enum):
    random_over_sampler   = 'random_over_sampler'
    random_under_sampler  = 'random_under_sampler'
    smote                 = 'smote'
    svmsmote              = 'svmsmote'
    balancedbatch         = 'balancedbatch'
    noresample            = 'noresample'
    all                   = 'all'

    def __str__(self):
        return self.value

        


num_epochs = 50
batch_size = 10


def fit_predict_imbalanced_model(model_name,num_classes,X_train, y_train, X_test, y_test,batch_size = batch_size,epochs=num_epochs):
    output_dim    = num_classes if num_classes > 0 else y_train.shape[1]
    model         = create_model(model_name,X_train.shape[1],output_dim)
    history       = model.fit(X_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(X_test, y_test))
    y_pred_proba  = model.predict_proba(X_test, batch_size=batch_size)
    y_pred        = model.predict(X_test, batch_size=batch_size)

    if num_classes > 0:                         # get the class label if necessary
        y_pred    = np.argmax(y_pred, axis=1)

    return model,y_pred,y_pred_proba,history



def fit_predict_balanced_model(model_name,num_classes,X_train, y_train, X_test, y_test,batch_size = batch_size,epochs=num_epochs):
    model               = create_model(model_name,X_train.shape[1],num_classes)
    training_generator  = BalancedBatchGenerator(X_train, y_train,
                                                batch_size=batch_size,
                                                random_state=42)
    valid_generator     = BalancedBatchGenerator(X_test, y_test,
                                                batch_size=batch_size,
                                                random_state=42)
    history             = model.fit_generator(generator=training_generator, epochs=epochs, verbose=1, validation_data=valid_generator)
    y_pred_proba        = model.predict_proba(X_test, batch_size=batch_size)
    y_pred              = model.predict(X_test, batch_size=batch_size)
    y_pred              = np.argmax(y_pred, axis=1)

    return model,y_pred,y_pred_proba,history
    

def fit_predict_model(model_name,X_train, Y_train, X_test, Y_test,class_names,resample):
    start_time  = time.time()
    history     = None
    num_feats   = X_train.shape[1]
    num_classes = len(class_names) if not class_names is None else 0
    
    if model_name[0:3] == 'MLP' or model_name =='DeepFM':
        
        if resample is None:
            model,y_pred,y_pred_proba,history = fit_predict_imbalanced_model(model_name,num_classes,X_train, Y_train, X_test, Y_test)
        else:
            model,y_pred,y_pred_proba,history = fit_predict_balanced_model(model_name,num_classes,X_train, Y_train, X_test, Y_test)
    else:
        X_t_train = X_train.copy()
        Y_t_train = Y_train.copy().ravel()
        X_t_test  = X_train.copy()
        Y_t_test  = Y_train.copy().ravel()

        if resample==Resample.random_under_sampler:
            rus                   = RandomUnderSampler(random_state=0)
            X_t_train, Y_t_train  = rus.fit_resample(X_t_train, Y_t_train)

        if resample==Resample.random_over_sampler:
            ros                   = RandomOverSampler(random_state=0)
            X_t_train, Y_t_train  = ros.fit_resample(X_t_train, Y_t_train)

        if resample==Resample.smote:
            sos                   = SMOTE(random_state=0)
            X_t_train, Y_t_train  = sos.fit_resample(X_t_train, Y_t_train)

        if resample==Resample.svmsmote:
            sos                   = SVMSMOTE(random_state=0)
            X_t_train, Y_t_train  = sos.fit_resample(X_t_train, Y_t_train)


        model           = create_model(model_name,num_feats,num_classes)
        lr_fit          = model.fit(X_t_train, Y_t_train)
        y_pred_proba    = model.predict_proba(X_test)
        y_pred          = model.predict(X_test)
        print(model.classes_)
        print(len(model.classes_) )

        #print(y_pred)
        #print(y_pred_proba.shape,len(class_names),np.argmax(y_pred_proba,axis=1))
        #assert y_pred_proba.shape[1] == len(class_names)
        if y_pred_proba.shape[1] < len(class_names):
            print("WARNING: Not all classes are present in output of predict_proba()")
            d               = len(class_names) - y_pred_proba.shape[1]
            z               = np.zeros((y_pred_proba.shape[0],d))
            y_pred_proba    = np.concatenate((y_pred_proba,z),axis=1)
    elapsed_time    = time.time() - start_time

    return elapsed_time, (model,y_pred,y_pred_proba,history)




def train_cv(trace,X,Y,class_names,model_name,cv=3,resample=None):
    if cv < 2:
        print("The number of folds for cross-validation need to be 2 or higher.")
        print("Use the --cv option.")
        sys.exit(1)
    
    if class_names is None:
        print("Using K-fold cross validation, no class information available.")
        kfold             = KFold(n_splits=cv, shuffle=True, random_state=0)
    else:
        print("Using stratefied K-fold cross validation.")
        kfold             = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)

    parms                 = { 'model'     : model_name,
                              'cv'        : cv,
                              'resample'  : str(resample) }

    print("Training model '" + model_name + "'.",flush = True) 
    for i,(train_idxs, test_idxs) in tqdm(enumerate(kfold.split(X, Y)),total=cv):

        X_t_train   = X[train_idxs]
        X_t_test    = X[test_idxs]
        Y_t_train   = Y[train_idxs]
        Y_t_test    = Y[test_idxs]

        elapsed_time, (model,y_pred,y_pred_proba,history) = fit_predict_model(model_name,X_t_train, Y_t_train, X_t_test, Y_t_test,class_names,resample)

        if i == 0:
            trace.write_model_info_and_training_info(model,parms)

        trace.append_history(history)
        trace.append_results(test_idxs,Y_t_test,y_pred,y_pred_proba,elapsed_time)


    trace.write_results()
     


def train(train_name,model_name,X_train, X_test, Y_train, Y_test,class_names,cv,resample,config=None,verbose=False):

    if (model_name == 'MLP' or model_name == 'DeepFM') and not resample is None and resample != Resample.balancedbatch:
        print("Resampling for model '" + model_name + "' can only be 'balancedbatch', not '" + str(resample) + "'.")
        sys.exit(1)

    if model_name != 'MLP' and model_name != 'DeepFM' and resample == Resample.balancedbatch:
        print("Resampling for model'" + model_name + "' cannot be 'balancedbatch', not'" + str(resample) + "'.")
        sys.exit(1)

    test        = not X_test is None and not Y_test is None

    if model_name is None:
        model_names     = create_model(None,0,0)
    elif isinstance(model_name,str):
        model_names     = [ model_name ]

    for model_name in model_names:
        if resample == Resample.all:
            if model_name[0:3] == 'MLP' or model_name == 'DeepFM':
                resample_methods = [ None, Resample.balancedbatch ]
            else:
                resample_methods = [ None, Resample.random_over_sampler ]
        else:
            resample_methods = [ resample ]

        for resample_method in resample_methods:
            if not cv is None and cv > 1:
                trace_cv   = Trace('cv',  train_name,model_name=model_name,resample=resample_method,y_shape=Y_train.shape,config=None,class_names=class_names)
                train_cv(trace_cv,X_train,Y_train,class_names=class_names,cv=cv,model_name=model_name,resample=resample_method)

            if test:
                trace_test = Trace('test',train_name,model_name=model_name,resample=resample_method,y_shape=Y_test.shape, config=None,class_names=class_names)

                elapsed_time, (model,y_pred,y_pred_proba,history) = fit_predict_model(model_name,X_train, Y_train, X_test, Y_test,class_names,resample_method)

                trace_test.append_history(history)
                trace_test.append_results(None,Y_test,y_pred,y_pred_proba,elapsed_time)
                trace_test.write_results()
                trace_test.write_model(model)



def train_from_file(train_name,data_filename,model_name,cv,resample,test_size=0.2,test=False,config=None,verbose=False):

    class_names                       = ['NoShow_No','NoShow_yes']
    X_train, X_test, Y_train, Y_test  = get_data(data_filename,verbose=False,test_size=test_size)

    if not test:
        X_test = None
        Y_test = None

    train(train_name,model_name,X_train, X_test, Y_train, Y_test ,class_names,cv,resample,config,verbose)


def create_learning_curve(train_name,data_filename,model_name,cv,resample,test_size=0.2,filename_out=None):

    X_train, _ , Y_train, _ = prepare_data(data_filename,verbose=False,test_size=test_size)
    model                   = create_model(model_name)
    train_sizes             = np.linspace(.1, 1.0, 5)
    resampler               = None

    if resample==Resample.random_under_sampler:
        resampler             = RandomUnderSampler(random_state=0)

    if resample==Resample.random_over_sampler:
        resampler             = RandomOverSampler(random_state=0)

    if resample==Resample.smote:
        resampler             = SMOTE(random_state=0)

    if resample==Resample.svmsmote:
        resampler             = SVMSMOTE(random_state=0)

    if not resampler is None:
        model                 = pipeline.make_pipeline(resampler,model)
  
    train_sizes, train_scores, test_scores = learning_curve(
                                  model, X_train, Y_train.ravel(), cv=cv, n_jobs=1, train_sizes=train_sizes, scoring = 'roc_auc')

    plot_learning_curve('Learning curve', train_sizes, train_scores, test_scores, ylim=None,filename_out = filename_out)



def get_data(folder,x_filename,y_filename,test_size=0.2,verbose = False):
    """
    """
    def load_data(filename,verbose):
        print("Loading '" + filename + "' ... ",end='')
        if filename[-4:] == '.pkl':
            df      = pd.read_pickle(filename)
            data    = df.to_numpy()
            names   = list(df.columns)[0].split(':')
        elif filename[-4:] == '.npy':
            data    = np.load(filename)
            names   = []

        print("Shape = " + str(data.shape) + '.')
        return data,names


    X,_     = load_data(make_filename(folder,None,x_filename,should_exist=True),verbose)

    # code for multi-target
    if isinstance(y_filename,list):
        Y       = []
        names   = None
        for filename in y_filename:
            tmp,_  = load_data(make_filename(folder,None,filename,should_exist=True),verbose)
            Y.append(to_categorical(tmp) )
        Y       = np.concatenate(Y,axis=1)
    else:
        Y,names = load_data(make_filename(folder,None,y_filename,should_exist=True),verbose)
  
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=2)

    if verbose:
        print("Train size (input) : ",X_train.shape)
        print("Train size (target): ",Y_train.shape)
        print("Test size (input)  : ",X_test.shape)
        print("Test size (input)  : ",Y_test.shape)
        print("Class names        : ",names)
    return X_train, X_test, Y_train, Y_test, names


if __name__ == '__main__':
    cv                                = 10
    model_name                        = 'DeepFM'
    model_name                        = 'MLP'
    model_name                        = 'LinearSVC'
    model_name                        = 'RadialSVC'
    model_name                        = 'LogisticRegression'
    model_name                        = 'Multinomial'
    model_name                        = 'RandomForest'
    model_name                        = 'MLP_MultiTarget'
    model_name                        = 'RandomForest'
    train_name                        = 'Run-001'
    resample                          = Resample.random_over_sampler
    resample                          = Resample.smote
    resample                          = Resample.balancedbatch
    resample                          = Resample.all
    resample                          = None

    if False:
        X_train, X_test, Y_train, Y_test,class_names  = get_data('../data/v1','audio-30.npy',['target-Smoker.pkl','target-Citrus fruits.pkl'],verbose=True)

        train(train_name,model_name,X_train, X_test, Y_train, Y_test,class_names,cv,resample,config=None,verbose=False)
    #create_learning_curve(train_name=train_name,data_filename='../data/Medical Appointments.csv',resample=Resample.random_over_sampler,
    #               cv=10,model_name=model_name,filename_out='../reports/final/results/lr_learning_cur.png')