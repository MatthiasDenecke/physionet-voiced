
# code to disable tensorflow warnings
import warnings
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import sys
import os
import re 
import argparse

from utils import make_filename
from data import load_data_csv,clean_data,describe_data,encode_data
from train import train,Resample,create_model
from compare import compare
from visualization import plot_class_distribution_histogram, plot_pca,plot_correlation
from sklearn.decomposition import PCA


scale5 = ['never','almost never','sometimes','almost always','always']
meta   = {  'Age'                                       : { 'encoding' : 'float', 'order' : None },
            'Gender'                                    : { 'encoding' : 'label', 'order' : None },
            'Diagnosis'                                 : { 'encoding' : 'label', 'order' : 'alphabetical' },
            'Occupation status'                         : { 'encoding' : 'label', 'order' : 'alphabetical' },
            'Voice Handicap Index (VHI) Score'          : { 'encoding' : 'float', 'order' : None },
            'Reflux Symptom Index (RSI) Score'          : { 'encoding' : 'float', 'order' : None },
            'Smoker'                                    : { 'encoding' : 'label', 'order' : [ 'no','casual smoker','yes' ] }, 
            'Number of cigarettes smoked per day'       : { 'encoding' : 'float', 'order' : None }, 
            'Alcohol consumption'                       : { 'encoding' : 'label', 'order' : [ 'nondrinker','casual drinker','habitual drinker' ] },
            "Amount of water's litres drink every day"  : { 'encoding' : 'float', 'order' : None },
            'Carbonated beverages'                      : { 'encoding' : 'label', 'order' : scale5 },
            'Tomatoes'                                  : { 'encoding' : 'label', 'order' : scale5 }, 
            'Coffee'                                    : { 'encoding' : 'label', 'order' : scale5 }, 
            'Chocolate'                                 : { 'encoding' : 'label', 'order' : scale5 }, 
            'Soft cheese'                               : { 'encoding' : 'label', 'order' : scale5 }, 
            'Citrus fruits'                             : { 'encoding' : 'label', 'order' : scale5 } }



def explore_targets(target_filename,meta,report_folder=None,verbose=True):
    df      = load_data_csv(target_filename,verbose=verbose)
    df      = clean_data(df,verbose=verbose)

    with open(make_filename(report_folder,None,'Exploratory.md'),'w') as file:
        for column in df.columns:
            n   = column.replace(' ','_')
            fn  = make_filename(report_folder,'exploratory',n + '.png')
            plot_class_distribution_histogram(df,column,order=meta[column]['order'],filename_out=fn)
            describe_data(df,column,file,os.path.join('exploratory',n + '.png'))

        _,df,_    = encode_data(df,meta=meta,verbose=verbose,folder=None,name=None)
        c = df.corr().to_numpy()
        for i in range(c.shape[1]):
            c[i][i] = 0
        print(np.max(c),np.min(c))

        fn  = make_filename(report_folder,'exploratory','correlation.png')

        plot_correlation(df,filename_out = fn)


        print('',file=file)
        print('#### Correlation of the predicted variables',file=file)

        print('',file=file)
        print("![correlation](" + fn + ")",file=file)
        print('',file=file)



def explore_audio(audio_filename,report_folder=None,verbose=True):
    df      = pd.read_csv(audio_filename)
    fn      = make_filename(report_folder,'exploratory','pca.png')
    del df['ID']

    data        = df.to_numpy() #scaler.fit_transform(df)
    plot_pca(data,"Audio Data Number of Components vs Variance",filename_out = fn)


def explore(audio_filename,target_filename,meta,report_folder=None,verbose=True):
    explore_audio(audio_filename,report_folder,verbose)
    explore_targets(target_filename,meta,report_folder,verbose)


def prepare(audio_filename,target_filename,meta,n_components,folder,verbose=True):

    df      = load_data_csv(target_filename,verbose=False)
    df      = clean_data(df,verbose=verbose)
    encode_data(df,meta=meta,verbose=verbose,folder=folder,name='target')

    df      = pd.read_csv(audio_filename)
    del df['ID']

    for n in n_components:
        fn      = make_filename(folder,None,'audio-' + str(n) + '.npy')
        if n is None:
            X   = df.to_numpy()
        else:
            pca     = PCA(n_components=n)
            pca.fit(df)
            X       = pca.transform(df)
        np.save(fn,X)
        if verbose:
            print("Shape of compressed audio = " + str(X.shape))
            print("Creating '" + fn + "'.")


def basic():
    audio_filename  = '../data/audio_features.csv'  
    target_filename = '../data/individual_features.csv'  
    report_folder   = '../reports/final'
    verbose         = True
    n_components    = [15,20,30,None]
    folder          = '../data/v1'
    train_name      = 'Run-001'

    #explore(audio_filename,target_filename,meta,report_folder,verbose)
    #prepare(audio_filename,target_filename,meta,n_components,folder,verbose)
    compare(train_name=train_name)

    sys.exit(1)

def main():
    
    parser      = argparse.ArgumentParser(description="Train and evaluate models for no-show prediction. Use 'python3 main.py {command} --help' for more information on the individual commands.")
    subparsers  = parser.add_subparsers(help='Command to execute',dest='command') 
    
    # create parser for the eda command
    p_eda   = subparsers.add_parser('explore', help='Exploratory data analysis')
    p_eda.add_argument('--audio_filename',        type=str,       help='File name from which to load the audio data.',default=None,required=True)
    p_eda.add_argument('--target_filename',       type=str,       help='File name from which to load the target data.',default=None,required=True)
    p_eda.add_argument('--report_folder',         type=str,       help='Folder in which the report resides.',default=None,required=True)

    # create parser for the tune command
    p_eval = subparsers.add_parser('tune', help='Hyperparameter tuning and comparison of different classifiers')
    p_eval.add_argument('--run',            type=str,       help='Name of the training run',default='TestRun',required=False)
    p_eval.add_argument('--filename',       type=str,       help='File name from which to load the training data.',default=None,required=True)
    p_eval.add_argument('--resample',       type=Resample,  help='Resample training data',choices = list(Resample),required=False,default=None)
    p_eval.add_argument('--model',          type=str,       help='Model name (if not specified evaluate all models)',default=None,choices=list(create_model(None)))
    p_eval.add_argument('--cv',             type=int,       help='Number of splits for cross-validation.',default=3,required=False)
    #p_eval.add_argument('--config',         type=str,       help='Name of the configuration file to be used (optional)',default=None,required=False)
    p_eval.add_argument('--frac',           type=float,     help='Fraction of the test set size',default=0.2,required=False)
    
    # create parser for the tune command
    p_eval = subparsers.add_parser('train', help='Training the model without cross validation')
    p_eval.add_argument('--run',            type=str,       help='Name of the training run',default='TestRun',required=False)
    p_eval.add_argument('--filename',       type=str,       help='File name from which to load the training data.',default=None,required=True)
    p_eval.add_argument('--resample',       type=Resample,  help='Resample training data',choices = list(Resample),required=False,default=None)
    p_eval.add_argument('--model',          type=str,       help='Model name (if not specified evaluate all models)',default=None,choices=list(create_model(None)))
    #p_eval.add_argument('--config',         type=str,       help='Name of the configuration file to be used (optional)',default=None,required=False)
    p_eval.add_argument('--frac',           type=float,     help='Fraction of the test set size',default=0.2,required=False)

    # create parser for the predict command
    p_compare = subparsers.add_parser('compare', help='Compares all classifiers in the same training run and creates comparison graphics')
    p_compare.add_argument('--run',         type=str,       help='Name of the training run',default='TestRun',required=False)


    args        = parser.parse_args()
    args.config = None

    if args.command == 'explore':
        explore(args.audio_filename,args.target_filename,meta,args.report_folder)
    elif args.command == 'tune':
        train(train_name=args.run,data_filename=args.filename,model_name=args.model,cv=args.cv,resample=args.resample,config=args.config,test=False)
    elif args.command == 'train':
        train(train_name=args.run,data_filename=args.filename,model_name=args.model,cv=None,resample=args.resample,config=args.config,test=True)
    elif args.command == 'compare':
        compare(train_name=args.run)
    elif args.command is None:
        print("Use '--help' to get a list of all commands.")
        sys.exit(1)
    else:
        print("Illegal command '" + str(args.command) + "'.")
        sys.exit(1)      


if __name__ == '__main__':
    basic()
    main()

