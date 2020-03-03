
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
import pandas as pd
import numpy as np
import collections
import sys

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from data import load_data_csv,clean_data
from sklearn.decomposition import PCA



def set_fonts():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font',      size=SMALL_SIZE)        # controls default text sizes
    plt.rc('axes',      titlesize=BIGGER_SIZE)   # fontsize of the axes title
    plt.rc('axes',      labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick',     labelsize=SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('ytick',     labelsize=SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('legend',    fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure',    titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_histogram(data,column_name,title=None,order_by_count=False,filename_out = None,figsize=(20,10) ):
    if title is None:
        title = column_name
    plt.figure(figsize=figsize)
    plt.rc('axes', titlesize=12)     # fontsize of the axes title
    plt.rc('axes', labelsize=10)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=6)    # fontsize of the tick labels

    if order_by_count:
        ax = sns.countplot(data[column_name],order=data[column_name].value_counts().index)
    else:
        ax = sns.countplot(data[column_name])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
    plt.title(title)
    plt.tight_layout()

    #for label in plt.axes().get_xticklabels():
    #    label.set_rotation(90)
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()


def plot_histograms(data,column_names,filename_out = None):
    fig, ax = plt.subplots(4, 3, figsize=(20, 10))
    for column_name, subplot in zip(column_names, ax.flatten()):
        sns.countplot(data[column_name], ax=subplot)
        for label in subplot.get_xticklabels():
            label.set_rotation(90)
        #subplot.xaxis.set_major_locator(plt.MaxNLocator(3))
        #subplot.yaxis.set_major_locator(plt.MaxNLocator(3))

 


def plot_correlation(df,filename_out = None):
    plt.figure(figsize=(10,8))
    print(df.columns)

    sns.heatmap(df.corr(),cmap='Blues',annot=False) 
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()


def plot_correlation2(df,filename_out = None):
    k       = len(df.columns) #number of variables for heatmap
    print(df.columns)
    print(df.corr())
    cols    = df.corr()['No-show'].index
    cm      = df[cols].corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, cmap = 'viridis')
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()


def plot_outliers(f,filename_out = None):
    l                   = [ 'Age' , 'Delay' ]#df.columns.values
    number_of_columns   = len(l)
    number_of_rows      = len(l)-1/number_of_columns
    plt.figure(figsize=(number_of_columns,5*number_of_rows))
    for i in range(0,len(l)):
        plt.subplot(number_of_rows + 1,number_of_columns,i+1)
        sns.set_style('whitegrid')
        sns.boxplot(df[l[i]],color='green',orient='v')
        plt.tight_layout()
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()



def plot_history_cv(loss,acc,filename_out = None):
    fig, ax       = plt.subplots(1, 2,figsize=(12,7))
    #training_loss = d['loss']
    #test_loss     = d['val_loss']
    #training_acc  = d['accuracy']
    #test_acc      = d['val_accuracy']
    #epoch_count   = range(1, len(training_loss) + 1)

    fig.suptitle('Loss and Accuracy')
    sns.lineplot(x="epoch", y="value", hue="measure",data=loss,ax=ax[0])
    sns.lineplot(x="epoch", y="value", hue="measure",data=acc ,ax=ax[1])
    
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()




def plot_confusion_matrix(y_test,y_pred,class_names,filename_out = None):

    conf_mat = confusion_matrix(y_test, y_pred,normalize='all')
    text     = 'Confusion matrix\n'
    text     = text + '|             |' + '|'.join(['{0:13}'.format(class_name) for class_name in class_names]) + '|\n'
    text     = text + '|' + ' -----------:|' * (len(class_names) + 1) + '\n'

    for i,class_name in enumerate(class_names):
        text = text + '|{0:12} |'.format(class_name)
        text = text + '|'.join(['{0:13.2f}'.format(conf_mat[i][j]) for j in range(len(class_names))]) + '|\n'
        

    conf_mat = confusion_matrix(y_test, y_pred,normalize='all')
    plt.figure(figsize=(4,4))
    sns.heatmap(conf_mat, annot=True, fmt='f', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()

    return text



def plot_roc_curve1(Y_test,Y_pred,title,model_name,filename_out = None):
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    lb = LabelBinarizer()
    lb.fit(Y_test)
    Y_test = lb.transform(Y_test)
    
        #Y_pred    = model.predict(X_test)
    Y_pred    = lb.transform(Y_pred)
    fpr, tpr, threshold = roc_curve(Y_test.ravel(), Y_pred.ravel())
        
    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model_name, auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc="best")
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()


def plot_learning_curve(title, train_sizes, train_scores, test_scores, ylim=None,filename_out = None):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",label="Cross-validation score")

    plt.legend(loc="best")

    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()


def plot_class_distribution_histogram(df, col,ax=None,filename_out = None,title=None,order=None):
    if title is None:
        title = 'Distribution of ' + col

    if order == 'count_asc':
        order = df[col].value_counts().index
    elif order == 'count_desc':
        order = reverse(df[col].value_counts().index)
    elif order == 'alphabetical':
        order = list(df[col].value_counts().index)
        order.sort()

    plt.figure(figsize=(6,6))
    plt.subplots_adjust(bottom=0.4)

    plt.rc('axes', titlesize=12)         # fontsize of the axes title
    plt.rc('axes', labelsize=10)         # fontsize of the x and y labels

    values = df[col].unique()
    max    = np.max([len(str(v) ) for v in values])
    if len(values) > 10 or max > 20:
        plt.xticks(rotation=90)
        plt.rc('xtick', labelsize=8)
        if len(values) > 30:
            plt.rc('xtick', labelsize=6)
    elif len(values) > 5 or max > 10:
        plt.xticks(rotation=45)
        plt.rc('xtick', labelsize=10)


    sns.countplot(df[col],palette="PuBuGn_d",order=order)
    plt.title(title)
    if filename_out is None:
        print('Showing ' + col)
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()




def plot_pca(data,title,filename_out = None,figsize=(8,8)):
    set_fonts()
    pca = PCA().fit(data)
    plt.figure(figsize=figsize)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') 
    plt.title(title)
    if filename_out is None:
        plt.show()
    else:
        print('  Creating',filename_out)
        plt.savefig(filename_out)
        plt.close()



if __name__ == '__main__':
    pass


    