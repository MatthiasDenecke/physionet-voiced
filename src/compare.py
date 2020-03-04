
import os
import json
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from visualization import plot_roc_curve1

rc={'font.size': 12, 'axes.labelsize': 24, 'legend.fontsize': 16.0, 
    'axes.titlesize': 32, 'xtick.labelsize': 12, 'ytick.labelsize': 8}
sns.set(rc=rc)


def show_roc_curve(y_test,y_pred,title,model_name,filename_out = None):
    plt.figure(figsize=(16, 12))
    plt.plot([0, 1], [0, 1], 'k--')
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    fpr, tpr, threshold = roc_curve(y_test.ravel(), y_pred.ravel() )
   
    plt.plot(fpr, tpr, label='{}, AUC = {:.3f}'.format(model_name, auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend()
    if filename_out is None:
        plt.show()
    else:
        plt.savefig(filename_out)    
        plt.close()



def show_cap_curve(y_test,y_pred,title,model_name,filename_out = None):
    plt.figure(figsize=(16, 12))
    
    total           = y_test.shape[0]
    class_1_count   = np.sum(y_test) 
    class_0_count   = total - class_1_count
    model_y         = [y for _, y in sorted(zip(y_pred, y_test), reverse = True)]
    y_values        = np.append([0], np.cumsum(model_y))
    x_values        = np.arange(0, total + 1)

    index           = int( (50 * total / 100) ) # Point where vertical line will cut trained model
    class_1_observed = y_values[index] * 100 / max(y_values)

    plt.plot([0, total], [0, class_1_count], c = 'r', linestyle = '--', label = 'Random Model')
    plt.plot([0, class_1_count, total],  [0, class_1_count, class_1_count],  c = 'grey', linewidth = 2, label = 'Perfect Model')
    plt.plot(x_values,y_values, c = 'b', label = model_name, linewidth = 4)

    plt.plot([index, index], [0, y_values[index]], c ='g', linestyle = '--')            ## 50% Verticcal line from x-axis
    plt.plot([0, index], [y_values[index], y_values[index]], c = 'g', linestyle = '--') ## Horizontal line to y-axis from prediction model
    plt.text(y_values[index] * 1.1, y_values[index] * 1.02,str(int(class_1_observed * 100.) / 100) + ' %' )
    plt.title(title)
    plt.legend()

    if filename_out is None:
        plt.show()
    else:
        plt.savefig(filename_out)    
        plt.close()

def show_results(data,title,filename_out):

    plt.figure(figsize=(16, 12))
    sns.boxplot(data=data,orient='h')
    sns.despine(top=True, right=True, left=True)
    plt.xlabel(title)
    plt.ylabel('')
    plt.title('Results for ' + title)

    if filename_out is None:
        plt.show()
    else:
        plt.savefig(filename_out)    
        plt.close()


def create_roc_curve(data,name,sample,fn_png,d):
    print('  Creating',fn_png)
    y_test = data['y_test'].astype(int)
    y_pred = np.argmax(data['y_prob'], axis=1)
    show_roc_curve(y_test,y_pred,'ROC Curve ' + name + ' (' + sample + ')',name,fn_png)
    

def create_cap_curve(data,name,sample,fn_png,d):
    print('  Creating',fn_png)
    show_cap_curve(data['y_test'],data['y_prob'][:,1],'CAP Curve ' + name + ' (' + sample + ')',name,fn_png)


def create_comparisons(dataset,names,samples,foldername,tpe):
    if len(dataset) == 0:
        return

    keys = dataset[0].keys()
    v    = {}

    if not os.path.exists(foldername):
        print("  Creating folder '" + foldername + "'.")
        os.makedirs(foldername)

    for key in keys:
        v[key] = {}

    for (data,name,sample) in zip(dataset,names,samples):
        for key in keys:
            v[key][name + ' ' + sample]  = data[key]['values']

    for key in keys:
        print(key)
        print(v[key])
        df = pd.DataFrame(v[key])
        fn = os.path.join(foldername,key + '.' + tpe + '.png')
        print('  Creating',fn)
        show_results(df,key,fn)
 



def create_graphics(train_name,create_func,create_func2,fn_data,fn_out,tpe):
    d       = '../results/' + train_name
    results = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o)) and o != 'comparisons' ]
    dataset = []
    names   = []
    samples = []
   
    for result in results:
        name   = result[len(d)+1:].split('-')[1]
        sample = result[len(d)+1:].split('-')[2]
        fn_res = os.path.join(result,fn_data)
        fn_png = os.path.join(result,fn_out)
       
        if os.path.exists(fn_res) and os.path.isfile(fn_res):
            if fn_res[-4:] == '.npz':
                data = np.load(fn_res)
            elif fn_res[-5:] == '.json':
                with open (fn_res) as f:
                    data = json.load(f)

            if not create_func is None:
                create_func(data,name,sample,fn_png,d)

            dataset.append(data)
            names.append(name)
            samples.append(sample)

    if not create_func2 is None:
        create_func2(dataset,names,samples,os.path.join(d,fn_out),tpe)


def compare(train_name):
    create_graphics(train_name,None,             create_comparisons, 'measures.cv.json',    'comparisons','cv')
    create_graphics(train_name,None,             create_comparisons, 'measures.test.json',  'comparisons','test')
    create_graphics(train_name,create_roc_curve, None,               'predictions.test.npz','roc_curve.png','test')
    create_graphics(train_name,create_cap_curve, None,               'predictions.test.npz','cap_curve.png','test')


if __name__ == '__main__':
    compare('TestRun')
    sys.exit(1)
    fn = '../results/TestRun/train-logisticregression-smote-(default)/predictions.test.npz'

    l  = np.load(fn)
    print(type(l))
    for key in l.keys():
        print(key)
    