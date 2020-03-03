

import os
import json
import sys

import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score,accuracy_score,confusion_matrix,auc,mean_squared_error
from joblib import dump

from visualization import plot_confusion_matrix,plot_history_cv,plot_roc_curve1


def _form_folder_name(category,model_name,resample,config_name = None):
    folder_name = category + '-' + model_name + '-' + str(resample)
    if config_name == None:
        folder_name = folder_name + '-(default)'
    else:
        folder_name = folder_name + '-(' + str(config_name) + ')'
    return folder_name.lower()



def get_foldername_results(filename_in,name,overwrite = True):
    dir             = '../results'
    data            = os.path.splitext(os.path.basename(filename_in))[0]
    foldername      = os.path.join(dir,data,name)

    if not os.path.exists(foldername):
        print("  Creating folder '" + foldername + "'.")
        os.makedirs(foldername)
    elif overwrite:
        print("  Overwriting files in folder '" + foldername + "'.")
    else:
        print('')
        print("Path '" + foldername + "' already exists.")
        sys.exit(1)

    return foldername


def cap_auc_score(y_test, y_pred_proba,multi_class='raise'):
    """
        1. Calculate the area under the perfect model (aP) till the random model (a)
        2. Calculate the area under the prediction model (aR) till the random model (a)
        3. Calculate Accuracy Rate (AR) = aR / aP
        https://towardsdatascience.com/machine-learning-classifier-evaluation-using-roc-and-cap-curves-7db60fe6b716
    """
    
    total           = y_test.shape[0]
    class_1_count   = np.sum(y_test) 
    class_0_count   = total - class_1_count
    print(y_pred_proba)
    print(y_test)
    sys.exit(1)
    model_y         = [y for _, y in sorted(zip(y_pred_proba, y_test), reverse = True)]
    y_values        = np.append([0], np.cumsum(model_y))
    x_values        = np.arange(0, total + 1)

    a               = auc([0, total], [0, class_1_count])                                   # Area under Random Model
    aP              = auc([0, class_1_count, total], [0, class_1_count, class_1_count]) - a # Area between Perfect and Random Model
    aR              = auc(x_values, y_values) - a                                           # Area between Trained and Random Model

    return aR / aP

def roc_auc_score_FIXED(y_true, y_pred,num_classes):
    if len(np.unique(y_true)) < num_classes: # bug in roc_auc_score
        print(y_true,y_pred, np.argmax(y_pred,axis=1))
        
        return accuracy_score(y_true, np.argmax(y_pred,axis=1))
    return roc_auc_score(y_true, y_pred,multi_class='ovr')

class Trace:
    """
        Trace is a class that records information from a training run and stores it
        in json files from which reports can be generated at a later point.
    """
    def __init__(self,name,foldername,model_name,resample,config,y_shape,class_names,output_dim = 0):
        self.training_run_name  = _form_folder_name('train',model_name,resample,config)
        self.foldername         = get_foldername_results(foldername,self.training_run_name)
        self.history_loss       = None
        self.history_acc        = None
        self.name               = name
        self.model_name         = model_name

        self.measures           = { 'accuracy'  : { 'values' : [], 'mean' :0., 'std' : 0. },
                                    'f1'        : { 'values' : [], 'mean' :0., 'std' : 0. },
                                    'precision' : { 'values' : [], 'mean' :0., 'std' : 0. },
                                    'recall'    : { 'values' : [], 'mean' :0., 'std' : 0. },
                                    'roc_auc'   : { 'values' : [], 'mean' :0., 'std' : 0. },
                                    'cap_auc'   : { 'values' : [], 'mean' :0., 'std' : 0. },
                                    'time'      : { 'values' : [], 'mean' :0., 'std' : 0. } }

        self.multi_target       = class_names is None
        if self.multi_target:
            self.num_classes    = 0
            self.output_dim     = output_dim
        else:
            self.num_classes    = len(class_names)
            self.output_dim     = len(class_names)

        self.class_names        = class_names
        self.y_pred             = np.zeros(shape=y_shape)
        self.y_prob             = np.zeros(shape=(y_shape[0],self.num_classes) )
        self.y_test             = np.zeros(shape=y_shape)


           
    def append_history(self,history):
        """
            This method appends the history (i.e, accuracy and loss) to the trace.
        """
        if history is None:
            return

        if self.history_loss is None:
            self.history_loss   = pd.DataFrame({ 'epoch':[],'measure':[],'value':[] } )
            self.history_acc    = pd.DataFrame({ 'epoch':[],'measure':[],'value':[] } )

        for key in history.history.keys():
            x = pd.DataFrame({ 'epoch'    : range(len(history.history[key]) )   ,
                               'measure'  : [ key ] * len(history.history[key]) ,
                               'value'    : history.history[key] })
            if key in ['loss','val_loss']:
                self.history_loss  = pd.concat([self.history_loss,x])
            else:
                self.history_acc   = pd.concat([self.history_acc ,x])


    def append_results(self,valid_idxs,y_test,y_pred, y_pred_proba,elapsed_time):
        if valid_idxs is None:
            valid_idxs is range(0,y_test.shape[0])

        average_method          = 'macro'
        prediction              = y_pred # 

        if self.num_classes == 0:
            scores              = mean_squared_error(y_test,y_pred)
        else:
            scores              = accuracy_score(y_test,y_pred)
        
            if self.num_classes == 2:
                roc_auc             = roc_auc_score(y_test, y_pred_proba[:,1])
                cap_auc             = cap_auc_score(y_test, y_pred_proba[:,1])
            else:
                roc_auc             = roc_auc_score_FIXED(y_test, y_pred_proba,self.num_classes )
                cap_auc             = 0 # cap_auc_score(y_test, y_pred_proba)

            #print(self.y_test[valid_idxs].shape,y_test.shape)
            self.y_test[valid_idxs] = y_test.reshape(self.y_test[valid_idxs].shape)
            self.y_pred[valid_idxs] = y_pred.reshape(-1,1)
            self.y_prob[valid_idxs] = y_pred_proba

            self.measures['accuracy']['values'].append(scores * 100)
            self.measures['precision']['values'].append(precision_score(y_test, prediction, average=average_method)*100)
            self.measures['recall']['values'].append(recall_score(y_test, prediction, average=average_method)*100)
            self.measures['f1']['values'].append(f1_score(y_test, prediction, average=average_method)*100)
            self.measures['roc_auc']['values'].append(roc_auc)
            self.measures['cap_auc']['values'].append(cap_auc)
        self.measures['time']['values'].append(elapsed_time)



    def write_model_info_and_training_info(self,model,parms):
        model_filename  = os.path.join(self.foldername,'model-info.json')
        train_filename  = os.path.join(self.foldername,'train-info.json')
        is_written      = False

        with open(model_filename,'w') as file:
            to_json = getattr(model, "to_json", None)
            if callable(to_json):
                print(json.dumps(json.loads(model.to_json() ),indent=2),file=file)
                is_written = True
            get_params = getattr(model, "get_params", None)
            if callable(get_params):
                print(json.dumps(model.get_params(),indent=2),file=file)
                is_written = True

            if not is_written:
                print('Could not obtain parameters from model.')

        with open(train_filename,'w') as file:
            print(json.dumps(parms,indent=2),file=file)


    def write_model(self,model):
        model_filename = os.path.join(self.foldername,'model.bin')
        if self.model_name[0:3] == 'MLP':
            model.save(model_filename)
        else:
            dump(model,model_filename)
        print("Model written to '" + model_filename + "'.")


    def write_results(self):
        for key,values in self.measures.items():
            values['mean'] = np.mean(values['values'])
            values['std']  = np.std(values['values'])

        with open(os.path.join(self.foldername,'measures.' + self.name + '.json'),'w') as file:
            print(json.dumps(self.measures,indent=2),file=file)


        if not self.history_acc is None:
            self.__write_loss_and_accuracy(self.history_loss,self.history_acc,'loss-and-accuracy.' + self.name)
      
        if self.num_classes > 0:
            self.__write_confusion_matrix(self.y_test,self.y_pred,'confusion-matrix.' + self.name)
        #plot_roc_curve1(self.y_test,self.y_pred,'ROC Curve ' + self.model_name,self.model_name,'roc-curve.' + self.name + '.png')

        np.savez_compressed(os.path.join(self.foldername,'predictions.' + self.name + '.npz'),y_test=self.y_test,y_pred=self.y_pred,y_prob=self.y_prob)


    def __write_loss_and_accuracy(self,history_loss,history_acc,filename):
        plot_history_cv(history_loss,history_acc,filename_out=os.path.join(self.foldername,filename + '.png') )

        with open(os.path.join(self.foldername,filename + '.json'),'w') as file:
            acc   = json.loads(history_acc.to_json(orient='table') ) 
            loss  = json.loads(history_loss.to_json(orient='table') )

            print(json.dumps({ 'accuracy' : acc, 'loss' : loss },indent=2),file=file)



    def __write_confusion_matrix(self,y_test,y_pred,filename):
        c = confusion_matrix(y_test,y_pred,normalize='all')
        d = pd.DataFrame(data=c,columns=self.class_names,index=self.class_names)
        with open(os.path.join(self.foldername,filename + '.json'),'w') as file:
            print(json.dumps(json.loads(d.to_json() ),indent=2),file=file)

        plot_confusion_matrix(y_test,y_pred,self.class_names,filename_out=os.path.join(self.foldername,filename + '.png'))
