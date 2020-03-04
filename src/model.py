
import sys
import tensorflow as tf
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC


__models = {
        'MLP'                   : ( True,None ),
        #'MLP_MultiTarget'       : ( True,None ),
        'RandomForest'          : ( False,RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0) ),
        'LogisticRegression'    : ( False,LogisticRegression(random_state=0,max_iter=4000,solver='saga') ),
        'RadialSVC'             : ( False, SVC(kernel='rbf',probability=True)),
        'LinearSVC'             : ( False, SVC(kernel='linear',probability=True))
    }

# According to sklearn documentation , the method 'predict_proba' is not defined for 'LinearSVC'
# Workaround:
# LinearSVC_classifier = SklearnClassifier(SVC(kernel='linear',probability=True))
# https://stackoverflow.com/questions/47312432/attributeerrorlinearsvc-object-has-no-attribute-predict-proba



def create_model_mlp(name,input_dim,num_classes):
    """
    """
    print("Num classes",num_classes)
  

    num_hidden  = int(input_dim / 3)
    model       = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_hidden,input_dim=input_dim,kernel_regularizer=tf.keras.regularizers.l2(0.001),activation='relu'))
    model.add(tf.keras.layers.Dropout(0.075))
    model.add(tf.keras.layers.Dense(num_hidden / 2,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.075))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax',kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    if name == 'MLP_MultiTarget':
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def is_batch_training(model_name):
   if model_name not in __models:
        print("Invalid model name '" + model_name + "'.")
        print("Valid model names are: " + ', '.join(__models.keys() ) )
        sys.exit(1)
   else:
        return __models[model_name][0]


def create_model(model_name,num_features,num_classes):

    if model_name is None:
        return __models.keys()
    elif model_name not in __models:
        print("Invalid model name '" + model_name + "'.")
        print("Valid model names are: " + ', '.join(__models.keys() ) )
        sys.exit(1)
    elif model_name[0:3] == 'MLP':
        return create_model_mlp(model_name,num_features,num_classes) 
    else:
        return __models[model_name][1]

