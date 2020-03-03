
import sys
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split

from utils import make_filename


def __cat_to_num(df,column):
    """
        This function converts the given column to categorical type and 
        renames its elements with numbers from 0 to n - 1 with n begin 
        the number of categories
    """
    a = df[column].astype('category')
    c = ':'.join(list(df[column].unique()))
    return pd.DataFrame(a.cat.codes,columns=[c])


def __cat_to_one_hot(df,column):
    """
        This function converts the strings in a column to a one-hot encoding
        assuming that each integer correspoonds to one cateogy
    """
    # This function encodes a column as one hot encodings
    return pd.get_dummies(df[column])


def __numcat_to_one_hot(df,column):
    """
        This function converts the integers in a column to a one-hot encoding
        assuming that each integer correspoonds to one cateogy
    """
    a         = pd.get_dummies(df[column].apply(str) )
    a.columns = [ column + '_' + str(i) for i in range(a.shape[1])]
    return a


def __int_to_float_01(df,column):
    """
        This function converts all integers in a given column to floats between 0 and 1
    """

    result              = pd.DataFrame(index=df.index,columns=[column])
    result[[column]]    = MinMaxScaler().fit_transform(df[[column]])
    return result





def load_data_csv(file_name,verbose=False):
    df = pd.read_csv(file_name)

    if verbose:
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)
        """
        print(df.columns)
        print(df.head())
        for col in df:
            print(col + '---------')
            print(df[col].value_counts() )
            print(df[col].describe() )
            print('')

    return df



def clean_data(df,verbose = True):
    def conv(x):
        x = x.replace(',','.')
        return float(x)
    del df['ID']
    df.loc[df['Smoker'] == 'No', 'Smoker'] = 'no'
    df["Amount of water's litres drink every day"] = df["Amount of water's litres drink every day"].apply(conv)
    df.loc[df['Number of cigarettes smoked per day'] == 'NU','Number of cigarettes smoked per day'] = 0
    df['Number of cigarettes smoked per day'] = df['Number of cigarettes smoked per day'].astype(int)

    return df


    
def encode_data(df,meta,verbose = True,folder=None,name='data'):
    """
        This function encodes the data in the dataframe in preparation for machine learning
        as specified by the meta information. It returns a dictionary mapping column names to 
        data frames, and a data frame containing all encoded columns
    """
    encoded_columns             = {}
    encoded_columns_cat         = {}

    for column,m in meta.items():
        if verbose:
            print("Converting column '" + column + "' to " + m['encoding'] + " ... " ,end='')
        if m['encoding'] == 'float':
            encoded_column = __int_to_float_01(df,column)
        elif m['encoding'] == '1-hot':
            encoded_column =  __cat_to_one_hot(df,column)
        elif m['encoding'] == 'label':
            encoded_column =  __cat_to_num(df,column)
        else:
            print("Illegal encoding '" + m['encoding'] + "'.")
            sys.exit(1)

        encoded_columns[column] = encoded_column
        if m['encoding'] == 'label' or m['encoding'] == '1-hot':
            encoded_columns_cat[column] = encoded_column

        if verbose:
            print(" Shape = " + str(encoded_column.shape) + ".")

        if not folder is None:
            fn = make_filename(folder,None,name + '-' + column + '.pkl')
            if verbose:
                print("Creating '" + fn + "'.")
            encoded_column.to_pickle(fn)
    df      = pd.concat(encoded_columns.values(),axis=1,sort=False).dropna()
    df_cat  = pd.concat(encoded_columns_cat.values(),axis=1,sort=False).dropna()
    df_cat.columns = encoded_columns_cat.keys()

    if verbose:
        print("Shape of combined data frame = " + str(df.shape) + ".")

    if not folder is None:
        fn = make_filename(folder,None,name + '.pkl')
        if verbose:
            print("Creating '" + fn + "'.")
        df.to_pickle(fn)
        fn = make_filename(folder,None,name + '_cat.pkl')
        if verbose:
            print("Creating '" + fn + "'.")
        df_cat.to_pickle(fn)
    else:
        print("Not saving encoded data, because no folder name is given.")

    return encoded_columns,df,df_cat



        
def describe_data(df,col,file=None,img_filename=None):
    print('',file=file)
    print('#### Descriptive stats for {}'.format(col),file=file)
    print(''*(len(col)+22),file=file)
    if file is None:
        print(df.groupby(col)[col].describe())
    else:
        print(df[col].describe().to_markdown(),file=file)

    print('',file=file)

    if not img_filename is None:
        print("![" + col + "](" + img_filename + ")",file=file)
        print('',file=file)


if __name__ == '__main__':
    pass