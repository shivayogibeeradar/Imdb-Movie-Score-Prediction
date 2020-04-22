import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
#from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
min_max_scaler = preprocessing.MinMaxScaler()
path="https://raw.githubusercontent.com/sundeepblue/movie_rating_prediction/master/movie_metadata.csv"

text_features = ['genre', 'plot_keywords', 'movie_title']
catagory_features = ['country', 'content_rating', 'language']
number_features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes','cast_total_facebook_likes','budget', 'gross',"facenumber_in_poster"]
all_selected_features = ['country', 'content_rating', 'language', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes','cast_total_facebook_likes','budget', 'gross', 'genres',"facenumber_in_poster","imdb_score"]
eliminate_if_empty_list = [ 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'director_facebook_likes','cast_total_facebook_likes', 'gross']


def data_import(path):
    data = pd.read_csv(path)
    return data
#preprocessing
def data_clean(path):
    read_data = pd.read_csv(path)
    select_data = read_data[all_selected_features]
    
    data = select_data.dropna(axis = 0, how = 'any', subset = eliminate_if_empty_list)
    #data = select_data.dropna(axis = 0, how = 'any')
    data = data.reset_index(drop = True)
    for x in catagory_features:
        data[x] = data[x].fillna('None').astype('category')
    for y in number_features:
        data[y] = data[y].fillna(0).astype(np.float)
    return data


def column_extract(data):
    selected_data = data[all_selected_features]
    return selected_data

def empty_row_column_val_drop(data):
    data = data.dropna(axis = 0, how = 'any', subset = eliminate_if_empty_list)
    data = data.reset_index(drop = True)
    return data

def data_fillna(data): 
    for x in catagory_features:
        data[x] = data[x].fillna('None').astype('category')
    for y in number_features:
        data[y] = data[y].fillna(0.0).astype(np.float)
    return data

def append_data(data1, data2):
    result_data = np.append(data1, data2, 1)
    return result_data
    
def preprocessing_numerical_minmax(data):
   
    scaled_data = min_max_scaler.fit_transform(data)
    return scaled_data
    
def preprocessing_categorical(data):
    label_encoder = LabelEncoder()
    label_encoded_data = label_encoder.fit_transform(data) 
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarized_data = label_binarizer.fit_transform(label_encoded_data) 
    return label_binarized_data

def preprocessing_text(data):  
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized_text = tfidf_vectorizer.fit_transform(data)  
    return tfidf_vectorized_text
def preprocessing_catagory(data):
    data_c=0
    for i in range(len(catagory_features)):
        new_data = data[catagory_features[i]]
        new_data_c = preprocessing_categorical(new_data)
        if i == 0:
            data_c=new_data_c
        else:
            data_c = np.append(data_c, new_data_c, 1)
    return data_c

def preprocessing_numerical(data):
    data_list_numerical = list(zip(data['director_facebook_likes'], data['actor_1_facebook_likes'], 
                                   data['actor_2_facebook_likes'], data['actor_3_facebook_likes'], 
                                   data['cast_total_facebook_likes'], data['budget']))

    data_numerical = np.array(data_list_numerical)
    data_numerical = preprocessing_numerical_minmax(data_numerical)
    return data_numerical

def preprocessed_agregated_data(database): 
    numerical_data = preprocessing_numerical(database)
    categorical_data = preprocessing_catagory(database)
    all_data = np.append(numerical_data, categorical_data, 1)
    return all_data