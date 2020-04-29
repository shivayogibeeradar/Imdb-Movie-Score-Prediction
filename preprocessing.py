
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
min_max_scaler = MinMaxScaler()
text_features = ['genre', 'plot_keywords', 'movie_title']
catagory_features = ['content_rating']
number_features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes','cast_total_facebook_likes','budget', 'gross',"facenumber_in_poster"]
all_selected_features = ['content_rating',  'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes','cast_total_facebook_likes','budget', 'gross', 'genres',"facenumber_in_poster","imdb_score"]
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
        data[y] = data[y].fillna(data[y].median()).astype(np.float)
    return data


def column_extract(data):
    selected_data = data[all_selected_features]
    return selected_data

def empty_row_column_val_drop(data):
    data = data.dropna(axis = 0, how = 'any', subset = eliminate_if_empty_list)
    data = data.reset_index(drop = True)
    return data
def preprocessing_numerical_minmax(data):
    global min_max_scaler
    scaled_data = min_max_scaler.fit_transform(data)
    return scaled_data
    
def preprocessing_categorical(data):
    label_encoder = LabelEncoder()
    label_encoded_data = label_encoder.fit_transform(data) 
    label_binarizer = LabelBinarizer()
    label_binarized_data = label_binarizer.fit_transform(label_encoded_data) 
    return label_binarized_data
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
    data_list_numerical = data._get_numeric_data()
    data_numerical = np.array(data_list_numerical)
    data_numerical = preprocessing_numerical_minmax(data_numerical)
    return data_numerical

def preprocessed_agregated_data(database): 
    numerical_data = preprocessing_numerical(database)
    categorical_data = preprocessing_catagory(database)
    all_data = np.append(numerical_data, categorical_data, 1)
    return all_data
