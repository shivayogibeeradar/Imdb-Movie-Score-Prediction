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
        data[y] = data[y].fillna(data[y].median()).astype(np.float)
    return data


def column_extract(data):
    selected_data = data[all_selected_features]
    return selected_data

def empty_row_column_val_drop(data):
    data = data.dropna(axis = 0, how = 'any', subset = eliminate_if_empty_list)
    data = data.reset_index(drop = True)
    return data

