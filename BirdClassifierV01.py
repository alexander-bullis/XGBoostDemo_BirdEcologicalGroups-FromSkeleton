#%% imports
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%%set up project parameters
project_name = 'Bird_Classifier_V01'
project_dir = '/DataScience/Projects/' + project_name
target_data_webdirectory = 'zhangjuefei/birds-bones-and-living-habits'
target_data_filename = 'bird.csv'

#create wroking directory if does not exist
if not os.path.exists(project_dir):
    os.makedirs(project_dir)
os.chdir(project_dir)

#%% download dataset from web service to project folder (websource = Kaggle)
kaggle = KaggleApi()
kaggle.authenticate()
kaggle.dataset_download_file(target_data_webdirectory,
                          file_name=target_data_filename,
                          path='./')
#Create dataframe from csv data
df = pd.read_csv(target_data_filename)
print(df.dtypes)
#%% data cleanup

#remove Nan/NULL rows altogether
df.dropna()

#divide dataset into features and categories
X = df.iloc[:,1:10]
Y_txt = df.iloc[:,11]

#encode categories as numeric values
Y = pd.Categorical(Y_txt, categories=Y_txt.unique()).codes

#%% build xgboost model

#split dataset to training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=5)

#fit model no training data
xgbmodel = XGBClassifier(booster='gbtree',silent = 1)
xgbmodel.fit(X_train, y_train)

#make predictions for test data, rounding to nearest whole integer
y_classifier = xgbmodel.predict(X_test)
output = [round(res_raw) for res_raw in y_classifier]

#%% evaluate predictions
res_accuracy = accuracy_score(y_test, output)
print('accuracy pct: '+str(res_accuracy * 100))

