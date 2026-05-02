import prepare_data
from prepare_data import load_data
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df= load_data()
classification_expert = df['initial_classification']
df_features = df.drop(columns=['initial_classification'])

le = LabelEncoder()
classification_expert = le.fit_transform(df['initial_classification'])
#df_features = le.transform(df['initial_classification']) #transforming data to center it

X_train, X_test, Y_train, Y_test = train_test_split(df_features, classification_expert,train_size = 0.68)
#X_train = features for training
#X_test = features for testing
#Y_train = labels for training
#Y_test = labels for testing

prelim_model = XGBClassifier()
prelim_model.fit(X_train, Y_train)
predictions = prelim_model.predict(X_test)
score = accuracy_score(predictions, Y_test)

print(score*100)