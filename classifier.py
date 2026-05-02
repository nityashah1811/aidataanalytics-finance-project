import prepare_data
from prepare_data import load_data
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df= load_data()
#classification_expert = df['initial_classification']
columns_categorical = ['gender', 'year_in_school', 'major', 'preferred_payment_method']
for col in columns_categorical:
    df[col]=df[col].astype('category') #switching it into a pandas, classified as a category here
#to prevent leakage
col_leak = ['monthly_income', 'tuition', 'housing', 'food', 'transportation', 'books_supplies','entertainment','personal_care','technology','health_wellness','miscellaneous', 'total_monthly_spending', 'net_income']
df_features = df.drop(columns=['initial_classification']+col_leak)

le = LabelEncoder()
classification_expert = le.fit_transform(df['initial_classification'])
#he = OneHotEncoder()
#df_features = he.fit_transform(df['initial_classification']) #transforming data to center it

X_train, X_test, Y_train, Y_test = train_test_split(df_features, classification_expert,train_size = 0.68)
#X_train = features for training
#X_test = features for testing
#Y_train = labels for training
#Y_test = labels for testing

prelim_model = XGBClassifier(enable_categorical=True)
prelim_model.fit(X_train, Y_train)
predictions = prelim_model.predict(X_test)

score = accuracy_score(predictions, Y_test)

print("score")
print(score*100)

#got a score of 100%--testing to see why

"""
print("columns used to predict")
print(df_features.columns)

print("what where the acc values")
print(df['initial_classification'].value_counts())

print("check dataset")
print(Y_train.mean(), Y_test.mean())
"""

"""
print(
)
print(df.shape)

print()
print(df_features.head())
print(df['initial_classification'].head())

print()
print(df_features.dtypes)
"""
