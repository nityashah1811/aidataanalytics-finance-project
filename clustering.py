import prepare_data
from prepare_data import load_data
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



df= load_data()

clustering_features = ['housing', 'food', 'entertainment', 'technology','personal_care', 'transportation', 'health_wellness','net_income', 'financial_aid']
    #didnt include total_monthly_spending -- this way the model actually learns off of the patterns

clustering = df[clustering_features]

scaler = StandardScaler()
clustering_scaled=scaler.fit_transform(clustering)

k_values=[2,3,4,5]
scores={} #to store it all later

for k in k_values:
    kmean = KMeans(n_clusters=k, random_state=50)
    label = kmean.fit_predict(clustering_scaled)
    score = silhouette_score(clustering_scaled, label)
    scores[k]=score
    #print("k")
    #print(k)
    #print("score")
    #print(score)

#2 is the highes score in the cluster, so k=2
#but they're all kind of low--if we time in our project let's try finding a btter dataset so that theres not as much overlap in the data
k=2
kmean_final = KMeans(k, random_state=50)
labels_final = kmean_final.fit_predict(clustering_scaled)

df['cluster_labels']=labels_final

#checking cluster differences
#clusters_means = df.groupby('cluster_labels')[clustering_features].mean()
#print(clusters_means)

#2 groups; 0 is better, 1 is worse
centers = kmean_final.cluster_centers_

def predicting_user_cluster(user_data, scaler, model):
    #i need an array of the user_data here of all th values from the clustering_features
    user_scaled = scaler.transform([user_data])
    cluster=model.predict(user_scaled)[0] #which cluster the user belongs to
    return cluster