import pandas as pd
df_master = pd.read_csv("synchronized_daily_feed.csv")
df_baseline = df_master.iloc[0:15] # 0 to 15 rows for training
X_train_4_features = df_baseline[['sleep_duration', 'screen_on_time', 'systolic_bp', 'text_sentiment']].values
X_test_4_features = df_master.iloc[15:][['sleep_duration', 'screen_on_time', 'systolic_bp', 'text_sentiment']].values   # 15 to end rows for testing

# --------------------------------------Input norm-------------


print("Norm of KNN")
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

#scaler max-min your data
scaler = MinMaxScaler()
X_train_sclaled = scaler.fit_transform(X_train_4_features)
X_test_sclaled = scaler.transform(X_test_4_features)

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sclaled)

distances, = knn.kneighbors(X_test_sclaled)
knn_anomaly_score = np.mean(distances, axis=1)

#----------------------------------------
print("Norm of Mahalanobis")
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

# Normalize the data using StandardScaler
sclaler = StandardScaler()
X_train_scaled = sclaler.fit_transform(X_train_4_features)
X_test_scaled = sclaler.transform(X_test_4_features)

covariance_matrix = np.cov(X_train_scaled, rowvar=False)
inv_covariance_matrix = np.linalg.inv(covariance_matrix)
mean_baseline = np.mean(X_train_scaled, axis=0)

mahalanobis_scores = []
for row in X_test_scaled:
    delta = row - mean_baseline   # use delta to measure how far they are away from center while mean give center point
    distance = np.sqrt(np.dot(np.dot(delta, inv_covariance_matrix), delta.T))
    mahalanobis_scores.append(distance)

#-------------------------------------------
#Isolation Forest
#no normalisation as IF handles standardized data beautifully
from pyod import IsolationForest 
from pyod import COPULA as copod

iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train_scaled)
# score_samples returns the negative anomaly score by default in scikit-learn so we invert
if_anomaly_scores = -iso_forest.score_samples(X_test_scaled)


#----------------------------------------output norm---------------------
raw_knn    = knn.kneighbors(X_test_4_features).mean(axis=1)       # e.g., values from 0 to 3.5
raw_maha   = np.array([mahalanobis_scores(row) for row in X_test_4_features])   # e.g., values from 0 to 25.0
raw_if     = -IsolationForest.score_samples(X_test_4_features)                      # e.g., values from -0.2 to 0.4
raw_copod  = copod.decision_function(X_test_4_features)                # e.g., values from 0 to 35.0

#initialize
output_scaler = MinMaxScaler(feature_range=(0, 1))

norm_knn   = output_scaler.fit_transform(raw_knn.reshape(-1, 1)).flatten()
norm_maha  = output_scaler.fit_transform(raw_maha.reshape(-1, 1)).flatten()
norm_if    = output_scaler.fit_transform(raw_if.reshape(-1, 1)).flatten()
norm_copod = output_scaler.fit_transform(raw_copod.reshape(-1, 1)).flatten()

master_anomaly_line = (norm_knn + norm_maha + norm_if + norm_copod) / 4.0