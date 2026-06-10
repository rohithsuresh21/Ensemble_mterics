import pandas as pd
df_master = pd.read_csv("synchronized_daily_feed.csv")
df_baseline = df_master.iloc[0:15]
X_train_4_features = df_baseline[['sleep_duration', 'screen_on_time', 'systolic_bp', 'text_sentiment']].values
X_test_4_features = df_master.iloc[15:][['sleep_duration', 'screen_on_time', 'systolic_bp', 'text_sentiment']].values

# --------------------------------------
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
print("Norm of Isolation Forest")
