from sklearn.datasets import Load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_splits
from sklearn.metrics import accuracy_score

iris = Load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_splits(X, y, test_size=0.2, random_state=42)

#scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn= KNeighborsClassifier(n_neighbors=3,
                        metric='Manhattan'
                          )

y_pred = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

