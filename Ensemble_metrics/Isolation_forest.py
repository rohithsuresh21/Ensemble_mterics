import numpy as np
from pyod.models.iforest import IForest

X_train = np.array([
    [7.5, 42],
    [7.0, 40],
    [8.0, 45],
    [7.8, 41],
    [8.2, 43]
])

X_test = np.array([
    [7.6, 41],  # Normal day
    [2.0, 95]   # Highly anomalous day
])

clf = IForest(
    contamination = 0.2,
    random_state = 42
)
clf.fit(X_train)

test_labels = clf.predict(X_test)
test_scores = clf.decision_function(X_test) # raw

for i, (label, score) in enumerate(zip(test_labels, test_scores)):
    status = "Anomalous" if label == 1 else "Normal"
    print(f"Test Point {i}: {status} (Anomaly Score: {score:.2f})")
