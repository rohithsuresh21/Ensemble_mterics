import numpy as np
from pyod.models.copod import COPOD

X_train = np.array([
    [7.5, 42],
    [7.0, 40],
    [8.0, 45],
    [7.8, 41],
    [8.2, 43]
])

X_test = np.array([
    [7.6, 41], 
    [2.0, 95]
])

clf = COPOD(
    contamination=0.01
)
clf.fit(X_train)

# decision_function() returns the raw un-normalized tail-extremeness score
train_scores = clf.decision_function(X_train)
test_scores = clf.decision_function(X_test)

for i, score in enumerate(train_scores):
    print(f"Train sample {i}: score = {score}")

for i, score in enumerate(test_scores):
    print(f"Test sample {i}: score = {score}")

print("Evaluating new days")
print(f"New Day A (Normal Behavior {X_test}): Score = {test_scores[0]:.2f}")
print(f"New Day B (Anomalous Behavior {X_test}): Score = {test_scores[1]:.2f}")