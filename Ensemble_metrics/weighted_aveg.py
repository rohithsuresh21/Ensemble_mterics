import numpy as np
from Normalization import norm_maha, norm_copod, norm_if, norm_knn

def evaluate(norm_maha, norm_copod, norm_if, norm_knn):
    weights = np.array([
        0.35,  # Mahalanobis (Highest trust - Correlation)
        0.25,  # COPOD (High trust - Tail probabilities)
        0.20,  # Isolation Forest (Structural isolation)
        0.20   # KNN (Spatial distance)
    ])

    # Stack the model outputs into a matrix
    # Shape: (4 models, N days)
    model_scores = np.array([norm_maha, norm_copod, norm_if, norm_knn])

    # np.dot multiplies each model's score by its weight and sums them together
    final_score = np.dot(weights, model_scores)
    
    #apply threshold
    if final_score >= 0.60:
        print("Anomaly")

    else:
        print("Normal")


    return final_score 