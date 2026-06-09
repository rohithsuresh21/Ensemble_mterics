import numpy as np
from scipy.spatial import distance

data = np.array([
    [170, 65],
    [165, 62],
    [180, 75],
    [175, 71],
    [190, 85] 
])

#calculate the mean and covariance matrix
mean = np.mean(data, axis=0)
cov = np.cov(data, rowvar=False)
inverse_cov = np.linalg.inv(cov)

#compute
print("Mahalanobis distances:")
for point in data:
    dist  = distance.mahalanobis(point, mean, inverse_cov)
    print(f"Point: {point}, Distance: {dist:.2f}")