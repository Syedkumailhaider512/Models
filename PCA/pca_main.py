from pca_math_breakdown import pca
import numpy as np


X = np.array([[54,24,37],[14,59,46],[37,58,19]])
components = 1

pca = pca(X, components)
