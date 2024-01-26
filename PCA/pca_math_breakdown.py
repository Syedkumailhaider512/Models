import numpy as np

def pca(X, num_components):
    X_standard = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    print("Transformed Data\n")
    print(X_standard)

    X_cov = np.cov(X_standard, rowvar=False)

    print("Covarience Data\n")
    print(X_cov)

    eigenvalues, eigenvectors = np.linalg.eigh(X_cov)

    print("Eigen Values\n")
    print(eigenvalues)

    print("Eigen Vectors\n")
    print(eigenvectors)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    print("Eigen Values 2\n")
    print(eigenvalues)

    print("Eigen Vectors 2\n")
    print(eigenvectors)

    principal_components = eigenvectors[:, :num_components]

    X_transform = np.dot(X_standard, principal_components)

    print("\nData Transform\n")
    print(X_transform)
