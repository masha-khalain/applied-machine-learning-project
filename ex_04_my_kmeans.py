from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    
    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean", 
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """
    def __init__(self, k: int, max_iter: int = 100, distance_metric: DISTANCE_METRICS = "euclidean", init_method: INIT_METHOD = "kmeans++"):
        self.k: int = k
        self.max_iter: int = max_iter
        self.centroids: np.ndarray | None = None
        self.distance_metric: DISTANCE_METRICS = distance_metric
        self.inertia_: float | None = None
        self.init_method: INIT_METHOD = init_method

    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data.
        
        Args:
            x (np.ndarray | pd.DataFrame): Training data of shape (n_samples, n_features).
        
        Returns:
            MyKMeans: Fitted estimator instance.
        """
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        
        n_samples = x.shape[0]
        self.centroids = self._initialize_centroids(x)
        if self.centroids is None:
            raise RuntimeError("Centroids were not initialized.")

        for _ in tqdm(range(self.max_iter), desc = "Training"):
            distance = self._compute_distance(x, self.centroids)
            labels = np.argmin(distance, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.k):
                pts = x[labels == i]
                if len(pts) == 0:
                    # Keep old centroid to avoid instability
                    new_centroids[i] = self.centroids[i]
                else:
                    new_centroids[i] = pts.mean(axis=0)

            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                break

            self.centroids = new_centroids
        
        self.labels_ = labels
        self.inertia_ = np.sum(np.min(self._compute_distance(x, self.centroids), axis=1) ** 2)
        
        return self
    

    def fit_predict(self, x: np.ndarray):
        """
        Fit the K-means model to the data and return the predicted labels.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError("Input data must be a numpy array or a pandas DataFrame")
        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray):
        """
        Predict the closest cluster for each sample in x.
        
        Args:
            x (np.ndarray): New data to predict, of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """
        # Compute distances between samples and centroids
        distances = self._compute_distance(x, self.centroids)
        
        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the kmeans++ method.
        
        Args:
            x (np.ndarray): Training data.
            
        Returns:
            np.ndarray: Initial centroids.
        """
        n_samples = x.shape[0]
        if self.init_method == "random":
            indices = np.random.choice(n_samples, self.k, replace=False)
            centroids = x[indices]
        elif self.init_method == "kmeans++":
            centroids = [x[np.random.randint(n_samples)]]
            for _ in range(1, self.k):
                distances = np.min(self._compute_distance(x, np.array(centroids)), axis=1)
                probs = distances ** 2 / np.sum(distances ** 2)
                next_centroid = x[np.random.choice(n_samples, p=probs)]
                centroids.append(next_centroid)
            centroids = np.array(centroids)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
        
        # Final safety check:
        if centroids is None or len(centroids) != self.k:
            raise RuntimeError("Failed to initialize centroids properly.")
        
        return centroids

    def _compute_distance(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute the distance between samples and centroids.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, n_features) or (n_samples, time_steps, n_features).
            centroids (np.ndarray): Centroids of shape (k, n_features) or (k, time_steps, n_features).
            
        Returns:
            np.ndarray: Distances between each sample and each centroid, shape (n_samples, k).
        """
        if centroids is None:
            raise ValueError("Centroids must not be None")

        n_samples = x.shape[0]
        k = centroids.shape[0]
        distances = np.zeros((n_samples, k))

        for i in range(k):
            c = centroids[i]

            # Determine data dimensionality
            if x.ndim == 2:
                # 2D data: (n_samples, n_features)
                # Broadcast centroid: c shape (n_features,) -> (1, n_features)
                if c.ndim == 1:
                    c = c.reshape(1, -1)
                diff = x - c  

                if self.distance_metric == "euclidean":
                    distances[:, i] = np.linalg.norm(diff, axis=1)
                elif self.distance_metric == "manhattan":
                    distances[:, i] = np.sum(np.abs(diff), axis=1)
                elif self.distance_metric == "dtw":
                    # DTW on vectors — might not make sense for 2D, fallback to euclidean
                    distances[:, i] = np.linalg.norm(diff, axis=1)
                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

            elif x.ndim == 3:
                # 3D data: (n_samples, time_steps, n_features)
                # centroid shape: (time_steps, n_features)
                # Computing distance sample-wise between x[sample] and centroid c

                if self.distance_metric == "euclidean":
                    # Compute mean Euclidean distance across time_steps (feature-wise)
                    # For each sample: sqrt(sum over time and features of squared diff)
                    dist_per_sample = np.linalg.norm(x - c, axis=(1, 2))  # norm over last 2 axes
                    distances[:, i] = dist_per_sample

                elif self.distance_metric == "manhattan":
                    dist_per_sample = np.sum(np.abs(x - c), axis=(1, 2))
                    distances[:, i] = dist_per_sample

                elif self.distance_metric == "dtw":
                    # For each sample, compute DTW distance to centroid c
                    # Here you need to handle multivariate DTW or sum over features
                    dist_per_sample = np.zeros(n_samples)
                    for s in range(n_samples):
                        # You may want to sum DTW distances over features or average
                        # Simple approach: sum DTW over each feature dimension
                        total_dtw = 0.0
                        for f in range(x.shape[2]):
                            seq1 = x[s, :, f]
                            seq2 = c[:, f]
                            total_dtw += dtw.distance_fast(seq1, seq2)
                        dist_per_sample[s] = total_dtw
                    distances[:, i] = dist_per_sample

                else:
                    raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

            else:
                raise ValueError("Input data must be a 2D or 3D array")

        return distances


    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Simplified DTW distance computation using dtaidistance.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, time_steps, n_features) or (n_samples, n_features)
            centroids (np.ndarray): Centroids of shape (k, time_steps, n_features) or (k, n_features)
            
        Returns:
            np.ndarray: DTW distances between each sample and each centroid, shape (n_samples, k).
        """
        distances = []
        for i in range(x.shape[0]):
            if x.ndim == 3:
                # shape (n_samples, time_steps, features)
                seq1 = x[i].reshape(-1)
                seq2 = self.centroids.reshape(-1)
            else:
                seq1 = x[i]
                seq2 = self.centroids
            distances.append(dtw.distance_fast(seq1, seq2))
        return np.array(distances)
