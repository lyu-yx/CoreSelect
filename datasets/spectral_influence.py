import numpy as np
import scipy
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
from typing import Tuple, List, Optional, Dict


class SpectralInfluenceSelector:
    """
    A coreset selection method that combines spectral clustering with influence-based sampling.
    
    This approach:
    1. Uses spectral clustering to identify natural substructures in the data
    2. Computes per-sample influence scores within each cluster
    3. Selects the most representative samples from each cluster based on influence
    
    Theoretical foundations:
    - Spectral clustering: Based on spectral graph theory (Ng, Jordan, Weiss, 2002)
    - Influence functions: From robust statistics (Koh & Liang, 2017)
    - Representative sampling: Connected to k-center and facility location problems
    """
    
    def __init__(
        self,
        n_clusters_factor: float = 0.1,
        influence_type: str = "gradient_norm",
        balance_clusters: bool = True,
        affinity_metric: str = "rbf",
        random_state: int = 42
    ):
        """
        Initialize the SpectralInfluenceSelector.
        
        Args:
            n_clusters_factor: Factor to determine number of clusters (as a fraction of subset size)
            influence_type: Method to compute influence scores ('gradient_norm', 'loss', 'uncertainty')
            balance_clusters: Whether to enforce balanced selection across clusters
            affinity_metric: Metric for computing the affinity matrix ('rbf', 'cosine', 'nearest_neighbors')
            random_state: Random seed for reproducibility
        """
        self.n_clusters_factor = n_clusters_factor
        self.influence_type = influence_type
        self.balance_clusters = balance_clusters
        self.affinity_metric = affinity_metric
        self.random_state = random_state
        
    def generate_subset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_outputs: np.ndarray,
        subset_size: int,
        true_gradients: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a diverse and representative subset.
        
        Args:
            features: Feature representations of samples (e.g., embeddings or gradients)
            labels: Class labels for the samples
            model_outputs: Model's predictions (e.g., softmax outputs)
            subset_size: Size of the subset to select
            true_gradients: Pre-computed gradients if available (optional)
            
        Returns:
            Tuple of (selected indices, selection weights)
        """
        if subset_size <= 0 or len(features) <= subset_size:
            # If we need all samples or more, return all with equal weights
            return np.arange(len(features)), np.ones(len(features)) / len(features)
            
        # 1. Determine the number of clusters based on subset size
        n_clusters = max(2, int(subset_size * self.n_clusters_factor))
        
        # 2. Perform spectral clustering
        clusters, cluster_centers = self._perform_spectral_clustering(features, n_clusters)
        
        # 3. Compute influence scores for each sample
        influence_scores = self._compute_influence_scores(
            features, labels, model_outputs, true_gradients
        )
        
        # 4. Select representative samples from each cluster
        selected_indices, weights = self._select_representatives(
            clusters, influence_scores, subset_size, features, labels
        )
        
        return selected_indices, weights
        
    def _perform_spectral_clustering(
        self, features: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform spectral clustering on the features.
        
        Args:
            features: Feature representations of samples
            n_clusters: Number of clusters to form
            
        Returns:
            Tuple of (cluster assignments, cluster centers)
        """
        # Normalize features for better clustering
        normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Create affinity matrix based on specified metric
        if self.affinity_metric == 'rbf':
            # RBF kernel with adaptive bandwidth
            median_dist = np.median(scipy.spatial.distance.pdist(normalized_features))
            gamma = 1.0 / (2 * median_dist**2) if median_dist > 0 else 1.0
            affinity_matrix = rbf_kernel(normalized_features, gamma=gamma)
        elif self.affinity_metric == 'cosine':
            # Cosine similarity
            affinity_matrix = np.dot(normalized_features, normalized_features.T)
            # Ensure values are in [-1, 1] range then shift to [0, 1]
            affinity_matrix = (affinity_matrix + 1) / 2
        else:
            # Default to RBF
            affinity_matrix = rbf_kernel(normalized_features)
        
        try:
            # Perform spectral clustering
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=self.random_state,
                n_init=10  # Multiple initializations for better stability
            ).fit(affinity_matrix)
            
            # Get cluster assignments
            cluster_assignments = clustering.labels_
            
        except Exception as e:
            # Fallback to simpler clustering if spectral clustering fails
            print(f"Spectral clustering failed with error: {str(e)}. Falling back to K-means.")
            from sklearn.cluster import KMeans
            
            # Use K-means as fallback
            clustering = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            ).fit(normalized_features)
            
            cluster_assignments = clustering.labels_
        
        # Compute cluster centers
        centers = np.zeros((n_clusters, features.shape[1]))
        for i in range(n_clusters):
            cluster_members = np.where(cluster_assignments == i)[0]
            if len(cluster_members) > 0:
                centers[i] = np.mean(features[cluster_members], axis=0)
        
        return cluster_assignments, centers
        
    def _compute_influence_scores(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        model_outputs: np.ndarray,
        true_gradients: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute influence scores for each sample.
        
        Args:
            features: Feature representations of samples
            labels: Class labels for the samples
            model_outputs: Model's predictions (softmax outputs)
            true_gradients: Pre-computed gradients if available
            
        Returns:
            Array of influence scores for each sample
        """
        if self.influence_type == 'gradient_norm':
            # If true gradients are provided, use them
            if true_gradients is not None:
                # L2 norm of gradients as influence
                return np.linalg.norm(true_gradients, axis=1)
            
            # Otherwise, approximate gradients using model outputs
            # For cross-entropy loss, gradient ≈ (softmax - one_hot)
            one_hot = np.eye(model_outputs.shape[1])[labels]
            approx_gradients = model_outputs - one_hot
            return np.linalg.norm(approx_gradients, axis=1)
            
        elif self.influence_type == 'loss':
            # Cross-entropy loss as influence
            one_hot = np.eye(model_outputs.shape[1])[labels]
            cross_entropy = -np.sum(one_hot * np.log(model_outputs + 1e-10), axis=1)
            return cross_entropy
            
        elif self.influence_type == 'uncertainty':
            # Entropy of predictions as influence
            entropy = -np.sum(model_outputs * np.log(model_outputs + 1e-10), axis=1)
            return entropy
            
        else:
            # Default to gradient norm approximation
            one_hot = np.eye(model_outputs.shape[1])[labels]
            approx_gradients = model_outputs - one_hot
            return np.linalg.norm(approx_gradients, axis=1)
    
    def _select_representatives(
        self,
        clusters: np.ndarray,
        influence_scores: np.ndarray,
        subset_size: int,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select representative samples from each cluster based on influence scores.
        
        Args:
            clusters: Cluster assignments for each sample
            influence_scores: Influence score for each sample
            subset_size: Size of the subset to select
            features: Feature representations of samples
            labels: Class labels for the samples
            
        Returns:
            Tuple of (selected indices, weights)
        """
        # Identify unique clusters
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)
        
        # Handle class balance - get unique classes
        unique_classes = np.unique(labels)
        class_counts = {c: np.sum(labels == c) for c in unique_classes}
        
        # Allocate budget proportionally to classes and clusters
        if self.balance_clusters:
            # Balanced allocation across clusters
            samples_per_cluster = np.full(n_clusters, subset_size // n_clusters)
            # Distribute remaining samples
            remainder = subset_size - np.sum(samples_per_cluster)
            if remainder > 0:
                cluster_sizes = np.array([np.sum(clusters == c) for c in unique_clusters])
                largest_clusters = np.argsort(-cluster_sizes)[:remainder]
                samples_per_cluster[largest_clusters] += 1
        else:
            # Proportional allocation based on cluster size
            cluster_sizes = np.array([np.sum(clusters == c) for c in unique_clusters])
            samples_per_cluster = np.maximum(1, np.round(subset_size * cluster_sizes / np.sum(cluster_sizes)).astype(int))
            
            # Adjust to match exactly subset_size
            while np.sum(samples_per_cluster) > subset_size:
                largest_idx = np.argmax(samples_per_cluster)
                if samples_per_cluster[largest_idx] > 1:
                    samples_per_cluster[largest_idx] -= 1
            while np.sum(samples_per_cluster) < subset_size:
                smallest_idx = np.argmin(samples_per_cluster)
                samples_per_cluster[smallest_idx] += 1
        
        # Select samples from each cluster
        all_selected = []
        
        for i, cluster_id in enumerate(unique_clusters):
            # Get samples in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Skip empty clusters
            if cluster_size == 0:
                continue
                
            # Determine how many samples to select from this cluster
            n_to_select = min(samples_per_cluster[i], cluster_size)
            
            # Get influence scores for this cluster
            cluster_influences = influence_scores[cluster_indices]
            
            # Rank by influence (higher is more influential)
            ranked_indices = np.argsort(-cluster_influences)
            selected_from_cluster = cluster_indices[ranked_indices[:n_to_select]]
            
            all_selected.append(selected_from_cluster)
        
        # Combine all selected indices
        selected_indices = np.concatenate(all_selected)
        
        # If we didn't get enough samples, add more from the highest influence scores
        if len(selected_indices) < subset_size:
            remaining = subset_size - len(selected_indices)
            unselected = np.setdiff1d(np.arange(len(features)), selected_indices)
            
            if len(unselected) > 0:
                unselected_influences = influence_scores[unselected]
                top_remaining = unselected[np.argsort(-unselected_influences)[:remaining]]
                selected_indices = np.concatenate([selected_indices, top_remaining])
        
        # Compute weights based on influence scores (higher influence = higher weight)
        selected_influences = influence_scores[selected_indices]
        weights = selected_influences / np.sum(selected_influences)
        
        return selected_indices, weights