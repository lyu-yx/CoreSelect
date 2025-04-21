import numpy as np
from typing import Tuple, List, Optional
import torch
from enum import Enum

import numpy as np
import torch
from utils import submodular, craig
import time  # Changed from: from time import time

class SubsetGenerator:
    def __init__(self, greedy: bool = True, smtk: float = 1.0):
        """
        Initialize the SubsetGenerator
        
        Args:
            greedy: Whether to use greedy selection (True) or random (False)
            smtk: Smoothing parameter for softmax temperature
        """
        self.greedy = greedy
        self.smtk = smtk
        
        # Parameters for the spectral influence selector
        self.n_clusters_factor = 0.1
        self.influence_type = "gradient_norm"
        self.balance_clusters = True
        self.affinity_metric = "rbf"
        
        # Parameters for trimodal selection
        self.spectral_weight = 0.33  # Weight for spectral component
        self.dpp_submod_ratio = 0.5  # Equal weighting between DPP and submodular by default
        
    def generate_mixed_subset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        softmax_preds: np.ndarray,
        subset_size: int,
        dpp_weight: float = 0.5,
        submod_weight: float = 0.5,
        selection_method: str = "mixed"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a subset using a combination of methods for better coverage and diversity.
        
        Following the theory:
        F(S) = f(S) + λD(S)
        where f(S) is a coverage function and D(S) is a diversity function.
        
        Args:
            features: Model embeddings/outputs for each sample
            labels: True class labels
            softmax_preds: Prediction probabilities from the model
            subset_size: Size of the subset to select
            dpp_weight: Weight for DPP selection component (λ in the theory)
            submod_weight: Weight for submodular selection component
            selection_method: Method for selection ("mixed", "dpp", "submod", "spectral", "trimodal", "rand")
            
        Returns:
            Tuple of (selected indices, weights for the selected samples)
        """
        if selection_method == "rand":
            # Random selection
            indices = np.random.choice(len(features), subset_size, replace=False)
            weights = np.ones(subset_size) / subset_size  # Equal weights
            return indices, weights
            
        elif selection_method == "dpp":
            # Pure DPP selection for diversity
            indices, weights = self._dpp_selection(features, labels, subset_size)
            return indices, weights
            
        elif selection_method == "submod":
            # Pure submodular selection for coverage
            indices, weights = self._submodular_selection(features, labels, softmax_preds, subset_size)
            return indices, weights
            
        elif selection_method == "spectral":
            # Spectral clustering with influence functions (non-DPP approach)
            indices, weights = self.generate_spectral_influence_subset(
                features=features,
                labels=labels,
                softmax_preds=softmax_preds,
                subset_size=subset_size
            )
            return indices, weights
            
        elif selection_method == "trimodal":
            # Use the new trimodal mixed selection method
            from datasets.trimodal_mixed import trimodal_mixed_selection
            
            # Use approximately 1/3 of budget for each method
            spectral_weight = getattr(self, 'spectral_weight', 0.33)
            
            # If we have a DPP weight, distribute remaining weight between DPP and submod
            # based on the specified dpp_weight ratio
            remaining_weight = 1.0 - spectral_weight
            actual_dpp_weight = remaining_weight * dpp_weight / (dpp_weight + submod_weight)
            actual_submod_weight = remaining_weight * submod_weight / (dpp_weight + submod_weight)
            
            # Timing for performance evaluation
            time_start = time.time()
            
            # Call the trimodal selection method
            indices, weights = trimodal_mixed_selection(
                features=features,
                labels=labels,
                softmax_preds=softmax_preds,
                subset_size=subset_size,
                dpp_weight=actual_dpp_weight,
                submod_weight=actual_submod_weight,
                spectral_weight=spectral_weight,
                n_clusters_factor=getattr(self, 'n_clusters_factor', 0.1),
                influence_type=getattr(self, 'influence_type', 'gradient_norm'),
                balance_clusters=getattr(self, 'balance_clusters', True),
                affinity_metric=getattr(self, 'affinity_metric', 'rbf'),
                random_state=42
            )
            
            selection_time = time.time() - time_start
            print(f"Trimodal mixed selection completed in {selection_time:.2f} seconds")
            
            return indices, weights
            
        else:  # "mixed" or default - this is the joint objective approach
            # If not greedy, just return random samples with equal weights
            if not self.greedy:
                indices = np.random.choice(len(features), subset_size, replace=False)
                weights = np.ones(subset_size) / subset_size
                return indices, weights
            
            # Start timer for performance tracking
            start_time = time.time()  # Changed from: start_time = time()
            
            try:
                # Try to import utility functions
                from utils.submodular import get_orders_and_weights_hybrid
                
                # Normalize features for kernel computation (important for distance metrics)
                X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
                
                # Use the optimized hybrid implementation that combines coverage and diversity
                # with class-based parallel processing
                indices, weights, _, _, selection_time, _ = get_orders_and_weights_hybrid(
                    B=subset_size,
                    X=X,
                    metric="cosine",  # Cosine similarity works well for normalized embeddings
                    y=labels,
                    dpp_weight=dpp_weight,  # Use the provided diversity weight
                    mode="sparse" if len(features) > 10000 else "dense",  # Adaptive mode selection
                    num_n=128  # Number of neighbors for sparse computation
                )
                
                # Log performance
                print(f"Hybrid mixed selection completed in {selection_time:.2f} seconds")
                
                return indices, weights
                
            except ImportError:
                print("Could not import get_orders_and_weights_hybrid. Using class-parallel implementation.")
                # Fall back to the class-parallel implementation we added previously
                
                # Start timer for performance tracking
                start_time = time.time()  # Changed from: start_time = time()
                
                # Implement class-based parallel processing for mixed selection
                # This approach follows the pattern in get_orders_and_weights
                
                # 1. Split data by class
                classes = np.unique(labels)
                C = len(classes)  # number of classes
                
                # 2. Determine the number of samples to select per class
                class_counts = [np.sum(labels == c) for c in classes]
                class_fractions = np.array(class_counts) / len(labels)
                num_per_class = np.int32(np.ceil(class_fractions * subset_size))
                
                # Ensure we don't select more than subset_size elements
                while np.sum(num_per_class) > subset_size:
                    # Find the class with the largest allocation and reduce by 1
                    idx_to_reduce = np.argmax(num_per_class)
                    num_per_class[idx_to_reduce] -= 1
                
                print(f"Mixed selection: selecting {num_per_class} elements per class")
                
                # 3. Define function to process each class in parallel
                def process_class(class_data):
                    c_idx, c_num = class_data
                    class_indices = np.where(labels == classes[c_idx])[0]
                    if len(class_indices) == 0 or c_num == 0:
                        return [], []
                    
                    # Extract class-specific data
                    class_features = features[class_indices]
                    
                    # Normalize features for kernel computation
                    X = class_features / (np.linalg.norm(class_features, axis=1, keepdims=True) + 1e-8)
                    similarity_matrix = np.dot(X, X.T)
                    
                    try:
                        # Try to import submodlib functions
                        from submodlib import FacilityLocationFunction, LogDeterminantFunction
                        
                        # Define objectives per class
                        fl_obj = FacilityLocationFunction(
                            n=len(class_features),
                            mode="dense",
                            sijs=similarity_matrix,
                            separate_rep=False
                        )
                        
                        dpp_obj = LogDeterminantFunction(
                            n=len(class_features),
                            mode="dense",
                            sijs=similarity_matrix,
                            lambdaVal=1.0
                        )
                        
                        # Initialize selection for this class
                        class_selected = []
                        class_remaining = list(range(len(class_features)))
                        
                        # Select elements for this class
                        for _ in range(min(c_num, len(class_features))):
                            if not class_remaining:
                                break
                            
                            if len(class_selected) == 0:
                                # First element: compute objectives directly
                                fl_gains = np.sum(similarity_matrix[class_remaining, :], axis=1)
                                dpp_gains = np.log(np.diag(similarity_matrix)[class_remaining] + 1e-10)
                                
                                # Combine gains with weights
                                combined_gains = (1 - dpp_weight) * fl_gains + dpp_weight * dpp_gains
                                
                                # Find the best point
                                best_local_idx = np.argmax(combined_gains)
                                best_idx = class_remaining[best_local_idx]
                                
                            else:
                                # Subsequent elements: compute marginal gains
                                current_set = set(class_selected)
                                fl_prev = fl_obj.evaluate(current_set)
                                dpp_prev = dpp_obj.evaluate(current_set)
                                
                                best_gain = -float('inf')
                                best_idx = -1
                                
                                # Process in batches
                                batch_size = min(1000, len(class_remaining))
                                for i in range(0, len(class_remaining), batch_size):
                                    batch_indices = class_remaining[i:i+batch_size]
                                    
                                    fl_gains = []
                                    dpp_gains = []
                                    
                                    for idx in batch_indices:
                                        temp_set = current_set.copy()
                                        temp_set.add(idx)
                                        
                                        fl_curr = fl_obj.evaluate(temp_set)
                                        fl_gain = fl_curr - fl_prev
                                        
                                        dpp_curr = dpp_obj.evaluate(temp_set)
                                        dpp_gain = dpp_curr - dpp_prev
                                        
                                        fl_gains.append(fl_gain)
                                        dpp_gains.append(dpp_gain)
                                    
                                    # Convert to numpy arrays
                                    fl_gains = np.array(fl_gains)
                                    dpp_gains = np.array(dpp_gains)
                                    
                                    # Combine gains with weights
                                    combined_gains = (1 - dpp_weight) * fl_gains + dpp_weight * dpp_gains
                                    
                                    # Find the best point in this batch
                                    local_best_idx = np.argmax(combined_gains)
                                    local_best_gain = combined_gains[local_best_idx]
                                    
                                    if local_best_gain > best_gain:
                                        best_gain = local_best_gain
                                        best_idx = batch_indices[local_best_idx]
                            
                            # Add the best element
                            if best_idx != -1:
                                class_selected.append(best_idx)
                                class_remaining.remove(best_idx)
                        
                        # Convert local indices to global
                        global_indices = [class_indices[idx] for idx in class_selected]
                        
                        # Calculate weights for this class
                        if len(class_selected) > 0:
                            row_sums = np.sum(similarity_matrix[class_selected, :], axis=1)
                            class_weights = row_sums / np.sum(row_sums)
                        else:
                            class_weights = []
                            
                        return global_indices, class_weights
                    
                    except ImportError:
                        # Fall back to numpy implementation if submodlib not available
                        print("SubModLib not found. Using numpy implementation for class processing.")
                        
                        # Initialize selection
                        selected = []
                        remaining = list(range(len(class_features)))
                        
                        # Select elements greedily
                        for _ in range(min(c_num, len(class_features))):
                            if not remaining:
                                break
                                
                            if not selected:
                                # For first element, select point with best combined score
                                coverage_scores = np.sum(similarity_matrix[remaining, :], axis=1)
                                diversity_scores = np.log(np.diag(similarity_matrix)[remaining] + 1e-10)
                                combined_scores = (1 - dpp_weight) * coverage_scores + dpp_weight * diversity_scores
                                best_idx = remaining[np.argmax(combined_scores)]
                                selected.append(best_idx)
                                remaining.remove(best_idx)
                            else:
                                # For subsequent elements
                                best_idx = -1
                                best_score = -float('inf')
                                
                                for idx in remaining:
                                    # Coverage component: additional similarity to uncovered points
                                    coverage_score = np.sum(similarity_matrix[idx, :]) / len(class_features)
                                    
                                    # Diversity component: dissimilarity to already selected
                                    sim_to_selected = np.mean(similarity_matrix[idx, selected])
                                    diversity_score = 1 - sim_to_selected
                                    
                                    # Combined score
                                    score = (1 - dpp_weight) * coverage_score + dpp_weight * diversity_score
                                    
                                    if score > best_score:
                                        best_score = score
                                        best_idx = idx
                                        
                                selected.append(best_idx)
                                remaining.remove(best_idx)
                        
                        # Convert local indices to global
                        global_indices = [class_indices[idx] for idx in selected]
                        
                        # Calculate weights
                        if len(selected) > 0:
                            row_sums = np.sum(similarity_matrix[selected, :], axis=1)
                            class_weights = row_sums / np.sum(row_sums)
                        else:
                            class_weights = []
                            
                        return global_indices, class_weights
                
                # 4. Process each class in parallel with efficient result collection
                class_data = [(c_idx, num_per_class[c_idx]) for c_idx in range(len(classes))]
                processed_results = list(map(process_class, class_data))
                
                # 5. Combine results efficiently (replacing inefficient np.append loops)
                # Pre-calculate total size for pre-allocation
                total_indices = sum(len(indices) for indices, _ in processed_results)
                
                # Use list comprehensions for collecting data
                all_indices = [idx for result in processed_results for idx in result[0]]
                all_weights = [w for result in processed_results for w in result[1]]
                
                # Convert to numpy arrays
                indices = np.array(all_indices, dtype=np.int32)
                
                if len(indices) > 0:
                    # Normalize the weights
                    weights = np.array(all_weights, dtype=np.float32)
                    weights = weights / np.sum(weights)  # Normalize to sum to 1
                else:
                    weights = np.array([], dtype=np.float32)
                
                # Calculate total selection time
                selection_time = time.time() - start_time  # Changed from: selection_time = time() - start_time
                print(f"Mixed selection completed in {selection_time:.2f} seconds")
                
                return indices, weights
    
    def _joint_selection(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        softmax_preds: np.ndarray,
        size: int,
        dpp_weight: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Joint selection using the combined objective F(S) = f(S) + λD(S)
        where f(S) is the coverage function and D(S) is the diversity function.
        
        Args:
            features: Feature vectors for each sample
            labels: True labels
            softmax_preds: Model prediction probabilities
            size: Size of subset to select
            dpp_weight: Weight for diversity (λ in the theory)
            
        Returns:
            Tuple of (array of selected indices, weights for selected samples)
        """
        from submodlib import FacilityLocationFunction, LogDeterminantFunction
        
        if size <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
            
        if not self.greedy:
            # If not greedy, just return random samples with equal weights
            indices = np.random.choice(len(features), size, replace=False)
            weights = np.ones(size) / size
            return indices, weights
        
        # Normalize features for kernel computation
        X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(X, X.T)
        
        # Coverage function: Facility Location (submodular)
        # This approximates how well S represents the full dataset
        fl_obj = FacilityLocationFunction(
            n=len(features),
            mode="dense",
            sijs=similarity_matrix,
            separate_rep=False
        )
        
        # Diversity function: Log Determinant of kernel submatrix (DPP)
        # This ensures diversity among selected samples
        dpp_obj = LogDeterminantFunction(
            n=len(features),
            mode="dense",
            sijs=similarity_matrix,
            lambdaVal=1.0  # Regularization parameter within log det
        )
        
        # Create combined objective with weighted diversity
        # Following F(S) = f(S) + λD(S)
        from submodlib.functions.mixture import MixtureFunction
        
        # Define the mixture with weights: (1-dpp_weight) for coverage, dpp_weight for diversity
        mixture_obj = MixtureFunction(
                        functions=[fl_obj, dpp_obj],
            weights=[(1-dpp_weight), dpp_weight]
        )
        
        # Perform greedy optimization of the joint objective
        selected_indices = mixture_obj.maximize(
            budget=size,
            optimizer="LazyGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False
        )
        
        indices = np.array(selected_indices)
        
        # Calculate weights based on normalized similarity to other samples
        row_sums = np.sum(similarity_matrix[indices, :], axis=1)
        weights = row_sums / np.sum(row_sums)  # Normalize to sum to 1
        
        return indices, weights
        
    def _dpp_selection(self, features: np.ndarray, labels: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a diverse subset using Determinantal Point Process (DPP)
        Following the theory: D(S) = log det(L_S)
        
        Args:
            features: Feature vectors for each sample
            labels: True labels
            size: Size of subset to select
            
        Returns:
            Tuple of (array of selected indices, weights for selected samples)
        """
        try:
            # Try using submodlib's implementation first
            from submodlib import LogDeterminantFunction
            
            # Normalize features for kernel computation
            X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            similarity_matrix = np.dot(X, X.T)
            
            # Create LogDeterminant function (DPP objective)
            dpp_obj = LogDeterminantFunction(
                n=len(features),
                mode="dense",
                sijs=similarity_matrix,
                lambdaVal=1.0  # Regularization parameter
            )
            
            # Perform greedy optimization
            selected_indices = dpp_obj.maximize(
                budget=size,
                optimizer="LazyGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False
            )
            
            indices = np.array(selected_indices)
            
            # Calculate weights based on normalized diversity contribution
            if len(indices) > 0:
                # For DPP, we want to emphasize the diversity contribution of each point
                # Using diagonal elements of the kernel matrix as importance weights
                # This gives higher weight to points that are more orthogonal to others
                kernel_submatrix = similarity_matrix[np.ix_(indices, indices)]
                # We use diagonal elements which represent self-similarity
                diag_values = np.diag(kernel_submatrix)
                weights = diag_values / np.sum(diag_values)  # Normalize to sum to 1
            else:
                weights = np.array([], dtype=float)
                
            return indices, weights
            
        except ImportError:
            # Fall back to custom greedy DPP approximation
            if size <= 0:
                return np.array([], dtype=int), np.array([], dtype=float)
                
            # Compute similarity kernel
            X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            similarity = np.dot(X, X.T)
            
            # Basic greedy DPP selection - maximizing diversity
            selected = []
            selected_scores = []  # Store diversity scores for weight computation
            remaining = list(range(len(features)))
            
            if not self.greedy:
                # If not greedy, just return random samples
                indices = np.random.choice(len(features), size, replace=False)
                weights = np.ones(size) / size  # Equal weights
                return indices, weights
            
            # Greedy selection
            for i in range(min(size, len(features))):
                if not remaining:
                    break
                    
                if not selected:
                    # For first element, select point with maximum norm (most informative)
                    norms = np.linalg.norm(features[remaining], axis=1)
                    first_idx = remaining[np.argmax(norms)]
                    selected.append(first_idx)
                    selected_scores.append(1.0)  # Standard weight for first point
                    remaining.remove(first_idx)
                else:
                    # Compute scores for remaining items based on diversity
                    best_idx = -1
                    best_score = -float('inf')
                    
                    for idx in remaining:
                        # Compute minimum distance to already selected items
                        # Maximum minimum distance = maximum diversity
                        sim_to_selected = similarity[idx, selected]
                        min_sim = np.max(sim_to_selected)  # Highest similarity to any selected point
                        diversity_score = -min_sim  # Negative because lower similarity = higher diversity
                        
                        if diversity_score > best_score:
                            best_score = diversity_score
                            best_idx = idx
                    
                    selected.append(best_idx)
                    selected_scores.append(-best_score)  # Convert back to positive weight
                    remaining.remove(best_idx)
            
            # Convert to numpy array
            indices = np.array(selected)
            
            # Calculate weights based on diversity contribution
            if len(indices) > 0:
                weights = np.array(selected_scores)
                # Ensure positive weights
                weights = np.abs(weights) + 1e-8
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
            else:
                weights = np.array([], dtype=float)
                    
            return indices, weights
    
    def _submodular_selection(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        softmax_preds: np.ndarray,
        size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select a representative subset using Facility Location submodular optimization
        as used in CREST, for better coverage guarantees.
        
        Args:
            features: Feature vectors for each sample
            labels: True labels
            softmax_preds: Model prediction probabilities
            size: Size of subset to select
            
        Returns:
            Tuple of (array of selected indices, weights for selected samples)
        """
        try:
            # Try to import submodlib - make sure it's installed
            from submodlib import FacilityLocationFunction
        except ImportError:
            print("SubModLib not found. Please install with: pip install submodlib")
            # Fall back to the current implementation
            return self._legacy_submodular_selection(features, labels, softmax_preds, size)
            
        if size <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
            
        if not self.greedy:
            # If not greedy, just return random samples
            indices = np.random.choice(len(features), size, replace=False)
            weights = np.ones(size) / size  # Equal weights
            return indices, weights
        
        # Normalize features for cosine similarity
        X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Create similarity matrix (cosine similarity)
        similarity_matrix = np.dot(X, X.T)
        
        # Instantiate the Facility Location function
        # This maximizes representation by selecting points that are
        # most similar to other points in the dataset (coverage)
        fl_obj = FacilityLocationFunction(
            n=len(features),
            mode="dense",
            sijs=similarity_matrix,
            separate_rep=False
        )
        
        # Perform greedy optimization
        selected_indices = fl_obj.maximize(
            budget=size,
            optimizer="LazyGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False
        )
        
        indices = np.array(selected_indices)
        
        # Calculate weights based on how representative each point is
        # Points that are more similar to the rest of the dataset get higher weights
        if len(indices) > 0:
            # Calculate representation score for each selected point based on
            # how well it represents the entire dataset (row sum of similarities)
            rep_scores = np.sum(similarity_matrix[indices, :], axis=1)
            weights = rep_scores / np.sum(rep_scores)  # Normalize to sum to 1
        else:
            weights = np.array([], dtype=float)
        
        return indices, weights


    def _legacy_submodular_selection(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        softmax_preds: np.ndarray,
        size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Legacy implementation of submodular selection without SubModLib
        
        Args:
            features: Feature vectors for each sample
            labels: True labels
            softmax_preds: Model prediction probabilities
            size: Size of subset to select
            
        Returns:
            Tuple of (array of selected indices, weights for selected samples)
        """
        num_classes = softmax_preds.shape[1]
        
        # Calculate entropy for each sample
        entropies = -np.sum(softmax_preds * np.log(softmax_preds + 1e-10), axis=1)
        
        # Calculate per-class uncertainty
        class_uncertainty = {}
        for c in range(num_classes):
            # Samples where true class is c
            class_samples = np.where(labels == c)[0]
            if len(class_samples) > 0:
                # Uncertainty defined as 1 - P(true class)
                uncertainty = 1 - softmax_preds[class_samples, c]
                class_uncertainty[c] = (class_samples, uncertainty)
        
        selected = []
        selected_scores = []  # Store scores for weight computation
        remaining = list(range(len(features)))
        class_counts = np.zeros(num_classes)
        
        # Greedy selection trying to maximize coverage of uncertain samples
        for _ in range(min(size, len(features))):
            if not remaining:
                break
            
            best_idx = -1
            best_score = -float('inf')
            
            for idx in remaining:
                # Class of current sample
                cls = labels[idx]
                
                # Combine multiple objectives:
                # 1. Entropy (higher is better)
                # 2. Class balance (underrepresented classes preferred)
                # 3. Sample uncertainty for its own class
                class_balance_term = -class_counts[cls] / (sum(class_counts) + 1e-10)
                
                # Calculate score
                score = (
                    entropies[idx] +                    # Entropy term
                    class_balance_term * self.smtk +    # Class balance term with smoothing
                    (1 - softmax_preds[idx, cls])       # Uncertainty term
                )
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                # Store the score as the weight basis - higher scores indicate more important samples
                # The score combines entropy, class balance, and prediction uncertainty
                selected_scores.append(best_score)
                remaining.remove(best_idx)
                # Update class count
                class_counts[labels[best_idx]] += 1
        
        indices = np.array(selected)
        
        # Compute weights based on the selection scores
        if len(indices) > 0:
            # Convert scores to weights - higher scores get higher weights
            weights = np.array(selected_scores)
            # Ensure positive weights by shifting if needed
            if np.min(weights) < 0:
                weights = weights - np.min(weights) + 1e-8
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
        else:
            weights = np.array([], dtype=float)
                
        return indices, weights

    def generate_spectral_influence_subset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        softmax_preds: np.ndarray,
        subset_size: int,
        n_clusters_factor: float = 0.1,
        influence_type: str = "gradient_norm",
        balance_clusters: bool = True,
        affinity_metric: str = "rbf",
        true_gradients: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a diverse and representative subset using spectral clustering and influence functions.
        
        This approach is an alternative to DPP-based selection that directly leverages the manifold
        structure of the data through spectral clustering, and prioritizes influential samples.
        
        Args:
            features: Feature representations of samples (e.g., embeddings or gradients)
            labels: Class labels for the samples
            softmax_preds: Model's predictions (e.g., softmax outputs)
            subset_size: Size of the subset to select
            n_clusters_factor: Factor to determine number of clusters
            influence_type: Method to compute influence scores
            balance_clusters: Whether to enforce balanced selection across clusters
            affinity_metric: Metric for computing the affinity matrix
            true_gradients: Pre-computed gradients if available (optional)
            
        Returns:
            Tuple of (selected indices, selection weights)
        """
        # Import our spectral influence selector
        from datasets.spectral_influence import SpectralInfluenceSelector
        
        # Create the selector
        selector = SpectralInfluenceSelector(
            n_clusters_factor=n_clusters_factor,
            influence_type=influence_type,
            balance_clusters=balance_clusters,
            affinity_metric=affinity_metric
        )
        
        # Use it to generate a subset
        start_time = time.time()
        selected_indices, weights = selector.generate_subset(
            features=features,
            labels=labels,
            model_outputs=softmax_preds,
            subset_size=subset_size,
            true_gradients=true_gradients
        )
        selection_time = time.time() - start_time
        
        print(f"Spectral influence selection completed in {selection_time:.2f} seconds")
        
        return selected_indices, weights

    def generate_subset(
        self,
        preds,
        epoch,
        B,
        idx,
        targets,
        subset_printer=None,
        mode="dense",
        num_n=128,
        use_submodlib=True,
    ):
        if subset_printer is not None:
            subset_printer.print_selection(self.mode, epoch)

        fl_labels = targets[idx] - np.min(targets[idx])

        if len(fl_labels) > 50000:
            (
                subset,
                weight,
                _,
                _,
                ordering_time,
                similarity_time,
            ) = submodular.greedy_merge(preds, fl_labels, B, 5, "euclidean",)
        else:
            if use_submodlib:
                (
                    subset,
                    weight,
                    _,
                    _,
                    ordering_time,
                    similarity_time,
                ) = submodular.get_orders_and_weights(
                    B,
                    preds,
                    "euclidean",
                    y=fl_labels,
                    equal_num=True,
                    mode=mode,
                    num_n=num_n,
                )
            else:
                (
                    subset,
                    weight,
                    _,
                    _,
                    ordering_time,
                    similarity_time,
                ) = craig.get_orders_and_weights(
                    B, preds, "euclidean", y=fl_labels, equal_num=True, smtk=self.smtk
                )

            subset = np.array(idx[subset])  # (Note): idx
            weight = np.array(weight)

        return subset, weight, ordering_time, similarity_time