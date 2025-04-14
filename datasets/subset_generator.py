import numpy as np
from typing import Tuple, List, Optional
import torch
from enum import Enum

import numpy as np
import torch
from utils import submodular, craig


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
        Generate a subset using a combination of DPP and submodular methods
        for better coverage and diversity.
        
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
            selection_method: Method for selection ("mixed", "dpp", "submod", "rand")
            
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
            
        else:  # "mixed" or default - this is the joint objective approach
            try:
                # Use joint optimization if submodlib is available
                indices, weights = self._joint_selection(features, labels, softmax_preds, subset_size, dpp_weight)
                return indices, weights
            except ImportError:
                # Fall back to sequential approach if submodlib isn't available
                print("SubModLib's MixtureFunction not found. Using custom joint selection.")
                
                # Implement our own joint selection that properly balances diversity and coverage
                # This is a custom implementation combining DPP and Facility Location principles
                if not self.greedy:
                    # If not greedy, just return random samples with equal weights
                    indices = np.random.choice(len(features), subset_size, replace=False)
                    weights = np.ones(subset_size) / subset_size
                    return indices, weights
                
                # Normalize features for kernel computation
                X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
                similarity_matrix = np.dot(X, X.T)
                
                # Initialize selection data structures
                selected = []
                remaining = list(range(len(features)))
                selected_scores = []
                
                # Greedy joint selection - combining both coverage and diversity objectives
                for i in range(min(size, len(features))):
                    if not remaining:
                        break
                    
                    best_idx = -1
                    best_score = -float('inf')
                    
                    for idx in remaining:
                        # COVERAGE TERM: How well this point represents the dataset
                        # Higher similarity to many points = better coverage
                        coverage_score = np.sum(similarity_matrix[idx, :]) / len(features)
                        
                        # DIVERSITY TERM: How different this point is from already selected points
                        # Lower similarity to selected = better diversity
                        diversity_score = 0
                        if selected:
                            sim_to_selected = np.mean(similarity_matrix[idx, selected])
                            diversity_score = 1 - sim_to_selected  # Invert for diversity
                        
                        # Combined score with weighting
                        score = (1 - dpp_weight) * coverage_score + dpp_weight * diversity_score
                        
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    
                    if best_idx != -1:
                        selected.append(best_idx)
                        selected_scores.append(best_score)
                        remaining.remove(best_idx)
                
                indices = np.array(selected)
                
                # Calculate weights based on the selection scores
                if len(indices) > 0:
                    weights = np.array(selected_scores)
                    # Ensure positive weights
                    if np.min(weights) < 0:
                        weights = weights - np.min(weights) + 1e-8
                    # Normalize weights to sum to 1
                    weights = weights / np.sum(weights)
                else:
                    weights = np.array([], dtype=float)
                
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
            lambda_val=1.0  # Regularization parameter within log det
        )
        
        # Create combined objective with weighted diversity
        # Following F(S) = f(S) + λD(S)
        from submodlib.functions.mixture import MixtureFunction
        
        # Define the mixture with weights: (1-dpp_weight) for coverage, dpp_weight for diversity
        mixture_obj = MixtureFunction(
            n=len(features),
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
                lambda_val=1.0  # Regularization parameter
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