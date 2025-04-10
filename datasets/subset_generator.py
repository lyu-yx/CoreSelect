import numpy as np
from typing import Tuple, List, Optional
import torch

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
            weights = np.ones(len(indices))
            return indices, weights
            
        elif selection_method == "dpp":
            # Pure DPP selection for diversity
            indices = self._dpp_selection(features, labels, subset_size)
            weights = np.ones(len(indices))
            return indices, weights
            
        elif selection_method == "submod":
            # Pure submodular selection for coverage
            indices = self._submodular_selection(features, labels, softmax_preds, subset_size)
            weights = np.ones(len(indices))
            return indices, weights
            
        else:  # "mixed" or default - this is the joint objective approach
            try:
                # Use joint optimization if submodlib is available
                indices = self._joint_selection(features, labels, softmax_preds, subset_size, dpp_weight)
                weights = np.ones(len(indices))
                return indices, weights
            except ImportError:
                # Fall back to sequential approach if submodlib isn't available
                dpp_size = int(subset_size * dpp_weight)
                submod_size = subset_size - dpp_size
                
                # Get indices from both methods
                dpp_indices = self._dpp_selection(features, labels, dpp_size)
                
                # Exclude already selected samples for the submodular part
                mask = np.ones(len(features), dtype=bool)
                mask[dpp_indices] = False
                remaining_features = features[mask]
                remaining_labels = labels[mask]
                remaining_softmax = softmax_preds[mask]
                
                # Map the original indices
                original_indices = np.arange(len(features))[mask]
                
                # Get submodular indices from remaining samples
                if submod_size > 0 and len(remaining_features) > 0:
                    submod_rel_indices = self._submodular_selection(
                        remaining_features, 
                        remaining_labels,
                        remaining_softmax, 
                        min(submod_size, len(remaining_features))
                    )
                    submod_indices = original_indices[submod_rel_indices]
                else:
                    submod_indices = np.array([], dtype=int)
                
                # Combine indices
                indices = np.concatenate([dpp_indices, submod_indices])
                
                # Set weights (could be adjusted based on importance)
                weights = np.ones(len(indices))
                
                return indices, weights
    
    def _joint_selection(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        softmax_preds: np.ndarray,
        size: int,
        dpp_weight: float = 0.5
    ) -> np.ndarray:
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
            Array of selected indices
        """
        from submodlib import FacilityLocationFunction, LogDeterminantFunction
        
        if size <= 0:
            return np.array([], dtype=int)
            
        if not self.greedy:
            # If not greedy, just return random samples
            return np.random.choice(len(features), size, replace=False)
        
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
        
        return np.array(selected_indices)
        
    def _dpp_selection(self, features: np.ndarray, labels: np.ndarray, size: int) -> np.ndarray:
        """
        Select a diverse subset using Determinantal Point Process (DPP)
        Following the theory: D(S) = log det(L_S)
        
        Args:
            features: Feature vectors for each sample
            labels: True labels
            size: Size of subset to select
            
        Returns:
            Array of selected indices
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
            
            return np.array(selected_indices)
            
        except ImportError:
            # Fall back to custom greedy DPP approximation
            if size <= 0:
                return np.array([], dtype=int)
                
            # Compute similarity kernel
            # Use cosine similarity or RBF kernel
            X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            similarity = np.dot(X, X.T)
            
            # Basic greedy DPP selection
            selected = []
            remaining = list(range(len(features)))
            
            if not self.greedy:
                # If not greedy, just return random samples
                return np.random.choice(len(features), size, replace=False)
            
            # Greedy selection
            for _ in range(min(size, len(features))):
                if not remaining:
                    break
                    
                if not selected:
                    # Select first element (could use uncertainty or other metric)
                    selected.append(remaining.pop(0))
                else:
                    # Compute scores for remaining items based on diversity
                    scores = []
                    for idx in remaining:
                        # Compute score based on similarity to already selected items
                        # Lower similarity = higher diversity
                        sim_score = np.mean([similarity[idx, sel_idx] for sel_idx in selected])
                        scores.append(-sim_score)  # Negative because we want diverse examples
                    
                    # Select item with highest diversity score
                    best_idx = np.argmax(scores)
                    selected.append(remaining.pop(best_idx))
                    
            return np.array(selected)
    
    def _submodular_selection(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        softmax_preds: np.ndarray,
        size: int
    ) -> np.ndarray:
        """
        Select a representative subset using Facility Location submodular optimization
        as used in CREST, for better coverage guarantees.
        
        Args:
            features: Feature vectors for each sample
            labels: True labels
            softmax_preds: Model prediction probabilities
            size: Size of subset to select
            
        Returns:
            Array of selected indices
        """
        try:
            # Try to import submodlib - make sure it's installed
            from submodlib import FacilityLocationFunction
        except ImportError:
            print("SubModLib not found. Please install with: pip install submodlib")
            # Fall back to the current implementation
            return self._legacy_submodular_selection(features, labels, softmax_preds, size)
            
        if size <= 0:
            return np.array([], dtype=int)
            
        if not self.greedy:
            # If not greedy, just return random samples
            return np.random.choice(len(features), size, replace=False)
        
        # Normalize features for cosine similarity
        X = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Create similarity matrix (cosine similarity)
        similarity_matrix = np.dot(X, X.T)
        
        # Instantiate the Facility Location function
        # This maximizes representation by selecting points that are
        # most similar to other points in the dataset
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
        
        return np.array(selected_indices)

    def _legacy_submodular_selection(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        softmax_preds: np.ndarray,
        size: int
    ) -> np.ndarray:
        """Legacy implementation of submodular selection without SubModLib"""
        # Copy the existing implementation here as a fallback
        # This would contain all the current code from the original _submodular_selection
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
                remaining.remove(best_idx)
                # Update class count
                class_counts[labels[best_idx]] += 1
                
        return np.array(selected)
