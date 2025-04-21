import numpy as np
from typing import Tuple, List, Dict, Optional
import torch
import time
from sklearn.metrics.pairwise import rbf_kernel

def trimodal_mixed_selection(
    features: np.ndarray,
    labels: np.ndarray,
    softmax_preds: np.ndarray,
    subset_size: int,
    dpp_weight: float = 0.3,
    submod_weight: float = 0.3,
    spectral_weight: float = 0.4,
    n_clusters_factor: float = 0.1,
    influence_type: str = "gradient_norm",
    balance_clusters: bool = True,
    affinity_metric: str = "rbf",
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A trimodal selection method that combines submodular coverage, DPP diversity,
    and spectral influence in a unified framework.
    
    This approach:
    1. Allocates the selection budget across three methods
    2. Runs each method on the allocated budget
    3. Combines the results with careful deduplication and weighting
    
    Args:
        features: Feature representations of samples (e.g., embeddings or gradients)
        labels: Class labels for the samples
        softmax_preds: Model's predictions (e.g., softmax outputs)
        subset_size: Total size of the subset to select
        dpp_weight: Weight for DPP selection (determines portion of budget)
        submod_weight: Weight for submodular selection (determines portion of budget)
        spectral_weight: Weight for spectral influence selection (determines portion of budget)
        n_clusters_factor: Factor for spectral clustering
        influence_type: Method to compute influence scores
        balance_clusters: Whether to enforce balanced selection across clusters
        affinity_metric: Metric for computing the affinity matrix
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (selected indices, selection weights)
    """
    # Normalize weights to sum to 1
    total_weight = dpp_weight + submod_weight + spectral_weight
    if total_weight <= 0:
        # If all weights are zero or negative, use equal weights
        dpp_weight = submod_weight = spectral_weight = 1/3
    else:
        # Normalize weights
        dpp_weight /= total_weight
        submod_weight /= total_weight
        spectral_weight /= total_weight
    
    # Calculate budget allocation for each method
    dpp_budget = max(1, int(np.round(subset_size * dpp_weight)))
    submod_budget = max(1, int(np.round(subset_size * submod_weight)))
    # Allocate remaining budget to spectral to ensure we get exactly subset_size samples
    spectral_budget = subset_size - dpp_budget - submod_budget
    
    if spectral_budget <= 0:
        # Rebalance if spectral budget went negative
        if dpp_budget > 1 and submod_budget > 1:
            # Reduce both evenly
            reduction = (1 - spectral_budget) // 2
            dpp_budget -= reduction
            submod_budget -= (1 - spectral_budget - reduction)
        elif dpp_budget > 1:
            # Reduce DPP
            dpp_budget += spectral_budget - 1
        elif submod_budget > 1:
            # Reduce submod
            submod_budget += spectral_budget - 1
        spectral_budget = 1
    
    print(f"Budget allocation: DPP={dpp_budget}, Submod={submod_budget}, Spectral={spectral_budget}")
    
    # Prepare for tracking the selected indices and weights from each method
    all_selected_indices = []
    all_weights = []
    selection_methods = []
    used_indices = set()
    
    # Normalize features for similarity calculations
    normalized_features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix once and reuse it
    similarity_matrix = np.dot(normalized_features, normalized_features.T) if affinity_metric == "cosine" else None
    if affinity_metric == "rbf":
        # RBF kernel with adaptive bandwidth
        median_dist = np.median(scipy.spatial.distance.pdist(normalized_features))
        gamma = 1.0 / (2 * median_dist**2) if median_dist > 0 else 1.0
        similarity_matrix = rbf_kernel(normalized_features, gamma=gamma)
        
    # 1. Run DPP selection
    if dpp_budget > 0:
        try:
            # Try using submodlib's implementation
            from submodlib import LogDeterminantFunction
            
            # Create LogDeterminant function (DPP objective) - reuse existing similarity matrix
            dpp_obj = LogDeterminantFunction(
                n=len(features),
                mode="dense",
                sijs=similarity_matrix,
                lambdaVal=1.0
            )
            
            # Perform greedy optimization
            dpp_indices = dpp_obj.maximize(
                budget=dpp_budget,
                optimizer="LazyGreedy",
                stopIfZeroGain=False,
                stopIfNegativeGain=False,
                verbose=False
            )
            
            dpp_indices = np.array(dpp_indices)
            
            if len(dpp_indices) > 0:
                # Calculate weights based on normalized diversity contribution
                kernel_submatrix = similarity_matrix[np.ix_(dpp_indices, dpp_indices)]
                diag_values = np.diag(kernel_submatrix)
                dpp_weights = diag_values / np.sum(diag_values)
                
                # Store results
                all_selected_indices.append(dpp_indices)
                all_weights.append(dpp_weights * dpp_weight)  # Scale by method weight
                selection_methods.append("dpp")
                used_indices.update(dpp_indices)
            
        except Exception as e:
            print(f"DPP selection failed: {e}. Reallocating budget.")
            # Redistribute the budget to other methods
            extra = dpp_budget // 2
            submod_budget += extra
            spectral_budget += (dpp_budget - extra)
            dpp_budget = 0
    
    # 2. Run submodular selection
    if submod_budget > 0:
        try:
            # Try to import submodlib
            from submodlib import FacilityLocationFunction
            
            # Create a mask to avoid selecting already selected indices from DPP
            available_mask = np.ones(len(features), dtype=bool)
            if used_indices:
                available_mask[list(used_indices)] = False
            
            # Instantiate Facility Location function only on available indices
            available_indices = np.where(available_mask)[0]
            if len(available_indices) > 0:
                # Extract submatrix for available indices
                sub_matrix = similarity_matrix[np.ix_(available_indices, available_indices)]
                
                fl_obj = FacilityLocationFunction(
                    n=len(available_indices),
                    mode="dense",
                    sijs=sub_matrix,
                    separate_rep=False
                )
                
                # Perform greedy optimization
                local_indices = fl_obj.maximize(
                    budget=min(submod_budget, len(available_indices)),
                    optimizer="LazyGreedy",
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False
                )
                
                # Map back to global indices
                submod_indices = available_indices[local_indices]
                
                if len(submod_indices) > 0:
                    # Calculate representativeness scores
                    rep_scores = np.sum(similarity_matrix[submod_indices, :], axis=1)
                    submod_weights = rep_scores / np.sum(rep_scores)
                    
                    # Store results
                    all_selected_indices.append(submod_indices)
                    all_weights.append(submod_weights * submod_weight)  # Scale by method weight
                    selection_methods.append("submod")
                    used_indices.update(submod_indices)
            
        except Exception as e:
            print(f"Submodular selection failed: {e}. Reallocating budget.")
            # Redistribute the budget
            spectral_budget += submod_budget
            submod_budget = 0
    
    # 3. Run spectral influence selection
    if spectral_budget > 0:
        try:
            # Import our spectral influence selector
            from datasets.spectral_influence import SpectralInfluenceSelector
            
            # Create a mask to avoid selecting already selected indices
            available_mask = np.ones(len(features), dtype=bool)
            if used_indices:
                available_mask[list(used_indices)] = False
            
            available_indices = np.where(available_mask)[0]
            if len(available_indices) > 0:
                # Extract data for available indices
                avail_features = features[available_indices]
                avail_labels = labels[available_indices]
                avail_softmax = softmax_preds[available_indices]
                
                # Create the selector
                selector = SpectralInfluenceSelector(
                    n_clusters_factor=n_clusters_factor,
                    influence_type=influence_type,
                    balance_clusters=balance_clusters,
                    affinity_metric=affinity_metric,
                    random_state=random_state
                )
                
                # Run spectral influence selection
                local_indices, local_weights = selector.generate_subset(
                    features=avail_features,
                    labels=avail_labels,
                    model_outputs=avail_softmax,
                    subset_size=min(spectral_budget, len(available_indices)),
                    true_gradients=None  # Use approximation
                )
                
                if len(local_indices) > 0:
                    # Map back to global indices
                    spectral_indices = available_indices[local_indices]
                    
                    # Store results
                    all_selected_indices.append(spectral_indices)
                    all_weights.append(local_weights * spectral_weight)  # Scale by method weight
                    selection_methods.append("spectral")
                    used_indices.update(spectral_indices)
        
        except Exception as e:
            print(f"Spectral selection failed: {e}.")
            spectral_budget = 0
    
    # Combine all selected indices and weights
    all_indices = []
    all_combined_weights = []
    
    for indices, weights, method in zip(all_selected_indices, all_weights, selection_methods):
        all_indices.extend(indices)
        all_combined_weights.extend(weights)
    
    if not all_indices:
        # Fallback to random selection if all methods failed
        print("All selection methods failed. Falling back to random selection.")
        all_indices = np.random.choice(len(features), size=min(subset_size, len(features)), replace=False)
        all_combined_weights = np.ones(len(all_indices)) / len(all_indices)
    
    # Convert to numpy arrays
    all_indices = np.array(all_indices)
    all_combined_weights = np.array(all_combined_weights)
    
    # If we have more samples than requested, prioritize by weight
    if len(all_indices) > subset_size:
        top_indices = np.argsort(-all_combined_weights)[:subset_size]
        all_indices = all_indices[top_indices]
        all_combined_weights = all_combined_weights[top_indices]
    
    # Normalize weights to sum to 1
    if len(all_combined_weights) > 0:
        all_combined_weights = all_combined_weights / np.sum(all_combined_weights)
    
    return all_indices, all_combined_weights