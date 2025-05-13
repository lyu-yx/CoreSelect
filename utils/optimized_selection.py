import time
import numpy as np
from typing import Tuple, List, Set, Dict, Optional
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.functions.logDeterminant import LogDeterminantFunction


def optimized_joint_selection(
    features: np.ndarray,
    target_count: int,
    fl_obj,
    dpp_obj=None,
    dpp_weight: float = 0.3,
    batch_size: int = 256,
    progress_interval: int = 5,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Highly optimized implementation for joint facility location and DPP selection.
    
    Args:
        features: Feature vectors for samples
        target_count: Number of samples to select
        fl_obj: Facility location objective function
        dpp_obj: Log determinant (DPP) objective function
        dpp_weight: Weight for diversity (0=pure coverage, 1=pure diversity)
        batch_size: Size of batches for processing
        progress_interval: How often to print progress
        verbose: Whether to print progress
        
    Returns:
        Tuple of (selected indices, importance weights)
    """
    N = features.shape[0]
    
    # Using optimized batch implementation directly since MixtureFunction is not available
    if verbose:
        print(f"Using custom batch implementation for selection")
    start_time = time.time()
        
        # Fall back to our optimized batch implementation
        selected = set()
        ground_set = list(range(N))
        remaining = ground_set.copy()
        
        # Initialize timing
        start_time = time.time()
        last_progress = start_time
        
        # Pre-allocate arrays
        selected_indices = []
        selected_gains = []
        
        # For each selection round
        for i in range(min(target_count, N)):
            if not remaining:
                break
                
            # Compute all gains for remaining points in batches
            fl_gains, dpp_gains, combined_gains = compute_joint_marginal_gains(
                remaining_indices=remaining,
                selected_indices=selected,
                fl_obj=fl_obj,
                dpp_obj=dpp_obj,
                dpp_weight=dpp_weight,
                batch_size=batch_size,
                progress_interval=progress_interval,
                verbose=verbose and (time.time() - last_progress > 5)
            )
            
            # Find best element 
            if combined_gains:
                best_idx = max(combined_gains, key=combined_gains.get)
                best_gain = combined_gains[best_idx]
                
                # Add to selection
                selected.add(best_idx)
                selected_indices.append(best_idx)
                selected_gains.append(best_gain)
                
                # Remove from candidates
                remaining.remove(best_idx)
            else:
                break
                
            # Print progress
            if verbose and (time.time() - last_progress > 5 or i == target_count - 1):
                last_progress = time.time()
                print(f"Selected {i+1}/{target_count} elements in {time.time() - start_time:.3f}s")
        
        if verbose:
            print(f"Selection completed in {time.time() - start_time:.3f}s")
            
        return np.array(selected_indices), np.array(selected_gains)


def optimized_mixed_selection(
    features: np.ndarray,
    labels: np.ndarray,
    target_count: int,
    class_indices: np.ndarray = None,
    dpp_weight: float = 0.3,
    metric: str = "cosine",
    batch_size: int = 512,
    progress_interval: int = 10,
    verbose: bool = False,
    mode: str = "dense"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized implementation of mixed selection combining facility location (coverage)
    and DPP (diversity) in a single pass.
    
    Args:
        features: Feature vectors for samples
        labels: Optional class labels if using class-based selection
        target_count: Number of samples to select 
        class_indices: Optional class indices if using a subset
        dpp_weight: Weight for diversity vs. coverage (0 = pure coverage, 1 = pure diversity)
        metric: Similarity metric ("cosine" or "euclidean")
        batch_size: Size of batches for processing
        progress_interval: How often to print progress (iterations)
        verbose: Whether to print detailed progress
        mode: Whether to use dense or sparse similarity computations
        
    Returns:
        Tuple of (selected indices, weights)
    """
    # Extract class-specific features if needed
    if class_indices is not None:
        X = features[class_indices]
        N = X.shape[0]
    else:
        X = features
        N = X.shape[0]
        class_indices = np.arange(N)
    
    # Initialize submodular functions
    fl_obj = FacilityLocationFunction(
        n=N, 
        mode=mode,
        data=X,
        metric=metric
    )
    
    # Create DPP objective if using diversity
    if dpp_weight > 0:
        try:
            # Try to create the LogDeterminantFunction with error handling
            # Try with lambdaVal first
            try:
                dpp_obj = LogDeterminantFunction(
                    n=N,
                    mode=mode,
                    data=X,
                    metric=metric,
                    lambdaVal=1.0  # Adding required lambda parameter (regularization)
                )
            except TypeError:
                # If lambdaVal is not accepted, try without it
                dpp_obj = LogDeterminantFunction(
                    n=N,
                    mode=mode,
                    data=X,
                    metric=metric
                )
        except Exception as e:
            print(f"Warning: Failed to initialize DPP objective: {str(e)}. Using pure facility location.")
            dpp_obj = None
            dpp_weight = 0.0  # Fall back to pure facility location
    else:
        dpp_obj = None
    
    # Efficient greedy selection
    selected = set()
    remaining = list(range(N))
    
    # Pre-allocate result containers
    selected_indices = []
    selected_gains = []
    
    # Timing stats
    selection_start = time.time()
    
    # Use optimized joint selection to dramatically speed up the process
    if dpp_weight > 0 and dpp_obj is not None:
        # Use our optimized joint selection
        if verbose:
            print(f"Using optimized joint selection with DPP weight {dpp_weight}")
            
        # Call optimized joint selection function
        sel_indices, sel_gains = optimized_joint_selection(
            features=X,
            target_count=target_count,
            fl_obj=fl_obj,
            dpp_obj=dpp_obj,
            dpp_weight=dpp_weight,
            batch_size=batch_size,
            progress_interval=progress_interval,
            verbose=verbose
        )
        
        # Store results
        selected_indices = sel_indices.tolist()
        selected_gains = sel_gains.tolist()
        
        # Update selected set for later use
        selected = set(selected_indices)
    else:
        # Pure facility location - use built-in maximize for efficiency
        if verbose:
            print(f"Using pure facility location selection")
            
        # Use the built-in maximize function which is highly optimized
        greedyList = fl_obj.maximize(
            budget=target_count,
            optimizer="LazyGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=verbose
        )
        
        # Extract selected elements
        for item in greedyList:
            idx, gain = item[0], item[1]
            selected_indices.append(idx)
            selected_gains.append(gain)
            selected.add(idx)
            
    if verbose:
        selection_time = time.time() - selection_start
        print(f"Selection completed in {selection_time:.4f}s, selected {len(selected_indices)} elements")
    
    # Convert to numpy array and map to original indices
    selected_indices = np.array(selected_indices, dtype=np.int32)
    global_indices = class_indices[selected_indices]
    
    # Compute weights based on similarity to all points
    S = fl_obj.sijs  # Get similarity matrix from facility location objective
    
    # Calculating cluster sizes
    cluster_sizes = np.zeros(len(selected_indices), dtype=np.float64)
    
    # Vectorized operations to compute weights
    if mode == "dense":
        # For dense mode, we can do it efficiently
        # For each point i, find the most similar selected point
        for i in range(N):
            max_sim_idx = np.argmax(S[i, selected_indices])
            cluster_sizes[max_sim_idx] += 1
    else:
        # For sparse mode, we need to be careful
        for i in range(N):
            best_sim = 0
            best_idx = -1
            for j, idx in enumerate(selected_indices):
                sim = S[i, idx]
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            if best_idx >= 0:
                cluster_sizes[best_idx] += 1
    
    # Avoid zeros in weights
    cluster_sizes[cluster_sizes == 0] = 1.0
    
    # Return selected indices and their weights
    return global_indices, cluster_sizes


def facility_location_joint_objective(
    c, X, y, metric, num_per_class, dpp_weight=0.3, weights=None, mode="sparse", num_n=128, 
    batch_size=512, progress_interval=10, verbose=False
):
    """
    Optimized implementation of facility location with joint FL and DPP objectives.
    
    Args:
        c: Class label
        X: Feature matrix
        y: Labels array 
        metric: Similarity metric
        num_per_class: Number of elements to select per class
        dpp_weight: Weight for diversity vs. coverage (0 = pure FL, 1 = pure DPP)
        weights: Optional weights for individual points
        mode: Computation mode ("sparse" or "dense")
        num_n: Number of neighbors for sparse mode
        batch_size: Batch size for gain computation
        progress_interval: How often to print progress
        verbose: Whether to print detailed progress
        
    Returns:
        Tuple of (selected indices, selection weights, timing statistics)
    """
    # Get class-specific indices
    class_indices = np.where(y == c)[0]
    
    if len(class_indices) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64), 0, 0
    
    # Set mode based on dataset size
    if mode == "dense" or len(class_indices) <= 5000:
        actual_mode = "dense"
        num_n = None
    else:
        actual_mode = "sparse"
    
    # Time similarity computation
    sim_start = time.time()
    
    # Run the optimized selection
    selection_start = time.time()
    
    # Call the optimized mixed selection function
    selected_indices, cluster_sizes = optimized_mixed_selection(
        features=X,
        labels=y,
        target_count=num_per_class,
        class_indices=class_indices,
        dpp_weight=dpp_weight,
        metric=metric,
        batch_size=batch_size,
        progress_interval=progress_interval,
        verbose=verbose,
        mode=actual_mode
    )
    
    selection_time = time.time() - selection_start
    sim_time = selection_time  # Since similarity computation is integrated
    
    # Apply weights if provided
    if weights is not None:
        # Map weights to selected indices
        weighted_sizes = np.zeros_like(cluster_sizes)
        for i, idx in enumerate(selected_indices):
            original_idx = class_indices[idx] if idx < len(class_indices) else -1
            if original_idx >= 0 and original_idx < len(weights):
                weighted_sizes[i] = weights[original_idx]
        
        # Replace zeros with ones to avoid numerical issues
        weighted_sizes[weighted_sizes == 0] = 1.0
        cluster_sizes = weighted_sizes
    
    return selected_indices, cluster_sizes, selection_time, sim_time


def get_orders_and_weights_optimized(
    B, X, metric, y=None, weights=None, equal_num=False, dpp_weight=0.3, 
    mode="auto", batch_size=512, progress_interval=10, verbose=False
):
    """
    Get optimized selection orders and weights for all classes.
    
    Args:
        B: Number of elements to select
        X: Feature matrix
        metric: Similarity metric
        y: Labels array
        weights: Optional weights for elements
        equal_num: Whether to select equal number from each class
        dpp_weight: Weight for diversity vs. coverage (0 = pure FL, 1 = pure DPP)
        mode: Computation mode ("dense", "sparse", or "auto")
        batch_size: Batch size for gain computation
        progress_interval: How often to print progress
        verbose: Whether to print detailed progress
        
    Returns:
        Selection orders and weights
    """
    N = X.shape[0]
    
    # Use appropriate mode based on dataset size
    if mode == "auto":
        mode = "dense" if N <= 10000 else "sparse"
    
    # Handle cases with no labels
    if y is None:
        y = np.zeros(N, dtype=np.int32)
    
    # Get unique classes
    classes = np.unique(y)
    C = len(classes)
    
    # Determine number per class
    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = np.array([class_nums[i] < np.ceil(B / C) for i in range(len(classes))])
        
        if np.any(minority):
            extra = sum([max(0, np.ceil(B / C) - class_nums[i]) for i in range(len(classes))])
            for i in range(len(classes)):
                if not minority[i]:
                    num_per_class[i] += int(np.ceil(extra / np.sum(~minority)))
    else:
        class_counts = np.array([np.sum(y == c) for c in classes])
        class_fractions = class_counts / N
        num_per_class = np.ceil(class_fractions * B).astype(np.int32)
    
    if verbose:
        print(f"Optimized selection: targeting {num_per_class} elements per class")
    
    # Process each class
    all_indices = []
    all_weights = []
    all_timing = []
    all_sim_timing = []
    
    # Process classes sequentially
    for c_idx, c in enumerate(classes):
        if verbose:
            print(f"Processing class {c} ({c_idx+1}/{C})")
        
        indices, weights, timing, sim_timing = facility_location_joint_objective(
            c=c,
            X=X,
            y=y,
            metric=metric,
            num_per_class=num_per_class[c_idx],
            dpp_weight=dpp_weight,
            weights=weights,
            mode=mode,
            batch_size=batch_size,
            progress_interval=progress_interval,
            verbose=verbose
        )
        
        all_indices.append(indices)
        all_weights.append(weights)
        all_timing.append(timing)
        all_sim_timing.append(sim_timing)
    
    # Combine results
    order_mg = np.concatenate(all_indices) if all_indices and len(all_indices[0]) > 0 else np.array([], dtype=np.int32)
    weights_mg = np.concatenate(all_weights) if all_weights and len(all_weights[0]) > 0 else np.array([], dtype=np.float32)
    
    # Get max timing
    ordering_time = np.max(all_timing) if all_timing else 0
    similarity_time = np.max(all_sim_timing) if all_sim_timing else 0
    
    # Legacy output format for compatibility
    order_sz = []
    weights_sz = []
    
    return order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
