import time

import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction


def faciliy_location_order(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]
    print("len(class_indices)", len(class_indices))
    print("class_indices", class_indices)
    print("len(y)", len(y))
    print("c", c)

    if mode == "dense":
        num_n = None

    start = time.time()
    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz == 0)] = 1

    return class_indices[order], sz, greedy_time, S_time


def faciliy_location_order_detrimental(c, X, y, metric, num_per_class, weights=None, optimizer="LazyGreedy"):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    start = time.time()
    obj = FacilityLocationFunction(n=len(X), data=X, metric=metric, mode='dense')
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer=optimizer,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))  # same as order = [x[0] for x in greedyList]
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)
    cluster = -np.ones(N)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        cluster[i] = np.argmax(S[i, order])
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz==0)] = 1

    cluster[cluster>=0] += c * num_per_class

    return class_indices[order], sz, greedy_time, S_time, cluster


def faciliy_location_order_sim_panel(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None

    # Initialize the facility location objective
    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    S = obj.sijs  # Similarity matrix or graph

    # Greedy selection with diversity consideration
    selected_indices = []
    selected_scores = []
    available_indices = set(range(N))
    
    for _ in range(num_per_class):
        max_gain = -np.inf
        best_index = -1

        for i in available_indices:
            # Compute diversity penalty
            diversity_penalty = (
                np.max(S[i, selected_indices]) if selected_indices else 0
            )

            # Compute adjusted gain (similarity - diversity)
            gain = S[i, :].sum() - diversity_penalty

            if gain > max_gain:
                max_gain = gain
                best_index = i

        # Add the best index to the selected subset
        selected_indices.append(best_index)
        selected_scores.append(max_gain)
        available_indices.remove(best_index)

    # Compute cluster sizes (sz)
    sz = np.zeros(num_per_class, dtype=np.float64)
    for i in range(N):
        if np.max(S[i, selected_indices]) <= 0:
            continue
        if weights is None:
            sz[np.argmax(S[i, selected_indices])] += 1
        else:
            sz[np.argmax(S[i, selected_indices])] += weights[i]
    sz[np.where(sz == 0)] = 1

    return class_indices[selected_indices], sz, 0, 0  # Timing placeholders


def get_orders_and_weights(
    B,
    X,
    metric,
    y=None,
    weights=None,
    equal_num=False,
    outdir=".",
    mode="sparse",
    num_n=128,
):
    """
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist
    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    """

    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    classes = classes.astype(np.int32).tolist()
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(
            np.ceil(np.divide([sum(y == i) for i in classes], N) * B)
        )

    print(f"Greedy: selecting {num_per_class} elements")

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(
        *map(
            lambda c: faciliy_location_order(
                c[1], X, y, metric, num_per_class[c[0]], weights, mode, num_n
            ),
            enumerate(classes),
        )
    )
    print(
        f"time (sec) for computing facility location: {greedy_times} similarity time {similarity_times}",
    )

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))  # TODO

    order_mg_all = np.array(order_mg_all, dtype=object)
    cluster_sizes_all = np.array(cluster_sizes_all, dtype=object)

    for i in range(
        int(
            np.rint(
                np.max([len(order_mg_all[c]) / props[c] for c, _ in enumerate(classes)])
            )
        )
    ):
        for c, _ in enumerate(classes):
            ndx = slice(
                i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c]))
            )
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])

    order_mg = np.array(order_mg, dtype=np.int32)

    weights_mg = np.array(
        weights_mg, dtype=np.float32
    )  # / sum(weights_mg) TODO: removed division!
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = (
        []
    )  # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals



def get_orders_and_weights_detrimental(B, X, metric, y=None, weights=None, equal_num=False, num_classes=10, optimizer="LazyGreedy"):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    # print("in detrimental")
    # print('B', B)
    # print('X', X)
    # print('metric', metric)
    # print('y', y)
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        total = np.sum(num_per_class)
        diff = total - B
        chosen = set()
        for i in range(diff):
            j = np.random.randint(C)
            while j in chosen or num_per_class[j] <= 0:
                j = np.random.randint(C)
            num_per_class[j] -= 1
            chosen.add(j)

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times, cluster_all = zip(*map(
        lambda c: faciliy_location_order_detrimental(c, X, y, metric, num_per_class[c], weights, optimizer=optimizer), classes))

    order_mg = np.concatenate(order_mg_all).astype(np.int32)
    weights_mg = np.concatenate(cluster_sizes_all).astype(np.float32)
    class_indices = [np.where(y == c)[0] for c in classes]
    class_indices = np.concatenate(class_indices).astype(np.int32)
    class_indices = np.argsort(class_indices)
    cluster_mg = np.concatenate(cluster_all).astype(np.int32)[class_indices]
    assert len(order_mg) == len(weights_mg)

    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time, cluster_mg
    return vals


def get_orders_and_weights_hybrid(
    B,
    X,
    metric,
    y=None,
    weights=None,
    equal_num=False,
    dpp_weight=0.3,  # Weight for DPP diversity (0 = pure FL, 1 = pure DPP)
    mode="sparse",
    num_n=128,
):
    """
    Hybrid facility location + DPP selection for both coverage and diversity.
    
    Args:
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
    - dpp_weight: float in [0,1], weight for diversity vs coverage
    - mode: str, one of ['sparse', 'dense'] for computation mode
    
    Returns:
    - order_mg: np.array, shape [B], type int64 - selected indices
    - weights_mg: np.array, shape [B], type float32, sums to 1 - weights
    """
    import numpy as np
    try:
        from submodlib.functions.facilityLocation import FacilityLocationFunction
        from submodlib.functions.logDeterminant import LogDeterminantFunction
        has_submodlib = True
    except ImportError:
        has_submodlib = False
        print("Warning: submodlib not found, falling back to numpy implementation")
    
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    
    classes = np.unique(y)
    classes = classes.astype(np.int32).tolist()
    C = len(classes)  # number of classes

    # Determine number of points to select per class
    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = np.array([class_nums[c] < np.ceil(B / C) for c in classes])
        
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c_idx, c in enumerate(classes):
                if not minority[c_idx]:
                    num_per_class[c_idx] += int(np.ceil(extra / sum(~minority)))
    else:
        num_per_class = np.int32(
            np.ceil(np.divide([sum(y == i) for i in classes], N) * B)
        )

    print(f"Hybrid selection: selecting {num_per_class} elements per class")
    
    # Define function to process each class in parallel
    def process_class(class_data):
        c_idx, c = class_data
        class_indices = np.where(y == c)[0]
        target_count = num_per_class[c_idx]
        
        if len(class_indices) == 0 or target_count == 0:
            return [], [], 0, 0
            
        class_features = X[class_indices]
        
        # Prepare for similarity computation - normalize if using cosine
        if metric == "cosine":
            norms = np.linalg.norm(class_features, axis=1, keepdims=True)
            class_features = class_features / (norms + 1e-8)
        
        # Time similarity computation
        sim_start = time.time()
        
        # Determine if we should use dense or sparse mode based on dataset size
        use_dense = mode == "dense" or len(class_indices) <= 10000
        
        # Compute similarity matrix if using dense mode or numpy fallback
        if use_dense or not has_submodlib:
            if metric == "cosine":
                similarity_matrix = np.dot(class_features, class_features.T)
            else:  # euclidean
                similarity_matrix = -np.sum((class_features[:, None, :] - 
                                         class_features[None, :, :]) ** 2, axis=2)
        
        sim_time = time.time() - sim_start
        
        # Time greedy selection
        greedy_start = time.time()
        
        if has_submodlib:
            # Using submodlib's implementation for efficiency
            # Determine which mode to use
            actual_mode = "dense" if use_dense else "sparse"
            
            # Configure parameters based on mode
            fl_params = {
                "n": len(class_features),
                "mode": actual_mode
            }
            
            if actual_mode == "dense":
                fl_params["sijs"] = similarity_matrix
                fl_params["separate_rep"] = False
            else:  # sparse mode
                fl_params["data"] = class_features
                fl_params["metric"] = metric
                fl_params["num_neighbors"] = num_n
                
            # Create facility location object with appropriate parameters
            fl_obj = FacilityLocationFunction(**fl_params)
            
            # For DPP (diversity)
            if dpp_weight > 0:
                dpp_params = {
                    "n": len(class_features),
                    "mode": actual_mode
                }
                
                if actual_mode == "dense":
                    dpp_params["sijs"] = similarity_matrix
                    dpp_params["lambdaVal"] = 1.0
                else:  # sparse mode
                    dpp_params["data"] = class_features
                    dpp_params["metric"] = metric
                    
                dpp_obj = LogDeterminantFunction(**dpp_params)
            
            # Custom hybrid greedy selection that combines FL and DPP objectives
            if dpp_weight > 0:
                # Use optimized batch implementation directly
                print(f"Using optimized batch implementation instead of MixtureFunction")
                
                # Batch optimization: evaluate gains in groups to reduce redundant calculations
                selected = set()
                ground_set = list(range(len(class_features)))
                remaining = ground_set.copy()
                
                # For each selection round
                for _ in range(min(target_count, len(class_features))):
                    if not remaining:
                        break
                    
                    batch_size = min(len(remaining), 128)  # Process in batches of 128
                    best_idx = -1
                    best_gain = float('-inf')
                        
                    # Process remaining candidates in batches
                    for i in range(0, len(remaining), batch_size):
                        batch = remaining[i:i+batch_size]
                        
                        # Prepare a single selected set for reuse in this batch
                        selected_set = set(selected)
                        
                        # Compute FL gains for batch
                        fl_batch_gains = {}
                        for idx in batch:
                            fl_batch_gains[idx] = fl_obj.marginalGain(selected_set, idx)
                        
                        # Compute DPP gains only if needed and only for promising candidates
                        if dpp_weight > 0:
                            # Sort indices by FL gain to prioritize promising candidates
                            sorted_indices = sorted(fl_batch_gains.keys(), 
                                                    key=lambda k: fl_batch_gains[k], 
                                                    reverse=True)
                            
                            # Only evaluate top candidates with DPP (faster)
                            top_k = min(len(sorted_indices), batch_size // 2)
                            for idx in sorted_indices[:top_k]:
                                dpp_gain = dpp_obj.marginalGain(selected_set, idx)
                                # Compute combined gain
                                combined_gain = (1 - dpp_weight) * fl_batch_gains[idx] + dpp_weight * dpp_gain
                                
                                if combined_gain > best_gain:
                                    best_gain = combined_gain
                                    best_idx = idx
                        else:
                            # Just find the best FL gain
                            batch_best_idx = max(fl_batch_gains, key=fl_batch_gains.get)
                            batch_best_gain = fl_batch_gains[batch_best_idx]
                            
                            if batch_best_gain > best_gain:
                                best_gain = batch_best_gain
                                best_idx = batch_best_idx
                    
                        if best_idx >= 0:
                            selected.add(best_idx)
                            remaining.remove(best_idx)
                        else:
                            break
            else:
                # Pure facility location - use built-in maximize which is highly optimized
                greedyList = fl_obj.maximize(
                    budget=target_count,
                    optimizer="LazyGreedy",  # LazyGreedy is faster than NaiveGreedy
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    verbose=False
                )
                # Extract selected elements from the greedy list
                selected = set(x[0] for x in greedyList)  # Convert to set for consistency
            
            # Compute cluster sizes (weights) - vectorized calculation for efficiency
            selected_list = list(selected) if isinstance(selected, set) else selected  # Convert to list for array indexing
            if use_dense:
                # Use vectorized operations with the precomputed similarity matrix
                if len(selected_list) > 0:
                    # Get similarity of all points to selected points
                    sim_to_selected = similarity_matrix[:, selected_list]
                    
                    # For each point, find most similar selected point
                    closest_selected = np.argmax(sim_to_selected, axis=1)
                    
                    # Count points belonging to each cluster (advanced indexing)
                    if weights is None:
                        # Simple count
                        cluster_sizes = np.zeros(len(selected_list))
                        for i in range(len(class_features)):
                            if np.max(sim_to_selected[i]) > 0:  # Only count if similarity > 0
                                cluster_sizes[closest_selected[i]] += 1
                    else:
                        # Weighted count
                        cluster_sizes = np.zeros(len(selected_list))
                        class_weights = weights[class_indices] if weights is not None else np.ones(len(class_indices))
                        for i in range(len(class_features)):
                            if np.max(sim_to_selected[i]) > 0:  # Only count if similarity > 0
                                cluster_sizes[closest_selected[i]] += class_weights[i]
                else:
                    cluster_sizes = np.array([])
            else:
                # For sparse mode, compute similarities only for selected points
                if len(selected_list) > 0:
                    sel_features = class_features[selected_list]
                    
                    # Compute similarity matrix efficiently
                    if metric == "cosine":
                        sel_sim = np.dot(sel_features, class_features.T)
                    else:  # euclidean
                        sel_sim = -np.sum((sel_features[:, None, :] - 
                                      class_features[None, :, :]) ** 2, axis=2)
                    
                    # Compute cluster sizes
                    closest_selected = np.argmax(sel_sim, axis=0)
                    cluster_sizes = np.zeros(len(selected_list))
                    
                    class_weights = weights[class_indices] if weights is not None else np.ones(len(class_indices))
                    
                    # Vectorized counting for better performance
                    for i in range(len(class_features)):
                        if np.max(sel_sim[:, i]) > 0:  # Only count if similarity > 0
                            cluster_sizes[closest_selected[i]] += class_weights[i] if weights is not None else 1
                else:
                    cluster_sizes = np.array([])
                
            # Ensure no zero weights
            cluster_sizes[cluster_sizes == 0] = 1
                
        else:
            # Numpy-based implementation (fallback)
            selected = set()  # Initialize as a set instead of a list
            remaining = list(range(len(class_features)))
            
            for _ in range(min(target_count, len(class_features))):
                if not remaining:
                    break
                
                selected_list = list(selected)  # Convert to list for array indexing
                if len(selected) == 0:
                    # First element - pure coverage
                    fl_gains = np.sum(similarity_matrix[remaining], axis=1)
                    best_idx = remaining[np.argmax(fl_gains)]
                else:
                    best_gain = -float('inf')
                    best_idx = -1
                    
                    for idx in remaining:
                        # Calculate marginal gain for facility location
                        current_values = np.zeros(len(class_features))
                        for j in selected_list:
                            current_values = np.maximum(current_values, similarity_matrix[j])
                            
                        candidate_values = np.maximum(current_values, similarity_matrix[idx])
                        fl_gain = np.sum(candidate_values) - np.sum(current_values)
                        
                        # Mix in diversity if needed
                        if dpp_weight > 0:
                            # Simple diversity penalty based on similarity to selected points
                            if len(selected) > 0:
                                sim_to_selected = np.mean(similarity_matrix[idx, selected_list])
                                diversity_term = 1 - sim_to_selected  # Higher value = more diverse
                            else:
                                diversity_term = 1.0
                                
                            # Combined gain
                            gain = (1 - dpp_weight) * fl_gain + dpp_weight * diversity_term
                        else:
                            gain = fl_gain
                            
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = idx
                            
                selected.add(best_idx)  # Use add() for sets instead of append()
                remaining.remove(best_idx)
            
            # Calculate weights
            selected_list = list(selected)  # Convert to list for array indexing
            cluster_sizes = np.sum(similarity_matrix[selected_list], axis=1) if selected else np.array([])
                
        greedy_time = time.time() - greedy_start
        
        # Map local indices back to global indices
        selected_list = list(selected) if isinstance(selected, set) else selected  # Convert to list for array indexing
        global_indices = class_indices[selected_list]
        
        return global_indices, cluster_sizes, greedy_time, sim_time
    
    # Process each class in parallel with map
    class_data = [(c_idx, c) for c_idx, c in enumerate(classes)]
    all_results = list(map(process_class, class_data))
    
    # Unpack results
    all_indices, all_weights, greedy_times, similarity_times = zip(*all_results)
    
    # More efficient merging using np.concatenate
    order_mg = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    weights_mg = np.concatenate(all_weights) if all_weights else np.array([], dtype=np.float32)
    
    # Normalize weights if we have any
    if len(weights_mg) > 0:
        weights_mg = weights_mg / np.sum(weights_mg)
    
    # Use max timing
    ordering_time = np.max(greedy_times) if greedy_times else 0
    similarity_time = np.max(similarity_times) if similarity_times else 0
    
    # Legacy output format for compatibility
    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals


def greedy_merge(X, y, B, part_num, metric):  
    '''
    preds, fl_labels, B, 5, "euclidean",

    fl_labels indicate largest gain first
    '''
    N = len(X)
    indices = list(range(N))
    part_size = int(np.ceil(N / part_num))
    part_indices = [
        indices[slice(i * part_size, min((i + 1) * part_size, N))]
        for i in range(part_num)
    ]
    print(f"GreeDi with {part_num} parts, finding {B} elements...", flush=True)

    order_mg_all, cluster_sizes_all, _, _, ordering_time, similarity_time = zip(
        *map(
            lambda p: get_orders_and_weights(
                int(B / 2), X[part_indices[p], :], metric, y=y[part_indices[p]],
            ),
            np.arange(part_num),
        )
    )

    order_mg_all = list(order_mg_all)
    order_mg = np.concatenate(order_mg_all, dtype=np.int32)
    weights_mg = np.concatenate(cluster_sizes_all, dtype=np.float32)
    print(
        f"GreeDi stage 1: found {len(order_mg)} elements in: {np.max(ordering_time)} sec",
    )

    (
        order,
        weights,
        order_sz,
        weights_sz,
        ordering_time_merge,
        similarity_time_merge,
    ) = get_orders_and_weights(
        B, X[order_mg, :], metric, y=y[order_mg], weights=weights_mg,
    )

    total_ordering_time = np.max(ordering_time) + ordering_time_merge
    total_similarity_time = np.max(similarity_time) + similarity_time_merge
    print(
        f"GreeDi stage 2: found {len(order)} elements in: {total_ordering_time + total_similarity_time} sec",
    )
    vals = (
        order,
        weights,
        order_sz,
        weights_sz,
        total_ordering_time,
        total_similarity_time,
    )
    return vals
