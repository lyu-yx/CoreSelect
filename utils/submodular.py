import time

import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.functions.disparitySum import DisparitySumFunction
from submodlib.functions.disparityMin import DisparityMinFunction
from submodlib.functions.setCover import SetCoverFunction
from submodlib.functions.logDeterminant import LogDeterminantFunction
from submodlib.functions.graphCut import GraphCutFunction
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def facility_location_order(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

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
    sz = list(map(lambda x: x[1], greedyList)) #weight of sample
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
def facility_location_order_dpp(c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None
    start = time.time()
    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order_cov = list(map(lambda x: x[0], greedyList))
    lambda_reg = 1e-5
    pca = PCA(n_components=0.95)
    pca.fit(X)
    projected = pca.transform(X)          # 投影到主成分空间
    energy = np.sum(projected**2,axis=1)    # 每个样本的投影能量
    pca_candidates = np.argsort(-energy)[:int(1.5*num_per_class)]
    candidate_features = X[pca_candidates]
    
    # 构建多样性核矩阵
    K = cosine_similarity(candidate_features)
    K += lambda_reg * np.eye(K.shape[0])
    
    # 贪心行列式最大化
    selected = []
    remaining = list(range(len(pca_candidates)))
    
    for _ in range(num_per_class):
        max_score = -np.inf
        best_idx = -1
        for idx in remaining:
            current_subset = selected + [idx]
            sub_K = K[np.ix_(current_subset, current_subset)]
            score = np.log(np.linalg.det(sub_K))
            if score > max_score:
                max_score = score
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    # 映射回原始索引
    core_indices = pca_candidates[selected]
    greedy_time = time.time() - start
    S_time = time.time() - start
    order = np.asarray(core_indices, dtype=np.int64)
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

    return class_indices[order],sz,greedy_time,S_time
def normalize(scores):
    if(np.max(scores)/sum(scores) >= 0.5):
        max_idx = np.argmax(scores)
        scores_cp = scores
        scores_cp = np.delete(scores_cp,max_idx)
        min_val, max_val = np.min(scores_cp),  np.max(scores_cp)
        scores = (scores - min_val) / (max_val - min_val + 1e-8)
        scores[max_idx] = 1 
        return scores
    else:
        min_val, max_val = np.min(scores), np.max(scores)
        return (scores - min_val) / (max_val - min_val + 1e-8)
        
def rerank(vec,N):
    i=0
    sum = 0
    rank = np.zeros(N)
    for elem in vec:
        rank[elem] = i
        i+=1
        sum+=elem
    rank[np.sum(np.arange(N))-sum] = 49
    return rank
        
def facility_location_order_div_panel(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    flag=0         #      0----sz      1----rank    2----pure div
    alpha = 0.9
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None
    #----------------Cover----------------- 
    #start = time.time()
    
    obj_cov = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    
    
   # S_time = time.time() - start
    
    start = time.time()
    greedyList_cov = obj_cov.maximize(
        budget=N-1,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order_cov = list(map(lambda x: x[0], greedyList_cov))
    sz_cov = list(map(lambda x: x[1], greedyList_cov)) #weight of sample
    greedy_time = time.time() - start
    

    #-----------Diversity------------
    start = time.time()
    obj_div = DisparitySumFunction(n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n)
    S_time = time.time() - start
    start = time.time()
    greedyList_div = obj_div.maximize(
        budget=N-1,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order_div = list(map(lambda x:x[0], greedyList_div))
    sz_div = list(map(lambda x:x[1],greedyList_div))
    sz_div.reverse()
    greedy_time = time.time() - start
    if(flag == 0):     # TODO:直接正则化效果不佳，两种排序的sz的比例不一致，比如cov中第一个sz会很大，而div中相对平均
        #-----------Norm & sort-----------
        cov_scores = np.zeros(N)
        div_scores = np.zeros(N)
        for idx, score in zip(order_cov, sz_cov):
            cov_scores[idx] = score
        for idx, score in zip(order_div, sz_div):
            div_scores[idx] = score
        cov_norm = normalize(cov_scores)
        div_norm = normalize(div_scores)
        final_score = alpha*cov_norm + (1-alpha)*div_norm
        sort_indices = np.argsort(final_score)[::-1]
        order = sort_indices[:num_per_class]
    elif(flag == 1):
       #----------Sort-------------
        cov_rank = rerank(order_cov,N)
        div_rank = rerank(order_div,N)
        rank = np.floor((cov_rank + div_rank)/2)
        sort_indices = np.argsort(rank)
        order = sort_indices[:num_per_class]
    elif(flag == 2):
        order = order_div 
    S = obj_div.sijs
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

def faciliy_location_order_sim_panel(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None
    #----------------Cover----------------- 
    #start = time.time()
    
    obj_cov = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    
    
   # S_time = time.time() - start
    
    start = time.time()
    num_first_stage = np.ceil(min(1.5*num_per_class,N-1)).astype(int)
    num_second_stage = np.ceil(min(1.2*num_per_class,N-1)).astype(int)
    greedyList_cov = obj_cov.maximize(
        budget=num_first_stage,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order_cov = list(map(lambda x: x[0], greedyList_cov))
    sz_cov = list(map(lambda x: x[1], greedyList_cov)) #weight of sample
    greedy_time = time.time() - start
    

    #-----------Diversity------------
    start = time.time()   # TODO:不在cov的基础上直接div排序，要考虑cov中不同元素的weight
    obj_div = DisparitySumFunction(n=num_first_stage, mode=mode, data=X[order_cov], metric=metric, num_neighbors=num_n)
    S_time = time.time() - start
    start = time.time()
    greedyList_div = obj_div.maximize(
        budget=num_per_class,
        optimizer="NaiveGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order_div = list(map(lambda x:x[0], greedyList_div))
    sz_div = list(map(lambda x:x[1],greedyList_div))
    sz_div.reverse()
    greedy_time = time.time() - start
    #--------------second cov----------------
    # obj_cov_2 = FacilityLocationFunction(
    #     n=num_second_stage, mode=mode, data=X[[order_cov[i] for i in order_div]], metric=metric, num_neighbors=num_n
    # )
    # greedyList_cov_2 = obj_cov_2.maximize(
    #     budget=num_per_class,
    #     optimizer="LazyGreedy",
    #     stopIfZeroGain=False,
    #     stopIfNegativeGain=False,
    #     verbose=False,
    # )
    # order_cov_2 = list(map(lambda x: x[0], greedyList_cov_2))
    # sz_cov_2 = list(map(lambda x: x[1], greedyList_cov_2)) #weight of sample
    # order_tem = [order_div[i] for i in order_cov_2]
    # order = [order_cov[j] for j in order_tem]
    #-----------Norm & sort-----------
    cov_scores = np.zeros(N)
    div_scores = np.zeros(N)
    for idx, score in zip(order_cov, sz_cov):
        cov_scores[idx] = score
    for idx, score in zip(order_div, sz_div):
        div_scores[order_cov[idx]] = score
    cov_norm = normalize(cov_scores)
    div_norm = normalize(div_scores)
    final_score = 0.9*cov_norm + 0.1*div_norm
    order = np.argsort(final_score)[::-1]
    #order=[order_cov[i] for i in order_div]
    order = order[:num_per_class]
    S = obj_cov.sijs
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

def faciliy_location_order_det(    
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None

    start = time.time()
    lambda_val = 1e-5
    obj_cov = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )
    obj_det = LogDeterminantFunction(
        n=len(X), mode=mode,lambdaVal= lambda_val,  data=X, metric=metric, num_neighbors=num_n
    )
    S_time = time.time() - start

    start = time.time()
    greedyList_det = obj_det.maximize(
        budget=N-1,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    greedyList_cov = obj_cov.maximize(
        budget=N-1,
        optimizer="LazyGreedy",
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    alpha = 0.5
    order_det = list(map(lambda x: x[0], greedyList_det))
    order_cov = list(map(lambda x: x[0], greedyList_cov))
    sz_det =  list(map(lambda x: x[1], greedyList_det))
    sz_cov = list(map(lambda x: x[1], greedyList_cov))
    greedy_time = time.time() - start
    cov_scores = np.zeros(N)
    det_scores = np.zeros(N)
    for idx, score in zip(order_cov, sz_cov):
        cov_scores[idx] = score
    for idx, score in zip(order_det, sz_det):
        det_scores[idx] = score
    cov_norm = normalize(cov_scores)
    div_norm = normalize(det_scores)
    final_score = alpha*cov_norm + (1-alpha)*div_norm
    sort_indices = np.argsort(final_score)[::-1]
    order = sort_indices[:num_per_class]
   # cov_rank = rerank(order_cov,N)
   #det_rank = rerank(order_det,N)
    #rank = np.floor((cov_rank + det_rank)/2)
   # sort_indices = np.argsort(rank)
   # order = sort_indices[:num_per_class]

    S = obj_cov.sijs
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
def faciliy_location_order_graphcut(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    if mode == "dense":
        num_n = None

    start = time.time()
    obj = GraphCutFunction(
        n=len(X), mode=mode, data=X, lambdaVal=0.9,metric=metric, num_neighbors=num_n
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
    sz = list(map(lambda x: x[1], greedyList)) #weight of sample
    greedy_time = time.time() - start
    S = obj.ggsijs
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
            lambda c: facility_location_order_dpp(
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
