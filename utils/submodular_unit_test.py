import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def pca_based_coreset_selection(gradients, m, features=None, n_components=0.95, candidate_ratio=2, lambda_reg=1e-5):
    """
    基于梯度主成分覆盖性的核心集选择
    :param gradients: 梯度矩阵，形状 (n_samples, grad_dim)
    :param m: 目标核心集大小
    :param features: 用于多样性计算的特征矩阵 (None则用PCA降维后的梯度)
    :param n_components: PCA主成分数量 (0.95表示保留95%方差)
    :param candidate_ratio: 候选集比例
    :param lambda_reg: 核矩阵正则化参数
    :return: 核心集索引
    """
    n_samples, grad_dim = gradients.shape
    candidate_size = min(candidate_ratio * m, n_samples)
    
    # 阶段1: PCA主方向覆盖性筛选
    # 计算梯度PCA主成分
    pca = PCA(n_components=n_components)
    pca.fit(gradients)
    
    # 计算样本在主成分上的投影能量（平方和）
    projected = pca.transform(gradients)          # 投影到主成分空间
    energy = np.sum(projected**2,axis=1)    # 每个样本的投影能量
    
    # 选择能量最高的候选集
    pca_candidates = np.argsort(-energy)[:candidate_size]
    
    # 阶段2: DPP多样性筛选
    # 准备特征矩阵（优先使用外部特征，否则用降维后的梯度）
    if features is None:
        features = pca.transform(gradients)      # 使用PCA降维后的梯度作为特征
    candidate_features = features[pca_candidates]
    
    # 构建多样性核矩阵
    K = cosine_similarity(candidate_features)
    K += lambda_reg * np.eye(K.shape[0])
    
    # 贪心行列式最大化
    selected = []
    remaining = list(range(len(pca_candidates)))
    
    for _ in range(m):
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
    return core_indices

# 示例用法
if __name__ == "__main__":
    # 生成模拟数据：100样本，梯度维度1000
    np.random.seed(42)
    n_samples = 100
    grad_dim = 1000
    features_dim = 50  # 假设外部特征维度
    
    gradients = np.random.randn(n_samples, grad_dim)
    features = np.random.randn(n_samples, features_dim)
    m = 20
    
    # 执行选择
    core_set = pca_based_coreset_selection(gradients, m, features=features)
    print("PCA-based core indices:", core_set)