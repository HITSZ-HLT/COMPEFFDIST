import os
import time
import numpy as np
# 设置CPU核心数，消除警告
os.environ['LOKY_MAX_CPU_COUNT'] = '16'  # 根据您的CPU核心数调整这个值

from collect_attr import load_json, save_json
from get_embedding import get_text_embedding_UAE


def _parsing_cluster_result(labels, attr_list):
    clusters = [] # 每个元素是一个字典，包含cluster_id, attr_names, total_count
    for i in range(max(labels)+1):
        clusters.append({
            'cluster_id': i,
            'attr_names': [],
            'total_count': 0,
        })
    for i in range(len(labels)):
        attr_name = attr_list[i]['attr']
        attr_count = attr_list[i]['count']
        clusters[labels[i]]['attr_names'].append(f'{attr_name}|({attr_count})')
        clusters[labels[i]]['total_count'] += attr_count
    
    for cluster in clusters:
        cluster['attr_names'].sort(key=lambda x: int(x.split('|')[1].strip('()')), reverse=True)
    clusters.sort(key=lambda x: x['total_count'], reverse=True)
    
    return clusters


def kmeans_clustering(embeddings, attr_list, k=100, weight=0.3):
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42, verbose=1)
    kmeans.fit(embeddings, sample_weight=[attr['count']**weight for attr in attr_list])
    labels = kmeans.labels_
    
    return _parsing_cluster_result(labels, attr_list)


# AffinityPropagation
def affinity_propagation_clustering(embeddings, attr_list, percentile_rate=0.5):
    # percentile_rate值越高 → 更多聚类中心（更多簇）

    from sklearn.cluster import AffinityPropagation
    from sklearn.metrics import pairwise_distances

    similarity_matrix = -pairwise_distances(embeddings, metric='euclidean')**2
    preference = np.percentile(similarity_matrix, percentile_rate)

    affinity_propagation = AffinityPropagation(
        damping=0.9, 
        max_iter=1000, 
        random_state=42, 
        verbose=1, 
        preference=preference
    )
    # 考虑每个属性的权重，通过修改每个属性对应的embeddings的次数来实现
    import math
    counts = [int(math.log1p(attr['count'])) for attr in attr_list] 
    extra_embeddings = []
    for i in range(len(attr_list)):
        extra_embeddings.extend(embeddings[i] for _ in range(counts[i]))
    extra_embeddings = np.array(extra_embeddings)
    print(extra_embeddings.shape, sum(counts))
    embeddings = np.concatenate([embeddings, extra_embeddings], axis=0)

    affinity_propagation.fit(embeddings)
    labels = affinity_propagation.labels_[:len(attr_list)]
    
    return _parsing_cluster_result(labels, attr_list)


# MeanShift
def mean_shift_clustering(embeddings, attr_list, bandwidth=0.5):
    from sklearn.cluster import MeanShift

    mean_shift = MeanShift(bandwidth=bandwidth)
    mean_shift.fit(embeddings)
    labels = mean_shift.labels_

    return _parsing_cluster_result(labels, attr_list)


# AgglomerativeClustering
def agglomerative_clustering(embeddings, attr_list, n_clusters=200):
    from sklearn.cluster import AgglomerativeClustering

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(embeddings)
    labels = agglomerative.labels_  

    return _parsing_cluster_result(labels, attr_list)

if __name__ == "__main__":
    attr_list = load_json('data/attr_list.json')

    if not os.path.exists('data/clustering/embeddings.npy'):
        embeddings = get_text_embedding_UAE(attr_list)
        np.save('data/clustering/embeddings.npy', embeddings)
    else:
        embeddings = np.load('data/clustering/embeddings.npy')

    # clusters = kmeans_clustering(embeddings, attr_list, k=200)
    # file_name = 'data/clustering/clusters.json'

    clusters = affinity_propagation_clustering(embeddings, attr_list, percentile_rate=0.1)
    file_name = 'data/clustering/affinity_propagation_clusters_p=0.1.json'

    # clusters = mean_shift_clustering(embeddings, attr_list, bandwidth=0.5)
    # file_name = 'data/clustering/mean_shift_clusters.json'

    # clusters = agglomerative_clustering(embeddings, attr_list, n_clusters=200)
    # file_name = 'data/clustering/agglomerative_clustering_n=200.json'

    print('cluster_num:', len(clusters))
    # 如果文件存在，则修改文件名称
    if os.path.exists(file_name):
        file_name = file_name.replace('.json', f'_{time.strftime("%Y%m%d_%H%M%S")}.json')
    save_json(clusters, file_name)
