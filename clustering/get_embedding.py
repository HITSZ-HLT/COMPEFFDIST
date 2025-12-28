from collect_attr import load_json, save_json
import numpy as np
from tqdm import tqdm
import random
import os


def _get_embeddings(f, texts, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = f(batch_texts)
        embeddings.append(batch_embeddings)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def get_text_embedding_sentence_transformer(attr_list, model_name, batch_size=32):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [attr['attr'] for attr in attr_list]

    return _get_embeddings(model.encode, texts, batch_size=batch_size)
    

def get_text_embedding_UAE(attr_list, batch_size=32):
    from angle_emb import AnglE

    model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls')
    texts = [attr['attr'] for attr in attr_list]
    f = lambda x: model.encode(x, normalize_embedding=True)

    return _get_embeddings(f, texts, batch_size=batch_size)


def get_topk_attr(attr_vectors, attr_list, K=10):
    idx_to_attr = {attr['idx']: f"{attr['attr']}|({attr['count']})" for attr in attr_list}
    n_attrs = len(attr_list)

    results = []
    for attr_idx in range(n_attrs):
        # 随机选择一个属性
        attr = idx_to_attr[attr_idx]
        vector = attr_vectors[attr_idx]
        # 计算与所有其他属性的余弦相似度
        similarities = np.dot(attr_vectors, vector) / (np.linalg.norm(attr_vectors, axis=1) * np.linalg.norm(vector))
        # 获取相似度最高的K个属性
        top_k_indices = np.argsort(similarities)[-K-1:-1]
        top_k_attrs = [f'{idx_to_attr[idx]}|{similarities[idx]:.2f}' for idx in top_k_indices]
        top_k_attrs.reverse()
        results.append({'idx': attr_idx, 'attr': attr, 'top_k_attrs': top_k_attrs})
    
    return results


# 二维可视化，使用tsne降维，每个点使用文本表示
def visualize_2d(attr_vectors, attr_list, title, sparsity_rate=0.25):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    ENV_PARAMS = {
        'figure_size': (16, 9),               # 图形大小
        'canvas_bounds': (-10, 10, -10, 10),  # 画布边界 (x_min, x_max, y_min, y_max)
        'dpi': 200,                           # 图像DPI
        'font_size': 8.,                      # 字体大小
        'bbox_style': 'round,pad=0.4',        # 文本框样式
        'bbox_alpha': 0.95,                   # 文本框透明度
    }

    TEXT_PARAMS = {
        'char_width_factor': 0.015,    # 字符宽度系数
        'line_height_factor': 0.03,    # 行高系数
        'width_padding': 0.1,          # 宽度额外填充
        'height_padding': 0.35,        # 高度额外填充
        'overlap_tolerance': 0.025,    # 重叠容差
        'max_radius': 1.3,             # 最大搜索半径
        'direction_count': 16          # 方向数量
    }
    
    # 使用tsne降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    attr_vectors = tsne.fit_transform(attr_vectors) 

    attr_vectors_normalized = (attr_vectors - attr_vectors.min(axis=0)) / (attr_vectors.max(axis=0) - attr_vectors.min(axis=0))
    attr_vectors_normalized = attr_vectors_normalized * 20 - 10  # 缩放到 -10 到 10 的范围

    # 创建一个干净的图，使用全局环境参数
    plt.figure(figsize=ENV_PARAMS['figure_size'])
    x_min, x_max, y_min, y_max = ENV_PARAMS['canvas_bounds']
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.axis('off')  # 完全隐藏坐标轴

    labels = np.arange(len(attr_list))
    # 获取唯一的簇标签并为每个簇分配颜色
    unique_labels = set(labels)
    # 使用更多的颜色，确保每个簇有不同颜色
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 前20个颜色使用tab20
    additional_colors = plt.cm.Set2(np.linspace(0, 1, 8))  # 额外颜色使用Set2
    all_colors = np.vstack([colors, additional_colors])
    
    # 打乱颜色顺序
    np.random.seed(42)
    random.seed(42)
    np.random.shuffle(all_colors)

    # 为每个簇分配颜色（注意颜色数量小于标签数量）
    for i in unique_labels:
        # 只显示一部分的点
        if random.random() < 1-sparsity_rate:
            continue

        color = all_colors[i % len(all_colors)]
        x, y = attr_vectors_normalized[i]
        text = attr_list[i]['attr']
        
        text_box_style = dict(
            facecolor='white', 
            alpha=ENV_PARAMS['bbox_alpha'], 
            boxstyle=ENV_PARAMS['bbox_style'], 
            edgecolor=color, 
            linewidth=1.2
        )

        plt.text(x, y, text,
                 fontsize=ENV_PARAMS['font_size'],
                 verticalalignment='center',
                 horizontalalignment='center',
                 bbox=text_box_style,
                 zorder=5  # 确保文本在最上层
        )
        
    plt.tight_layout(pad=0.5)
    plt.show()



if __name__ == "__main__":
    attr_list = load_json('./data/attr_list.json')
    print(len(attr_list))
    

    embeddings = get_text_embedding_UAE(attr_list)
    topk_attr = get_topk_attr(embeddings, attr_list)
    save_json(topk_attr, 'data/topk_attr.json')

    visualize_2d(embeddings, attr_list, 'text_embedding', sparsity_rate=0.25)