import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import gensim.downloader as api

# 単語リストをテキストファイルから読み込む関数
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words

# Word2Vecモデルを読み込む
def load_word2vec_model():
    return api.load("glove-wiki-gigaword-50")  # 50次元のGloVeモデルを使用

# 単語間の意味上の距離行列を計算
def compute_embedding_distance_matrix(words, model):
    size = len(words)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            if words[i] in model and words[j] in model:
                dist_matrix[i, j] = 1 - model.similarity(words[i], words[j])
            else:
                dist_matrix[i, j] = 1  # モデルに存在しない場合の距離を最大値に
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

# クラスタリング結果をファイルに出力する関数
def save_cluster_results(word_list, clusters_5th, clusters_8th, output_file):
    with open(output_file, 'w') as file:
        for word, cluster_5th, cluster_8th in zip(word_list, clusters_5th, clusters_8th):
            file.write(f"{word},{cluster_5th},{cluster_8th}\n")

# ファイルパスの指定
word_list_file = 'word_list.txt'  # ここに単語リストファイルのパスを指定
output_file = 'embedding_distance.txt'  # 出力ファイルのパス

# 単語リストを読み込む
word_list = load_word_list(word_list_file)

# Word2Vecモデルを読み込む
model = load_word2vec_model()

# 意味上の距離行列の計算
dist_matrix = compute_embedding_distance_matrix(word_list, model)

# 階層クラスタリング (ウォード法)
linkage_matrix = linkage(squareform(dist_matrix), method='ward')

# クラスタ数の計算
num_words = len(word_list)
num_clusters_5th = max(1, num_words // 5)
num_clusters_8th = max(1, num_words // 8)

# クラスタリング結果の取得
clusters_5th = fcluster(linkage_matrix, num_clusters_5th, criterion='maxclust')
clusters_8th = fcluster(linkage_matrix, num_clusters_8th, criterion='maxclust')

# クラスタリング結果をファイルに保存
save_cluster_results(word_list, clusters_5th, clusters_8th, output_file)

print(f"クラスタリング結果を '{output_file}' に保存しました。")
