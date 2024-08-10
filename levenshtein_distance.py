import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from Levenshtein import distance as levenshtein_distance

# 単語リストをテキストファイルから読み込む関数
def load_word_list(file_path):
    with open(file_path, 'r') as file:
        words = file.read().splitlines()
    return words

# レーベンシュタイン距離行列を計算
def compute_distance_matrix(words):
    size = len(words)
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist_matrix[i, j] = levenshtein_distance(words[i], words[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

# クラスタリング結果をファイルに出力する関数
def save_cluster_results(word_list, clusters_5th, clusters_8th, output_file):
    with open(output_file, 'w') as file:
        for word, cluster_5th, cluster_8th in zip(word_list, clusters_5th, clusters_8th):
            file.write(f"{word},{cluster_5th},{cluster_8th}\n")

# ファイルパスの指定
word_list_file = 'word_list.txt'  # ここに単語リストファイルのパスを指定
output_file = 'levenshtein_output.txt'  # 出力ファイルのパス

# 単語リストを読み込む
word_list = load_word_list(word_list_file)

# 距離行列の計算
dist_matrix = compute_distance_matrix(word_list)

# 階層クラスタリング (ウォード法)
linkage_matrix = linkage(dist_matrix, method='ward')

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
