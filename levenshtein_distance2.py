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
def save_cluster_results(word_list, clusters_6, clusters_9, output_file):
    with open(output_file, 'w') as file:
        for word, cluster_6, cluster_9 in zip(word_list, clusters_6, clusters_9):
            file.write(f"{word},{cluster_6},{cluster_9}\n")

# クラスタリングの停止条件をチェック
def check_cluster_size(linkage_matrix, word_list, max_size):
    for num_clusters in range(2, len(word_list) + 1):
        clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        cluster_sizes = [list(clusters).count(i) for i in set(clusters)]
        if max(cluster_sizes) >= max_size:
            return clusters

# ファイルパスの指定
word_list_file = 'word_list.txt'  # ここに単語リストファイルのパスを指定
output_file = 'levenshtein_output2.txt'  # 出力ファイルのパス

# 単語リストを読み込む
word_list = load_word_list(word_list_file)

# 距離行列の計算
dist_matrix = compute_distance_matrix(word_list)

# 階層クラスタリング (ウォード法)
linkage_matrix = linkage(dist_matrix, method='ward')

# クラスタサイズの最大値が6になる直前のクラスタ番号を取得
clusters_6 = check_cluster_size(linkage_matrix, word_list, max_size=6)

# クラスタサイズの最大値が9になる直前のクラスタ番号を取得
clusters_9 = check_cluster_size(linkage_matrix, word_list, max_size=9)

# クラスタリング結果をファイルに保存
save_cluster_results(word_list, clusters_6, clusters_9, output_file)

print(f"クラスタリング結果を '{output_file}' に保存しました。")
