import sys
import math
from collections import defaultdict
# import matplotlib.pyplot as plt

def load_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            object_id, x, y = line.strip().split('\t')
            data.append((object_id, float(x), float(y)))
    return data

def euclidean_distance(point1, point2):
    return math.sqrt((point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

def range_query(data, point, eps):
    neighbors = []
    for other_point in data:
        if euclidean_distance(point, other_point) <= eps:
            neighbors.append(other_point)
    return neighbors

def expand_cluster(data, labels, point, neighbors, cluster_id, eps, min_pts):
    labels[point[0]] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor = neighbors[i]
        if neighbor[0] not in labels:
            labels[neighbor[0]] = cluster_id
            new_neighbors = range_query(data, neighbor, eps)
            if len(new_neighbors) >= min_pts:
                neighbors.extend(new_neighbors)
        elif labels[neighbor[0]] == -1:
            labels[neighbor[0]] = cluster_id
        i += 1

def dbscan(data, eps, min_pts):
    labels = {}
    cluster_id = 0
    for point in data:
        if point[0] not in labels:
            neighbors = range_query(data, point, eps)
            if len(neighbors) < min_pts:
                labels[point[0]] = -1
            else:
                expand_cluster(data, labels, point, neighbors, cluster_id, eps, min_pts)
                cluster_id += 1
    return labels

def save_clusters(data, labels, num_clusters, file_name_prefix):
    clusters = defaultdict(list)
    for point in data:
        if labels[point[0]] != -1:
            clusters[labels[point[0]]].append(point[0])
    
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)[:num_clusters]
    
    for i, cluster in enumerate(sorted_clusters):
        with open(f"{file_name_prefix}_cluster_{i}.txt", 'w') as file:
            for object_id in cluster:
                file.write(f"{object_id}\n")

# def plot_clusters(data, labels, num_clusters):
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     for i in range(num_clusters):
#         cluster_data = [point for point in data if labels[point[0]] == i]
#         if cluster_data:
#             x_coords = [point[1] for point in cluster_data]
#             y_coords = [point[2] for point in cluster_data]
#             plt.scatter(x_coords, y_coords, c=colors[i % len(colors)], s=5, label=f"Cluster {i}")
    
#     noise_data = [point for point in data if labels[point[0]] == -1]
#     if noise_data:
#         x_coords = [point[1] for point in noise_data]
#         y_coords = [point[2] for point in noise_data]
#         plt.scatter(x_coords, y_coords, c='k', s=5, label='Noise')

#     plt.legend()
#     plt.show()

def main():
    if len(sys.argv) != 5:
        print("Usage: python 2021092379_성준모_hw3.py <input_file> <n> <eps> <min_pts>")
        return

    input_file = sys.argv[1]
    num_clusters = int(sys.argv[2])
    eps = float(sys.argv[3])
    min_pts = int(sys.argv[4])

    data = load_data(input_file)
    labels = dbscan(data, eps, min_pts)
    save_clusters(data, labels, num_clusters, input_file.split('.')[0])
    
    # For testing and visualization
    # plot_clusters(data, labels, num_clusters)

if __name__ == "__main__":
    main()
