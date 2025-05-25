def parse_opt_solution(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    clusters = []
    for line in lines:
        if line.strip().startswith('{'):
            line = line.strip().strip('{}')
            cluster = list(map(int, line.strip().split()))
            clusters.append(cluster)
    return clusters

def compute_objective(n, edges, clusters):
    node_to_cluster = {}
    for i, group in enumerate(clusters):
        for node in group:
            node_to_cluster[node - 1] = i  # CP-Lib 节点是1-based

    obj = 0
    for u, v, w in edges:
        if node_to_cluster.get(u) == node_to_cluster.get(v):
            obj += w
    return obj

def extract_opt_value(opt_file):
    with open(opt_file) as f:
        for line in f:
            if "Optimal value" in line:
                return int(line.strip().split(":")[1])
    return None