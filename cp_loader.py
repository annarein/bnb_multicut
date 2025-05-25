import os
import networkx as nx


def load_cp_lib_instance(filepath):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # /.../bnb_multicut
    abs_path = os.path.join(script_dir, filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Resolved path does not exist: {abs_path}")
    with open(filepath, 'r') as f:
        lines = list(map(int, f.read().split()))
    n = lines[0]
    weights = lines[1:]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_weights = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[idx]
            if w != 0:
                G.add_edge(i, j)
                edge_weights[(i, j)] = w
            idx += 1
    return G, edge_weights