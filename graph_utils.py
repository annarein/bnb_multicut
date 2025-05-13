import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def get_trivial_graph(seed=42):
    graph = nx.Graph()
    graph.add_edge('a', 'b', weight=4)
    graph.add_edge('a', 'c', weight=-1)
    graph.add_edge('a', 'd', weight=3)
    graph.add_edge('b', 'd', weight=-2)
    graph.add_edge('c', 'd', weight=2)
    graph.add_edge('b', 'c', weight=-2)
    pos = nx.spring_layout(graph, seed)
    costs = {(min(u, v), max(u, v)): graph[u][v]['weight'] for u, v in graph.edges}  # for ILP compatibility
    return graph, costs, pos

def get_random_costs_graph(seed=2, bias=0.3, shape=(5, 7)):
    np.random.seed(seed)
    graph = nx.grid_graph(shape)
    costs = {}
    for u, v in graph.edges():
        p = np.random.rand()
        p = 1 / (1 + (1 - p) / p * bias / (1 - bias))
        w = np.log(p / (1 - p))
        costs[(min(u, v), max(u, v))] = w
        graph.add_edge(u, v, weight=w)
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos

def get_test_zeros_graph(shape=(3, 3)):
    graph = nx.grid_graph(shape)
    costs = {(min(u, v), max(u, v)): 0 for u, v in graph.edges()}
    for u, v in graph.edges():
        graph[u][v]['weight'] = 0
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos

def plot_multicut_result(graph: nx.Graph, pos, multicut=None, node_labeling=None, title=None):
    plt.figure(figsize=(5, 5))

    edge_colors = ["green" if graph[u][v]['weight'] > 0 else "red" for u, v in graph.edges]
    edge_widths = [1 + abs(graph[u][v]['weight']) for u, v in graph.edges]
    edge_styles = [":" if multicut and multicut.get((min(u, v), max(u, v)), 0) == 1 else "-" for u, v in graph.edges]

    if not node_labeling and 'cluster' in graph.nodes[next(iter(graph.nodes))]:
        cluster_ids = {frozenset(c): i for i, c in enumerate(
            {frozenset(graph.nodes[n]['cluster']) for n in graph.nodes})}
        node_colors = [cluster_ids[frozenset(graph.nodes[n]['cluster'])] for n in graph.nodes]
    else:
        node_colors = [int(node_labeling[n]) for n in graph.nodes] if node_labeling else "gray"

    cmap = plt.get_cmap("tab20") if isinstance(node_colors[0], int) else None
    nx.draw(graph, pos, edge_color=edge_colors, width=edge_widths, style=edge_styles,
            node_color=node_colors, cmap=cmap)

    edge_labels = {(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
    nx.draw_networkx_labels(graph, pos, font_size=5, font_color="white")

    if title:
        plt.title(title)
    plt.show()