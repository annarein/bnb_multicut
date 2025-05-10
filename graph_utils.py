import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def get_trivial_graph():
    graph = nx.Graph()
    graph.add_edge('a', 'b', weight=4)
    graph.add_edge('a', 'c', weight=-1)
    graph.add_edge('a', 'd', weight=3)
    graph.add_edge('b', 'd', weight=-2)
    graph.add_edge('c', 'd', weight=2)
    graph.add_edge('b', 'c', weight=-2)

    costs = {(min(u, v), max(u, v)): data["weight"] for u, v, data in graph.edges(data=True)}
    pos = nx.spring_layout(graph, seed=42)
    return graph, costs, pos

def get_random_costs_graph(seed=2, bias=0.3, shape=(5, 7)):
    np.random.seed(seed)
    graph = nx.grid_graph(shape)
    costs = {}
    for u, v in graph.edges():
        p = np.random.rand()
        p = 1 / (1 + (1 - p) / p * bias / (1 - bias))
        # edge = tuple(sorted((u, v)))  # make sure edge key is sorted
        # costs[edge] = np.log(p / (1 - p))
        costs[(u, v)] = np.log(p / (1 - p))
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos

def get_test_zeros_graph(shape=(3, 3)):
    graph = nx.grid_graph(shape)
    costs = {(u, v): 0 for u, v in graph.edges()}
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos

def plot_multicut_result(graph: nx.Graph, costs: dict, pos, multicut: dict = None,
                         node_labeling: dict = None, title=None):
    plt.figure(figsize=(5, 5))
    edge_colors = ["green" if costs[(min(e), max(e))] > 0 else "red" for e in graph.edges]
    edge_widths = [1 + np.abs(costs[(min(e), max(e))]) for e in graph.edges]
    edge_styles = [":" if multicut and multicut.get((min(e), max(e)), 0) == 1 else "-" for e in graph.edges]
    node_colors = [int(node_labeling[n]) for n in graph.nodes] if node_labeling else "gray"
    cmap = plt.get_cmap("tab20") if node_labeling else None
    nx.draw(graph, pos, edge_color=edge_colors, width=edge_widths, style=edge_styles,
            node_color=node_colors, cmap=cmap)

    edge_labels = {e: f"{costs[(min(e), max(e))]:.2f}" for e in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)
    nx.draw_networkx_labels(graph, pos, font_size=5, font_color="white")

    if title:
        plt.title(title)
    plt.show()