import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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


def get_node_labeling_from_multicut(graph: nx.Graph, cut_edges: dict):
    g_copy = graph.copy()
    g_copy.remove_edges_from([e for e in g_copy.edges if cut_edges.get((min(e[0], e[1]), max(e[0], e[1])), 0) == 1])
    components = nx.connected_components(g_copy)
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
    return node_labeling


def visualize_multicut_solution(graph, costs, pos, multicut_dict, title):
    node_labels = get_node_labeling_from_multicut(graph, multicut_dict)
    plot_multicut_result(graph, costs, pos, multicut=multicut_dict, node_labeling=node_labels, title=title)