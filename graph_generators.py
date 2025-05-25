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

def get_random_costs_graph(seed=2, bias=0.3, shape=(5, 8)):
    np.random.seed(seed)
    graph = nx.grid_graph(shape)
    costs = {}
    for u, v in graph.edges():
        # edge = tuple(sorted((u, v)))
        p = np.random.rand()
        p = 1 / (1 + (1 - p) / p * bias / (1 - bias))
        costs[(u, v)] = np.log(p / (1 - p))
        # costs[edge] = np.log(p / (1 - p))
        # graph.add_edge(*edge, weight=costs)
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos

def get_test_zeros_graph(shape=(3, 3)):
    graph = nx.grid_graph(shape)
    costs = {(u, v): 0 for u, v in graph.edges()}
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos