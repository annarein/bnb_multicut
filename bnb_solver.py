import copy
import networkx as nx
from collections import deque

def contract_and_merge_costs(graph: nx.Graph, costs: dict, u, v, cut_edges: dict, log=False):
    for (a, b) in costs:
        a_ = u if a == v else a
        b_ = u if b == v else b
        if a_ == b_:
            continue
        key = (min(a_, b_), max(a_, b_))
        if cut_edges.get((min(a, b), max(a, b)), -1) == 1 or cut_edges.get(key, -1) == 1:
            return None, None
    new_graph = nx.contracted_nodes(graph, u, v, self_loops=False)
    new_costs = {}
    for (a, b), w in costs.items():
        a_ = u if a == v else a
        b_ = u if b == v else b
        if a_ == b_:
            continue
        key = (min(a_, b_), max(a_, b_))
        new_costs[key] = new_costs.get(key, 0) + w
    return new_graph, new_costs

def propagate_zero_labels(cut_edges, u, v, costs, log=False):
    graph = {}
    for (a, b), val in cut_edges.items():
        if val == 0:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
    visited = set()
    queue = deque([u, v])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    visited = list(visited)
    newly_uncut = []
    for i in range(len(visited)):
        for j in range(i + 1, len(visited)):
            n1, n2 = visited[i], visited[j]
            edge = (min(n1, n2), max(n1, n2))
            if edge in cut_edges and cut_edges[edge] == -1:
                cut_edges[edge] = 0
                newly_uncut.append(edge)
    total_added_cost = sum(costs[e] for e in newly_uncut if e in costs)
    return cut_edges, total_added_cost

def is_feasible_cut(graph: nx.Graph, cut_edges: dict):
    g_copy = graph.copy()
    g_copy.remove_edges_from([e for e, val in cut_edges.items() if val == 1])
    components = nx.connected_components(g_copy)
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
    for (u, v), val in cut_edges.items():
        if val != 1:
            continue
        if u not in node_labeling or v not in node_labeling:
            continue
        if node_labeling[u] == node_labeling[v]:
            return False
    return True

def update_best_if_feasible(graph, cut_edges, obj, best):
    if is_feasible_cut(graph, cut_edges):
        if obj > best['obj']:
            best['obj'] = obj
            best['cut'] = copy.deepcopy(cut_edges)
            best['count'] = 1
        elif obj == best['obj']:
            best['count'] += 1

def update_best_if_feasible_final(graph, cut_edges, obj, best):
    if is_feasible_cut(graph, cut_edges):
        if obj > best['obj']:
            best['obj'] = obj
            best['cut'] = cut_edges
            best['count'] = 1
        elif obj == best['obj']:
            best['cut'] = cut_edges
            best['count'] += 1

def heuristic_bound(graph, costs, cut_edges):
    g_copy = nx.Graph()
    g_copy.add_nodes_from(graph.nodes)
    for (u, v), val in cut_edges.items():
        if val == 0 and graph.has_edge(u, v):
            g_copy.add_edge(u, v)
    components = list(nx.connected_components(g_copy))
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
    potential_gain = sum(
        w for (u, v), w in costs.items()
        if w > 0 and graph.has_edge(u, v)
        and node_labeling.get(u) is not None
        and node_labeling.get(v) is not None
        and node_labeling[u] != node_labeling[v]
    )
    return potential_gain

def bnb_multicut(graph: nx.Graph, costs: dict, cut_edges, obj, best: dict, log=False):
    if not costs:
        cut_edges_copy = cut_edges.copy()
        for e in cut_edges_copy:
            if cut_edges_copy[e] == -1:
                cut_edges_copy[e] = 1
        update_best_if_feasible_final(graph, cut_edges_copy, obj, best)
        return None
    bound = sum(w for w in costs.values() if w > 0)
    if obj + bound < best['obj']:
        return None
    edge, max_cost = max(costs.items(), key=lambda item: item[1])
    u, v = edge
    edge_key = (min(u, v), max(u, v))
    graph_join, costs_join = contract_and_merge_costs(graph.copy(), costs, u, v, cut_edges, log=log)
    if graph_join is not None:
        cut_edges_join = cut_edges.copy()
        cut_edges_join[edge_key] = 0
        cut_edges_join, delta_obj = propagate_zero_labels(cut_edges_join, u, v, costs, log)
        obj_join = obj + max_cost + delta_obj
        bnb_multicut(graph_join, costs_join, cut_edges_join, obj_join, best, log)
    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    cut_edges_cut[edge_key] = 1
    bnb_multicut(graph_cut, costs_cut, cut_edges_cut, obj, best, log)

class BnBSolver:
    def __init__(self, graph, costs, log=False):
        self.graph = graph
        self.costs = costs
        self.log = log

    def solve(self):
        normalized_costs = {
            (min(u, v), max(u, v)): w
            for (u, v), w in self.costs.items()
        }
        cut_edges = {e: -1 for e in normalized_costs}
        best = {'obj': 0, 'cut': cut_edges, 'count': 0}
        bnb_multicut(self.graph.copy(), normalized_costs, cut_edges, obj=0, best=best, log=self.log)
        obj = 0
        return best['cut'], obj, best['count']