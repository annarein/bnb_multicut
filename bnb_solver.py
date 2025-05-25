import networkx as nx
from collections import deque


def contract_and_merge_costs(graph: nx.Graph, costs: dict, a, b, cut_edges: dict, log=False):
    if not graph.has_node(a) or not graph.has_node(b):
        return None, None, None

    # Step 1: prepare new cut_edges
    new_cut_edges = cut_edges.copy()

    neighbors = set(graph.neighbors(a)).union(graph.neighbors(b))
    neighbors.discard(a)
    neighbors.discard(b)

    for c in neighbors:
        key_ac = (min(a, c), max(a, c))
        key_bc = (min(b, c), max(b, c))
        cut_bc = cut_edges.get(key_bc, -1)
        if cut_bc == 1:
            new_cut_edges[key_ac] = 1

    # Step 2: prepare new_costs BEFORE merge
    new_costs = {}
    touched = set()

    for c in neighbors:
        key_ac = (min(a, c), max(a, c))
        key_bc = (min(b, c), max(b, c))
        cost_a = costs.get(key_ac, 0)
        cost_b = costs.get(key_bc, 0)
        new_costs[key_ac] = cost_a + cost_b
        touched.add(key_ac)

    # keep other costs unchanged
    for (u, v), w in costs.items():
        key = (min(u, v), max(u, v))
        if key not in touched and b not in key:
            new_costs[key] = w

    # Step 3: merge
    new_graph = nx.contracted_nodes(graph, a, b, self_loops=False)

    return new_graph, new_costs, new_cut_edges


def propagate_zero_labels(cut_edges, u, v, costs):
    label_graph = {}
    for (a, b), val in cut_edges.items():
        if val == 0:
            label_graph.setdefault(a, set()).add(b)
            label_graph.setdefault(b, set()).add(a)
    visited = set()
    queue = deque([u, v])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in label_graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    visited = list(visited)
    for i in range(len(visited)):
        for j in range(i + 1, len(visited)):
            n1, n2 = visited[i], visited[j]
            edge = (min(n1, n2), max(n1, n2))
            if edge in cut_edges and cut_edges[edge] == -1:
                cut_edges[edge] = 0
    return costs


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


def update_best_if_feasible_final(graph, cut_edges, obj, best):
    if is_feasible_cut(graph, cut_edges):
        if obj > best['obj']:
            best['obj'] = obj
            best['cut'] = cut_edges
            best['count'] = 1
        elif obj == best['obj']:
            best['cut'] = cut_edges
            best['count'] += 1


def bnb_multicut(graph: nx.Graph, costs: dict, cut_edges, obj, best: dict):
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
    graph_join, costs_join, cut_edges_join = contract_and_merge_costs(graph.copy(), costs, u, v, cut_edges)
    if costs_join is not None:
        # cut_edges_join = cut_edges.copy()
        cut_edges_join[edge_key] = 0
        for c in set(graph.neighbors(u)) & set(graph.neighbors(v)):
            e_uc = (min(u, c), max(u, c))
            e_vc = (min(v, c), max(v, c))
            if cut_edges_join.get(e_uc, -1) == 1 or cut_edges_join.get(e_vc, -1) == 1:
                if cut_edges_join.get(e_uc, -1) == -1:
                    cut_edges_join[e_uc] = 1
                if cut_edges_join.get(e_vc, -1) == -1:
                    cut_edges_join[e_vc] = 1
        propagate_zero_labels(cut_edges_join, u, v, costs_join)
        obj_join = obj + max_cost
        bnb_multicut(graph_join, costs_join, cut_edges_join, obj_join, best)
    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    cut_edges_cut[edge_key] = 1
    bnb_multicut(graph_cut, costs_cut, cut_edges_cut, obj, best)


class BnBSolver:
    def __init__(self, graph, costs):
        self.graph = graph
        self.costs = costs

    def solve(self):
        normalized_costs = {
            (min(u, v), max(u, v)): w
            for (u, v), w in self.costs.items()
        }
        cut_edges = {e: -1 for e in normalized_costs}
        best = {'obj': 0, 'cut': cut_edges, 'count': 0}
        bnb_multicut(self.graph.copy(), normalized_costs, cut_edges, obj=0, best=best)
        obj = 0
        for u, v in self.graph.edges():
            e = (min(u, v), max(u, v))
            if best['cut'].get(e, -1) == 1:
                cost = self.costs.get(e, 0)
                obj += cost
        return best['cut'], obj, best['count']