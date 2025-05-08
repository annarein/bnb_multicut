import networkx as nx
from collections import deque


def contract_and_merge_costs(graph: nx.Graph, costs: dict, u, v):
    """
    Contracts nodes u and v in the graph and merges edge costs accordingly.

    Edges connected to u or v are redirected to the merged node u,
    and parallel edge costs are summed.

    Args:
        graph (nx.Graph): Input undirected graph.
        costs (dict[tuple[int, int], float]): Edge cost map.
        u (int): One endpoint of the edge to contract.
        v (int): The other endpoint of the edge to contract.

    Returns:
        tuple[nx.Graph, dict]: The contracted graph and updated cost map.
    """
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


def propagate_zero_labels(cut_edges, u, v):
    """
    Incrementally enforces consistency of 0-labeled (uncut) edges within a connected component.

    Given a 0-labeled edge (u, v), this function finds all nodes connected to u or v
    through other 0-labeled edges, and sets to 0 any existing edges between these nodes
    that are not already labeled as 0. This ensures transitive closure in the current
    0-connected component while avoiding redundant updates.

    Args:
        cut_edges (dict[tuple[int, int], int]): Edge label map, where 0 means 'uncut'
            and 1 means 'cut'. Keys are undirected edges as (min(i, j), max(i, j)).
        u (int): One endpoint of a 0-labeled edge.
        v (int): The other endpoint of the 0-labeled edge.

    Returns:
        dict[tuple[int, int], int]: Updated edge labels with 0s propagated efficiently.
    """
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
    for i in range(len(visited)):
        for j in range(i + 1, len(visited)):
            n1, n2 = visited[i], visited[j]
            edge = (min(n1, n2), max(n1, n2))
            if cut_edges.get(edge, 1) != 0:  # only update if not already 0
                cut_edges[edge] = 0

    return cut_edges


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
            best['cut'] = cut_edges.copy()
            best['count'] = 1
        elif obj == best['obj']:
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
        if log:
            print(
                f"\033[92mcluster_obj={obj:.2f}\033[0m, "
                f"best_obj={best['obj']:.2f}"
            )
        update_best_if_feasible(graph, cut_edges, obj, best)
        return None

    bound = heuristic_bound(graph, costs, cut_edges)

    if log:
        print(
            f"\033[92mcluster_obj={obj:.2f}\033[0m, "
            f"\033[91mbound={bound:.2f}\033[0m, "
            f"best_obj={best['obj']:.2f}"
        )

    if is_feasible_cut(graph, cut_edges):
        update_best_if_feasible(graph, cut_edges, obj, best)
    else:
        if obj >= best['obj'] and log:
            print(f"[Skipping infeasible cut] obj={obj:.2f}")

    if obj + bound < best['obj']:
        return None

    edge, max_cost = max(costs.items(), key=lambda item: item[1])
    u, v = edge
    edge_key = (min(u, v), max(u, v))

    graph_join, costs_join = contract_and_merge_costs(graph.copy(), costs, u, v)
    cut_edges_join = cut_edges.copy()
    cut_edges_join[edge_key] = 0
    cut_edges_join = propagate_zero_labels(cut_edges_join, u, v)
    bnb_multicut(graph_join, costs_join, cut_edges_join, obj + max_cost, best, log)

    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    bnb_multicut(graph_cut, costs_cut, cut_edges_cut, obj, best, log)

    return best['obj'], best['cut']


class BnBSolver:
    def __init__(self, graph, costs, log=False):
        self.graph = graph
        self.costs = costs
        self.log = log

    def solve(self):
        cut_edges = {(min(u, v), max(u, v)): 1 for (u, v) in self.costs}
        best = {'obj': 0, 'cut': cut_edges, 'count': 0}
        bnb_multicut(self.graph.copy(), self.costs, cut_edges, obj=0, best=best, log=self.log)
        obj = sum(self.costs[e] for e, v in best['cut'].items() if v == 1)
        return best['cut'], obj, best['count']