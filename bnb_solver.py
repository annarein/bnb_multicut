import networkx as nx
from collections import deque


def merge_clusters(graph, u, v):
    original_u = graph.nodes[u]['cluster'].copy()
    original_v = graph.nodes[v]['cluster'].copy()
    graph.nodes[u]['cluster'] |= graph.nodes[v]['cluster']
    graph.nodes[v]['cluster'] = graph.nodes[u]['cluster']
    return ('merge', u, v, original_u, original_v)


def unmerge_clusters(graph, op):
    _, u, v, orig_u, orig_v = op
    graph.nodes[u]['cluster'] = orig_u
    graph.nodes[v]['cluster'] = orig_v


def remove_edge(graph, u, v):
    if graph.has_edge(u, v):
        weight = graph[u][v]['weight']
        graph.remove_edge(u, v)
        return ('delete', u, v, weight)
    return None


def restore_edge(graph, op):
    if op:
        _, u, v, weight = op
        graph.add_edge(u, v, weight=weight)


def propagate_zero_labels(graph: nx.Graph, u, v):
    label_graph = {}
    for a, b in graph.edges:
        if graph.nodes[a]['cluster'] == graph.nodes[b]['cluster']:
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
    total_added_cost = 0
    ops = []
    for i in range(len(visited)):
        for j in range(i + 1, len(visited)):
            n1, n2 = visited[i], visited[j]
            if graph.has_edge(n1, n2) and graph.nodes[n1]['cluster'] != graph.nodes[n2]['cluster']:
                total_added_cost += graph[n1][n2]['weight']
                op = merge_clusters(graph, n1, n2)
                ops.append(op)
    return total_added_cost, ops


def is_feasible(graph: nx.Graph):
    for u, v in graph.edges:
        if graph.nodes[u]['cluster'] != graph.nodes[v]['cluster']:
            return False  # There is an edge within different clusters — this violates feasibility
    return True


def update_best_if_feasible(graph, obj, best):
    if is_feasible(graph):
        if obj > best['obj']:
            best['obj'] = obj
            best['graph'] = {n: graph.nodes[n]['cluster'].copy() for n in graph.nodes}
            best['count'] = 1
        elif obj == best['obj']:
            best['graph'] = {n: graph.nodes[n]['cluster'].copy() for n in graph.nodes}
            best['count'] += 1


def bnb_multicut(graph: nx.Graph, obj, best: dict, log=False):
    remaining_edges = [(u, v) for u, v in graph.edges if graph.nodes[u]['cluster'] != graph.nodes[v]['cluster']]
    if not remaining_edges:
        update_best_if_feasible(graph, obj, best)
        return

    edge = max(remaining_edges, key=lambda e: graph[e[0]][e[1]]['weight'])
    u, v = edge
    w = graph[u][v]['weight']

    # --- join branch ---
    if not (len(remaining_edges) == 1 and w <= 0):
        op1 = merge_clusters(graph, u, v)
        delta_obj, merge_ops = propagate_zero_labels(graph, u, v)
        bnb_multicut(graph, obj + w + delta_obj, best, log)
        for op in reversed(merge_ops):
            unmerge_clusters(graph, op)
        unmerge_clusters(graph, op1)

    # --- cut branch ---
    op2 = remove_edge(graph, u, v)
    bnb_multicut(graph, obj, best, log)
    restore_edge(graph, op2)


class BnBSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def solve(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]['cluster'] = {node}  # reset cluster state for fresh solve

        best = {'obj': 0, 'graph': {}, 'count': 0}
        bnb_multicut(self.graph, obj=0, best=best)

        obj = sum(
            self.graph[u][v]['weight']
            for u, v in self.graph.edges
            if best['graph'].get(u) and best['graph'].get(v)
            and best['graph'][u] != best['graph'][v]
        )
        return best['graph'], obj, best['count']