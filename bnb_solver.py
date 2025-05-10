import networkx as nx
from collections import deque


def contract_and_merge_costs(graph: nx.Graph, costs: dict, u, v, cut_edges: dict, log=False):
    for (a, b) in costs:
        a_ = u if a == v else a
        b_ = u if b == v else b
        if a_ == b_:
            continue
        key = (min(a_, b_), max(a_, b_))
        if cut_edges.get((min(a, b), max(a, b)), -1) == 1 or cut_edges.get(key, -1) == 1:  # 我在这里就避免了本来cut的边merge以后又 uncut（或者是undecided?)
            if log:
                print(f"[Skip] merge ({u}, {v}) creates conflict: edge ({a}, {b}) or merged key {key} has cut label 1")
            return None, None  # Conflict found: skip this merge

    new_graph = nx.contracted_nodes(graph, u, v, self_loops=False)
    new_costs = {}
    for (a, b), w in costs.items():
        a_ = u if a == v else a
        b_ = u if b == v else b
        if a_ == b_:
            continue
        key = (min(a_, b_), max(a_, b_))
        new_costs[key] = new_costs.get(key, 0) + w  # 不知道为什么没有生成新的cost
    return new_graph, new_costs


def propagate_zero_labels(cut_edges, u, v, costs, log=False):
    if log:
        print(f"[PROP_ZERO] Start propagate_zero_labels from edge ({u}, {v})")
        print("  - Current cut_edges:")
        for edge, label in sorted(cut_edges.items()):
            print(f"    {edge}: {label}")
        print("  - Current costs:")
        for edge, value in sorted(costs.items()):
            print(f"    {edge}: {value:.2f}")

    uncut_adj = {}
    for (a, b), val in cut_edges.items():
        if val == 0:
            uncut_adj.setdefault(a, set()).add(b)
            uncut_adj.setdefault(b, set()).add(a)

    visited = set()
    queue = deque([u, v])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in uncut_adj.get(node, []):
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
                if log:
                    found = "FOUND" if edge in costs else "NOT FOUND"
                    value = costs.get(edge, 0)
                    print(f"  Propagate 0-label: edge {edge} with cost {value:.2f} ({found})")
    total_added_cost = sum(costs[e] for e in newly_uncut if e in costs)
    return cut_edges, total_added_cost


def is_feasible_cut(graph: nx.Graph, cut_edges: dict):  # is it necessary, will it produce infeasible when join or cut?
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


# def update_best_if_feasible(graph, cut_edges, obj, best):
#     if is_feasible_cut(graph, cut_edges):
#         if obj > best['obj']:
#             best['obj'] = obj
#             best['cut'] = copy.deepcopy(cut_edges)
#             best['count'] = 1
#         elif obj == best['obj']:
#             best['count'] += 1

def print_edge_labels_inline(graph, cut_edges, obj, best_obj):
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print(f"[GRAPH STATUS] obj = {obj:.2f}, best = {best_obj:.2f}")
    parts = []
    for u, v in graph.edges():
        e = (min(u, v), max(u, v))
        # label = cut_edges.get(e, -1)
        if e not in cut_edges:
            raise ValueError(f"Edge {e} not found in cut_edges!")
        label = cut_edges[e]
        if label == 1:
            parts.append(f"{RED}{e}{RESET}")
        elif label == 0:
            parts.append(f"{GREEN}{e}{RESET}")
        else:
            parts.append(f"{e}")  # undecided = default color
    print("  Edges: " + "  ".join(parts))


def update_best_if_feasible(graph, cut_edges, obj, best):
    if is_feasible_cut(graph, cut_edges):
        if obj > best['obj']:
            best['obj'] = obj
            best['cut'] = cut_edges
            best['count'] = 1
            print(f"[UPDATE] New best obj = {obj:.2f}")
            print_edge_labels_inline(graph, cut_edges, obj, best['obj'])
        elif obj == best['obj']:
            best['count'] += 1
            print(f"[TIE] Another feasible cut with obj = {obj:.2f}, total count = {best['count']}")


def update_best_if_feasible_final(graph, cut_edges, obj, best):
    if is_feasible_cut(graph, cut_edges):
        if obj > best['obj']:
            best['obj'] = obj
            best['cut'] = cut_edges
            best['count'] = 1
            print(f"[UPDATE] New best obj = {obj:.2f}")
            print_edge_labels_inline(graph, cut_edges, obj, best['obj'])
        elif obj == best['obj']:
            best[
                'cut'] = cut_edges  # 这是唯一的不一样，如果是obj最好， cut_edges 还是替换一下，为啥要换啊，因为如果之前存的临时的，可能所有边还没处理完？虽然 没处理完，但是obj肯定会越来越大，因为是正的，为啥要算作best呢
            best['count'] += 1
            print(f"[TIE] Another feasible cut with obj = {obj:.2f}, total count = {best['count']}")


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


def print_edge_label_groups(cut_edges: dict, tag: str = ""):
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    cut = {e for e, v in cut_edges.items() if v == 1}
    uncut = {e for e, v in cut_edges.items() if v == 0}
    undecided = {e for e, v in cut_edges.items() if v == -1}

    print(f"[EDGE LABELS{f' @ {tag}' if tag else ''}]")
    print(f"  cut edges:      {RED}{cut}{RESET}")
    print(f"  uncut edges:    {GREEN}{uncut}{RESET}")
    print(f"  undecided edges:{undecided}")


def bnb_multicut(graph: nx.Graph, costs: dict, cut_edges, obj, best: dict, log=False):
    if not costs:
        if log:
            print(f"[NO COSTS] \033[92mcluster_obj={obj:.2f}\033[0m, best_obj={best['obj']:.2f}")
        cut_edges_copy = cut_edges.copy()
        for e in cut_edges_copy:
            if cut_edges_copy[e] == -1:
                cut_edges_copy[e] = 1
        update_best_if_feasible_final(graph, cut_edges_copy, obj, best)
        return None

    # Compute the optimistic bound (e.g., sum of remaining positive weights)
    bound = sum(w for w in costs.values() if w > 0)
    # bound = heuristic_bound(graph, costs, cut_edges)

    if log:
        print(
            f"[ENTER BnB] \033[92mcluster_obj={obj:.2f}\033[0m, \033[91mbound={bound:.2f}\033[0m, best_obj={best['obj']:.2f}")

    if is_feasible_cut(graph, cut_edges):
        update_best_if_feasible(graph, cut_edges.copy(), obj, best)
    elif obj >= best['obj'] and log:
        print(f"[Skipping infeasible cut] obj={obj:.2f}")

    if obj + bound < best['obj']:
        if log:
            print(f"[PRUNE] Max possible obj = {obj + bound:.2f} < best obj = {best['obj']:.2f} → prune branch")
        return None

    edge, max_cost = max(costs.items(), key=lambda item: item[1])
    u, v = edge
    edge_key = (min(u, v), max(u, v))  # 所以就是这一步把 cut_edges 多出原来graph不存在的边的

    # Join branch
    if log:
        print_edge_label_groups(cut_edges, f"before JOIN ({u},{v})")
    graph_join, costs_join = contract_and_merge_costs(graph.copy(), costs, u, v, cut_edges, log=log)
    if graph_join is not None:
        cut_edges_join = cut_edges.copy()
        cut_edges_join[edge_key] = 0  # 所以就是这一步把 cut_edges 多出原来graph不存在的边的
        cut_edges_join, delta_obj = propagate_zero_labels(cut_edges_join, u, v, costs, log)
        obj_join = obj + max_cost + delta_obj
        if log:
            print(f"[BRANCH] Join: merging ({u}, {v}) with cost {max_cost:.2f} + delta_obj {delta_obj:.2f}")
            print(f"  - New objective: {obj_join:.2f}")
        bnb_multicut(graph_join, costs_join, cut_edges_join, obj_join, best, log)
        if log:
            print_edge_label_groups(cut_edges_join, f"after JOIN ({u},{v})")

    # Cut branch
    if log:
        print_edge_label_groups(cut_edges, f"before CUT ({u},{v})")
    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    cut_edges_cut[edge_key] = 1
    if log:
        print(f"[BRANCH] Cut: removing edge {edge_key} with cost {max_cost:.2f}, Objective unchanged: {obj:.2f}")
    bnb_multicut(graph_cut, costs_cut, cut_edges_cut, obj, best, log)
    if log:
        print_edge_label_groups(cut_edges_cut, f"after CUT ({u},{v})")


class BnBSolver:
    def __init__(self, graph, costs, log=False):
        self.graph = graph
        self.costs = costs
        self.log = log

    def solve(self):
        print(f"graph for bnb solver:")
        # normalized_costs = {
        #     (min(u, v), max(u, v)): w
        #     for (u, v), w in self.costs.items()
        # }
        normalized_costs = {}
        for (u, v), w in self.costs.items():
            key = (min(u, v), max(u, v))
            normalized_costs[key] = w
            print(f"{key}: {w:.2f}")
        cut_edges = {e: -1 for e in normalized_costs}
        best = {'obj': 0, 'cut': cut_edges, 'count': 0}
        bnb_multicut(self.graph.copy(), normalized_costs, cut_edges, obj=0, best=best, log=self.log)
        print("Cut edges:")
        obj = 0
        # for e, v in best['cut'].items(): #先注释掉不返回正确obj看看
        #     if v == 1:
        #         print(f"  {e} (cost={normalized_costs[e]:.4f})")
        #         obj += normalized_costs[e]
        # print(f"Total cost = {obj:.4f}")
        # obj = sum(self.costs[e] for e, v in best['cut'].items() if v == 1)
        print("[DEBUG] Final raw best cut:", best['cut'])
        return best['cut'], obj, best['count']
