import gurobipy as gp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB
import time


def get_trival_graph():
    graph = nx.Graph()
    graph.add_edge('a', 'b', weight=4)
    graph.add_edge('a', 'c', weight=-1)
    graph.add_edge('a', 'd', weight=3)
    graph.add_edge('b', 'd', weight=-2)
    graph.add_edge('c', 'd', weight=2)
    graph.add_edge('b', 'c', weight=-2)

    costs = {}
    for u, v, data in graph.edges(data=True):
        edge = (min(u, v), max(u, v))
        costs[edge] = data.get('weight')
    pos = nx.spring_layout(graph, seed=42)
    return graph, costs, pos


def get_random_costs_graph():
    # create a graph with random edge costs
    graph = nx.grid_graph((5, 5))
    bias = 0.3
    costs = {}
    for u, v in graph.edges():
        p = np.random.random()
        p = 1 / (1 + (1 - p) / p * bias / (1 - bias))
        costs[u, v] = np.log(p / (1 - p))
    pos = {n: n for n in graph.nodes}
    return graph, costs, pos


def plot_multicut_result(graph: nx.Graph, costs: dict, pos, multicut: dict = None, node_labeling: dict = None,
                         title=None):
    plt.figure(figsize=(6, 5))  # 设置图像大小

    edge_colors = ["green" if costs[(min(e), max(e))] > 0 else "red" for e in graph.edges]
    edge_widths = [1 + np.abs(costs[(min(e), max(e))]) for e in graph.edges]
    edge_styles = [":" if multicut and multicut.get((min(e), max(e)), 0) == 1 else "-" for e in graph.edges]

    # 设置节点颜色和颜色映射
    if node_labeling:
        node_colors = [int(node_labeling[n]) for n in graph.nodes]
        cmap = plt.get_cmap("tab20")
    else:
        node_colors = "gray"
        cmap = None

    nx.draw(
        graph,
        pos=pos,
        edge_color=edge_colors,
        width=edge_widths,
        style=edge_styles,
        node_color=node_colors,
        cmap=cmap
    )

    edge_labels = {e: f"{costs[(min(e), max(e))]:.2f}" for e in graph.edges}
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=6, font_color="black", label_pos=0.5
    )

    # node pos
    nx.draw_networkx_labels(graph, pos, labels={n: str(n) for n in graph.nodes}, font_size=5.5, font_color="white")

    if title:
        plt.title(title)
    plt.show()


def solve_multicut(graph: nx.Graph, costs: dict, log: bool = False):
    model = gp.Model()
    model.setParam('OutputFlag', 1 if log else 0)
    variables = model.addVars(costs.keys(), obj=costs, vtype=GRB.BINARY, name='e')

    def separate_cycle_inequalities(_, where):
        if where != GRB.Callback.MIPSOL:
            return
        vals = model.cbGetSolution(variables)
        g_copy = graph.copy()
        g_copy.remove_edges_from([e for e in g_copy.edges if vals.get((min(e), max(e)), 0) > 0.5])
        components = nx.connected_components(g_copy)
        node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
        for (u, v), x_uv in vals.items():
            if x_uv < 0.5 or node_labeling[u] != node_labeling[v]:
                continue
            path = nx.shortest_path(g_copy, u, v)
            assert len(path) >= 2
            model.cbLazy(variables[u, v] <= gp.quicksum(
                variables[min(path[i], path[i + 1]), max(path[i], path[i + 1])]
                for i in range(len(path) - 1)
            ))

    model.Params.LazyConstraints = 1
    model.optimize(separate_cycle_inequalities)
    solution = model.getAttr("X", variables)
    multicut = {(min(u, v), max(u, v)): 1 if x_e > 0.5 else 0 for (u, v), x_e in solution.items()}
    return multicut, model.ObjVal


def contract_and_merge_costs(graph: nx.Graph, costs: dict, u, v):
    new_graph = nx.contracted_nodes(graph.copy(), u, v, self_loops=False)
    new_costs = {}
    for (a, b), w in costs.items():
        a_ = u if a == v else a
        b_ = u if b == v else b
        if a_ == b_:
            continue
        key = (min(a_, b_), max(a_, b_))
        new_costs[key] = new_costs.get(key, 0) + w
    return new_graph, new_costs


# def propagate_zero_labels(cut_edges, u, v):
#     connected_nodes = set()
#     for (a, b), val in cut_edges.items():
#         if val == 0 and (u in (a, b) or v in (a, b)):
#             connected_nodes.update([a, b])
#     for n1 in connected_nodes:
#         for n2 in connected_nodes:
#             if n1 == n2:
#                 continue
#             edge = (min(n1, n2), max(n1, n2))
#             if edge in cut_edges:
#                 cut_edges[edge] = 0
#     return cut_edges


from collections import deque
def propagate_zero_labels(cut_edges, u, v):
    # Build adjacency list for edges with val == 0 (i.e., not cut)
    graph = {}
    for (a, b), val in cut_edges.items():
        if val == 0:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)

    # Start BFS from u and v to find all nodes in the same connected component
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

    # For any node pair in the component, set the corresponding edge to 0 (not cut)
    for n1 in visited:
        for n2 in visited:
            if n1 == n2:
                continue
            edge = (min(n1, n2), max(n1, n2))
            if edge in cut_edges:
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
            continue  # Skip node pairs not present in the current graph
        if node_labeling[u] == node_labeling[v]:
            return False
    return True


# 用一个更聪明的启发式：比如把剩下的边按值从高到低贪心地加入，不违反 cycle constraint 为止
def heuristic_bound(graph, costs, cut_edges):
    # ✅ 构建图中只包含未被切断（cut_edges[e] == 0）的边
    g_copy = nx.Graph()
    g_copy.add_nodes_from(graph.nodes)

    for (u, v), val in cut_edges.items():
        if val == 0 and graph.has_edge(u, v):
            g_copy.add_edge(u, v)

    # ✅ 获取连通分量并标注每个节点所属 cluster
    components = list(nx.connected_components(g_copy))
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}

    # ✅ 对所有还存在于图中的边中，找连接不同分量的“正权重边”
    potential_gain = sum(
        w for (u, v), w in costs.items()
        if w > 0
        and graph.has_edge(u, v)
        and node_labeling.get(u) is not None
        and node_labeling.get(v) is not None
        and node_labeling[u] != node_labeling[v]
    )
    return potential_gain

def heuristic_bound_log (graph, costs, cut_edges, verbose=False):
    # ✅ 构建图中只包含 cut_edges == 0 的边（即未被切断的边）
    g_copy = nx.Graph()
    g_copy.add_nodes_from(graph.nodes)

    for (u, v), val in cut_edges.items():
        if val == 0 and graph.has_edge(u, v):
            g_copy.add_edge(u, v)

    # ✅ 获取当前节点分组
    components = list(nx.connected_components(g_copy))
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}

    potential_gain = 0
    if verbose:
        print("[Heuristic Bound] candidate cross-cluster edges with positive weight:")

    for (u, v), w in costs.items():
        if w <= 0:
            continue
        if not graph.has_edge(u, v):
            continue
        c1 = node_labeling.get(u)
        c2 = node_labeling.get(v)
        if c1 is None or c2 is None or c1 == c2:
            continue
        potential_gain += w
        if verbose:
            print(f"  Edge ({u}, {v}) | weight = {w:.2f} | cluster {c1} ≠ {c2}")

    if verbose:
        print(f"[Heuristic Bound] total gain = {potential_gain:.2f}")

    return potential_gain

def bnb_multicut(graph: nx.Graph, costs: dict, cut_edges, obj, best: dict, log=False):
    # If no edges remain, update best solution at the leaf node if feasible
    if not costs:
        if obj > best['obj'] and is_feasible_cut(graph, cut_edges):
            best['obj'] = obj
            best['cut'] = cut_edges.copy()
        return  # 没有剩余边可选了 ⇒ 不再继续分支

    # Compute the optimistic bound (e.g., sum of remaining positive weights)
    # bound = sum(w for w in costs.values() if w > 0)
    # bound = heuristic_bound_log(graph, costs, cut_edges, verbose=log)
    bound = heuristic_bound(graph, costs, cut_edges)

    if log:
        print(
            f"\033[92mcluster_obj={obj:.2f}\033[0m, "
            f"\033[91mbound={bound:.2f}\033[0m, "
            f"best_obj={best['obj']:.2f}"
        )

    # If current partial solution is better and feasible, save it
    if obj > best['obj'] and is_feasible_cut(graph, cut_edges):
        best['obj'] = obj
        best['cut'] = cut_edges.copy()
    elif obj > best['obj']:
        if log:
            print(f"[Skipping infeasible cut] obj={obj:.2f}")

    # If current upper bound is worse than best, prune this branch
    if obj + bound < best['obj']:
        return

    edge, max_cost = max(costs.items(), key=lambda item: item[1])
    u, v = edge
    edge_key = (min(u, v), max(u, v))

    # Join branch
    graph_join, costs_join = contract_and_merge_costs(graph.copy(), costs, u, v)
    cut_edges_join = cut_edges.copy()
    cut_edges_join[edge_key] = 0
    cut_edges_join = propagate_zero_labels(cut_edges_join, u, v)
    bnb_multicut(graph_join, costs_join, cut_edges_join, obj + max_cost, best, log)

    # Cut branch
    graph_cut = graph.copy()
    graph_cut.remove_edge(u, v)
    costs_cut = {e: w for e, w in costs.items() if e != edge}
    cut_edges_cut = cut_edges.copy()
    bnb_multicut(graph_cut, costs_cut, cut_edges_cut, obj, best, log)

    return best['obj'], best['cut']


def main():
    np.random.seed(2)  # Set seed once here to ensure reproducibility across platforms
    graph, costs, pos = get_random_costs_graph()
    # graph, costs, pos = get_trival_graph()

    # plot the original graph
    plot_multicut_result(graph, costs, pos, title="Original Graph")

    # call the ILP solver
    g_copy = graph.copy()
    # ⏱️ Start timing
    start_time = time.time()
    multicut, obj = solve_multicut(g_copy, costs)
    # ⏱️ End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ILP_multicut took {elapsed_time:.4f} seconds")
    print("multicut ILP obj:", obj)
    print("cut edge set：", {e for e, v in multicut.items() if v == 1})

    g_copy.remove_edges_from([e for e in g_copy.edges if multicut.get((min(e[0], e[1]), max(e[0], e[1])), 0) == 1])
    components = nx.connected_components(g_copy)
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
    for (u, v), x_uv in multicut.items():
        if x_uv == 1 and (node_labeling[u] == node_labeling[v]):
            raise ValueError(f"Cycle inequality for edge {u}, {v} is violated")

    plot_multicut_result(graph.copy(), costs, pos, multicut, node_labeling, title="ILP Multicut Result")

    # branch and bound
    cut_edges = {(min(u, v), max(u, v)): 1 for (u, v) in costs}  # all edges are cut
    best = {'obj': 0, 'cut': cut_edges}
    g_copy = graph.copy()
    # ⏱️ Start timing
    start_time = time.time()
    bnb_multicut(g_copy, costs, cut_edges=cut_edges, obj=0, best=best)
    # ⏱️ End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"bnb_multicut took {elapsed_time:.4f} seconds")

    # Final BnB solution
    multicut = best['cut']
    obj = sum(costs[e] for e, v in multicut.items() if v == 1)
    print("multicut bnb obj:", obj)
    print("cut edge set: ", {e for e, v in multicut.items() if v == 1})

    # Generate node_labeling as ILP
    # g_copy = graph.copy()
    # g_copy.remove_edges_from([e for e, v in multicut.items() if v == 1])
    # components = nx.connected_components(g_copy)
    # node_labeling = {n: i for i, comp in enumerate(components) for n in comp}

    # Validate feasibility
    if not is_feasible_cut(graph, multicut):
        raise ValueError("Final BnB solution violates cycle inequality!")

    # Plot result
    plot_multicut_result(graph.copy(), costs, pos, multicut, node_labeling, title="bnb Multicut Result")


if __name__ == "__main__":
    main()