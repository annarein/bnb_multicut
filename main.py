import time
import networkx as nx
from ilp_solver import ILPSolver
from bnb_solver import BnBSolver
from graph_utils import get_random_costs_graph, plot_multicut_result


def get_node_labeling_from_multicut(graph: nx.Graph, cut_edges: dict):
    g_copy = graph.copy()
    g_copy.remove_edges_from([
        (u, v) for u, v in graph.edges
        if cut_edges.get((min(u, v), max(u, v)), 0) == 1
    ])
    components = nx.connected_components(g_copy)
    return {node: i for i, comp in enumerate(components) for node in comp}


def cluster_to_multicut(graph: nx.Graph, cluster_dict: dict):
    return {
        (min(u, v), max(u, v)): 1 if cluster_dict[u] != cluster_dict[v] else 0
        for u, v in graph.edges
    }

def visualize_multicut_solution(graph, pos, multicut_dict, title):
    node_labels = get_node_labeling_from_multicut(graph, multicut_dict)
    plot_multicut_result(graph, pos, title=title, multicut=multicut_dict, node_labeling=node_labels)

def main():
    graph, costs, pos = get_random_costs_graph(seed=37, shape=(2, 3))

    # === ILP Solver ===
    solver_ilp = ILPSolver(graph.copy(), costs, log=False)
    start_ilp = time.time()
    multicut_ilp, obj_ilp = solver_ilp.solve()
    time_ilp = time.time() - start_ilp
    print(f"ILP objective = {obj_ilp:.4f}, time = {time_ilp:.4f} s")
    visualize_multicut_solution(graph, pos, multicut_ilp, "ILP Multicut Result")

    # === Branch and Bound Solver ===
    solver_bnb = BnBSolver(graph.copy(), log=False)
    start_bnb = time.time()
    cluster_bnb, obj_bnb, count_bnb = solver_bnb.solve()
    time_bnb = time.time() - start_bnb
    print(f"BnB objective = {obj_bnb:.4f}, time = {time_bnb:.4f} s, count = {count_bnb}")

    multicut_bnb = cluster_to_multicut(graph, cluster_bnb)
    visualize_multicut_solution(graph, pos, multicut_bnb, "BnB Multicut Result")


def benchmark(num_instances=100, shape=(2, 3), tolerance=1e-6):
    for seed in range(num_instances):
        print(f"\n=== Seed {seed} ===")
        graph, costs, pos = get_random_costs_graph(seed=seed, shape=shape)

        # === ILP Solver ===
        solver_ilp = ILPSolver(graph.copy(), costs, log=False)
        multicut_ilp, obj_ilp = solver_ilp.solve()

        # === BnB Solver ===
        solver_bnb = BnBSolver(graph.copy(), log=False)
        cluster_bnb, obj_bnb, count_bnb = solver_bnb.solve()

        print(f"ILP: {obj_ilp:.6f},  BnB: {obj_bnb:.6f},  count = {count_bnb}")

        if abs(obj_ilp - obj_bnb) > tolerance:
            print("❌ Mismatch detected!")
            multicut_bnb = cluster_to_multicut(graph, cluster_bnb)
            visualize_multicut_solution(graph, pos, multicut_ilp, "ILP Multicut Result")
            visualize_multicut_solution(graph, pos, multicut_bnb, "BnB Multicut Result")
            break
    else:
        print(f"\n✅ All {num_instances} instances passed.")


if __name__ == "__main__":
    main()  # for single test + visualization
    # benchmark(num_instances=100, shape=(2, 3))  # for batch correctness check