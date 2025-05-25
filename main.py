import time
from ilp_solver import ILPSolver
from bnb_solver import BnBSolver
from graph_utils import get_random_costs_graph, get_trivial_graph, plot_multicut_result, get_test_zeros_graph
import networkx as nx


def get_node_labeling_from_multicut(graph: nx.Graph, cut_edges: dict):
    g_copy = graph.copy()
    g_copy.remove_edges_from([e for e in g_copy.edges if cut_edges.get((min(e[0], e[1]), max(e[0], e[1])), 0) == 1])
    components = nx.connected_components(g_copy)
    node_labeling = {n: i for i, comp in enumerate(components) for n in comp}
    return node_labeling


def visualize_multicut_solution(graph, costs, pos, multicut_dict, title):
    node_labels = get_node_labeling_from_multicut(graph, multicut_dict)
    plot_multicut_result(graph, costs, pos, multicut=multicut_dict, node_labeling=node_labels, title=title)


def main():
    graph, costs, pos = get_random_costs_graph(seed=37, shape=(10, 3))
    # for u, v in graph.edges():
    #     print(u, v, costs[(u, v)])

    plot_multicut_result(graph, costs, pos, title="Original Graph")

    # === ILP Solver ===
    solver_ilp = ILPSolver(graph.copy(), costs)
    start_time = time.time()
    multicut_ilp, obj_ilp = solver_ilp.solve()
    elapsed_ilp = time.time() - start_time
    print(f"ILP_multicut took {elapsed_ilp:.4f} seconds")
    # node_labeling_ilp = get_node_labeling(graph, multicut_ilp)
    # plot_multicut_result(graph, costs, pos, multicut_ilp, node_labeling_ilp, title="ILP Multicut Result")
    visualize_multicut_solution(graph, costs, pos, multicut_ilp, "ILP Multicut Result")

    # === Branch and Bound Solver ===
    solver_bnb = BnBSolver(graph.copy(), costs)
    start_time = time.time()
    multicut_bnb, obj_bnb, count_bnb = solver_bnb.solve()
    elapsed_bnb = time.time() - start_time
    print(f"bnb_multicut took {elapsed_bnb:.4f} seconds")
    print(f"count_bnb: {count_bnb}")
    print(obj_bnb, obj_ilp)
    # node_labeling_bnb = get_node_labeling(graph, multicut_bnb)
    # plot_multicut_result(graph, costs, pos, multicut_bnb, node_labeling_bnb, title="BnB Multicut Result")
    visualize_multicut_solution(graph, costs, pos, multicut_bnb, "BnB Multicut Result")


def benchmark(num_instances=1000, shape=(2, 3), tolerance=1e-6):
    for seed in range(num_instances):
        graph, costs, pos = get_random_costs_graph(seed=seed, shape=shape)
        # graph, costs, pos = get_test_zeros_graph()
        # graph, costs, pos = get_trivial_graph()

        # plot_multicut_result(graph, costs, pos, title="Original Graph")

        # === ILP Solver ===
        solver_ilp = ILPSolver(graph.copy(), costs)
        start_time = time.time()
        multicut_ilp, obj_ilp = solver_ilp.solve()
        elapsed_ilp = time.time() - start_time
        print(f"ILP_multicut took {elapsed_ilp:.4f} seconds")
        #
        # node_labeling_ilp = get_node_labeling(graph, multicut_ilp)
        # plot_multicut_result(graph, costs, pos, multicut_ilp, node_labeling_ilp, title="ILP Multicut Result")

        # === Branch and Bound Solver ===
        solver_bnb = BnBSolver(graph.copy(), costs)
        start_time = time.time()
        multicut_bnb, obj_bnb, count_bnb = solver_bnb.solve()
        elapsed_bnb = time.time() - start_time
        print(f"bnb_multicut took {elapsed_bnb:.4f} seconds")
        print(f"count_bnb: {count_bnb}")
        print(obj_bnb, obj_ilp)
        print(seed)
        assert abs(obj_bnb - obj_ilp) < tolerance

        # node_labeling_bnb = get_node_labeling(graph, multicut_bnb)
        # plot_multicut_result(graph, costs, pos, multicut_bnb, node_labeling_bnb, title="BnB Multicut Result")


if __name__ == "__main__":
    # main()  # for single test + visualization
    benchmark(num_instances=100, shape=(2, 3))  # for batch correctness check
