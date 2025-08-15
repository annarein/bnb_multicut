from bnb_solver import BnBSolver
from ilp_solver import ILPSolver
from graph_generators import get_random_costs_graph, get_test_zeros_graph
from visualizer import visualize_multicut_solution, plot_multicut_result
import time

def run_demo():
    # graph, costs, pos = get_random_costs_graph(seed=42, shape=(8, 3))
    # plot_multicut_result(graph, costs, pos, title="Original Graph")
    #
    # ilp = ILPSolver(graph.copy(), costs)
    # multicut_ilp, obj_ilp = ilp.solve()
    # print(f"[ILP] obj = {obj_ilp}")
    # visualize_multicut_solution(graph, costs, pos, multicut_ilp, "ILP Multicut")
    #
    # bnb = BnBSolver(graph.copy(), costs, False, False)
    # multicut_bnb, obj_bnb, count = bnb.solve()
    # print(f"[BnB] obj = {obj_bnb}, nodes = {count}")
    # visualize_multicut_solution(graph, costs, pos, multicut_bnb, "BnB Multicut")

    graph, costs, pos = get_random_costs_graph(seed=53, shape=(4, 3))
    # graph, costs, pos = get_test_zeros_graph(shape=(2, 2))
    for u, v in graph.edges():
        print(u, v, costs[(u, v)])
    # Original graph visualization (optional)
    plot_multicut_result(graph, costs, pos, multicut=None, node_labeling=None, title="Original Graph")

    # === ILP Solver ===
    solver_ilp = ILPSolver(graph.copy(), costs)
    start_time = time.time()
    multicut_ilp, obj_ilp = solver_ilp.solve()
    elapsed_ilp = time.time() - start_time
    print(f"ILP_multicut took {elapsed_ilp:.4f} seconds")
    visualize_multicut_solution(graph, costs, pos, multicut_ilp, "ILP Multicut Result")

    # === Branch and Bound Solver ===
    solver_bnb = BnBSolver(graph.copy(), costs, False, True)
    start_time = time.time()
    multicut_bnb, obj_bnb, count_bnb = solver_bnb.solve()
    elapsed_bnb = time.time() - start_time
    print(f"bnb_multicut took {elapsed_bnb:.4f} seconds")
    print(f"count_bnb: {count_bnb}")
    visualize_multicut_solution(graph, costs, pos, multicut_bnb, "BnB Multicut Result")

    print(f"obj_bnb: {obj_bnb}, obj_ilp: {obj_ilp}")
    assert abs(obj_bnb - obj_ilp) < 1e-6

if __name__ == "__main__":
    run_demo()
