from bnb_solver import BnBSolver
from ilp_solver import ILPSolver
from graph_generators import get_random_costs_graph

def run_benchmark(num_instances=300, shape=(5, 3), tolerance=1e-6):
    for seed in range(num_instances):
        graph, costs, _ = get_random_costs_graph(seed=seed, shape=shape)
        ilp = ILPSolver(graph.copy(), costs)
        _, obj_ilp = ilp.solve()

        bnb = BnBSolver(graph.copy(), costs)
        _, obj_bnb, _ = bnb.solve()

        print(f"[{seed}] ILP = {obj_ilp}, BnB = {obj_bnb}")
        assert abs(obj_bnb - obj_ilp) < tolerance

if __name__ == "__main__":
    run_benchmark()
