import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from bnb_solver import BnBSolver
from ilp_solver import ILPSolver
from graph_generators import get_random_costs_graph


# -----------------------------
# Core: per-shape micro-benchmark
# -----------------------------
def run_benchmark_one_shape(
    shape: Tuple[int, int],
    num_instances: int = 100,
    tolerance: float = 1e-6,
    base_seed: int = 42,
    out_dir: str = "results",
    do_warmup: bool = True,
    file_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run ILP vs BnB on the same batch of random-cost graphs for a single 'shape'.
    Produces:
      - CSV '<out_dir>/shape_{rows}x{cols}.csv' with columns: run,ilp_time_s,bnb_time_s,speedup
      - CSV '<out_dir>/shape_{rows}x{cols}_diag.csv' with extended diagnostics

    Returns a dict summary with arrays and aggregate stats for plotting/summary CSV.
    """
    rows, cols = shape
    tag = file_tag or f"shape_{rows}x{cols}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_main = out_dir / f"{tag}.csv"
    csv_diag = out_dir / f"{tag}_diag.csv"

    # fixed RNG & seeds for this shape
    rng = np.random.default_rng(base_seed)
    total_runs = num_instances + (1 if do_warmup else 0)
    seeds = rng.integers(0, 2**32 - 1, size=total_runs, dtype=np.uint32).tolist()

    run_ids, ilp_times, bnb_times, speedups = [], [], [], []
    last_nodes = last_edges = None
    mismatches = 0

    with csv_main.open("w", newline="") as f_main, csv_diag.open("w", newline="") as f_diag:
        w = csv.writer(f_main)
        wd = csv.writer(f_diag)

        # headers as requested
        w.writerow(["run", "ilp_time_s", "bnb_time_s", "speedup"])
        wd.writerow([
            "run", "seed", "nodes", "edges",
            "obj_ilp", "obj_bnb",
            "ilp_time_s", "bnb_time_s",
            "speedup", "status", "bnb_path_or_trace"
        ])

        run_id = 0

        for idx, seed in enumerate(seeds):
            # If your generator supports density or other knobs, pass them here.
            graph, costs, _ = get_random_costs_graph(seed=int(seed), shape=shape)
            last_nodes = graph.number_of_nodes()
            last_edges = graph.number_of_edges()

            # --- ILP ---
            t0 = time.time()
            ilp = ILPSolver(graph.copy(), costs)
            ilp_solution, obj_ilp = ilp.solve()
            t1 = time.time()
            ilp_time = t1 - t0

            # --- BnB ---
            t2 = time.time()
            bnb = BnBSolver(graph.copy(), costs, False, False)
            try:
                bnb_solution, obj_bnb, bnb_trace = bnb.solve()
            except ValueError:
                bnb_solution, obj_bnb = bnb.solve()
                bnb_trace = None
            t3 = time.time()
            bnb_time = t3 - t2

            # warm-up: run but do not log
            if do_warmup and idx == 0:
                print(f"[warmup|{tag}] seed={seed} Nodes={last_nodes}, Edges={last_edges} "
                      f"ILP={obj_ilp:.6f} ({ilp_time:.4f}s), BnB={obj_bnb:.6f} ({bnb_time:.4f}s)")
                continue

            run_id += 1
            speedup = ilp_time / bnb_time if bnb_time > 0 else float("inf")

            status = "OK"
            if abs(obj_bnb - obj_ilp) >= tolerance:
                status = "MISMATCH"
                mismatches += 1

            # console line
            print(f"[{tag}|{run_id}] seed={seed} n={last_nodes}, m={last_edges} | "
                  f"ILP={obj_ilp:.6f} ({ilp_time:.4f}s) | "
                  f"BnB={obj_bnb:.6f} ({bnb_time:.4f}s) | "
                  f"speedup(ILP/BnB)={speedup:.3f} | {status}")

            # write CSVs
            w.writerow([run_id, ilp_time, bnb_time, speedup])

            bnb_path_str = ""
            if bnb_trace is not None:
                try:
                    bnb_path_str = str(bnb_trace)
                    if len(bnb_path_str) > 2000:
                        bnb_path_str = bnb_path_str[:2000] + " ... [truncated]"
                except Exception:
                    bnb_path_str = "<unserializable trace>"

            wd.writerow([
                run_id, int(seed), last_nodes, last_edges,
                f"{obj_ilp:.12f}", f"{obj_bnb:.12f}",
                f"{ilp_time:.6f}", f"{bnb_time:.6f}",
                f"{speedup:.6f}", status, bnb_path_str
            ])

            # accumulate
            run_ids.append(run_id)
            ilp_times.append(ilp_time)
            bnb_times.append(bnb_time)
            speedups.append(speedup)

    # aggregate stats
    ilp_arr = np.array(ilp_times, dtype=float)
    bnb_arr = np.array(bnb_times, dtype=float)
    spd_arr = np.array(speedups, dtype=float)

    summary = {
        "shape": shape,
        "nodes": last_nodes,
        "edges": last_edges,
        "runs": len(run_ids),
        "mismatches": mismatches,
        "ilp_avg": float(ilp_arr.mean()) if ilp_arr.size else np.nan,
        "bnb_avg": float(bnb_arr.mean()) if bnb_arr.size else np.nan,
        "ratio_avg": float(spd_arr.mean()) if spd_arr.size else np.nan,
        "ratio_median": float(np.median(spd_arr)) if spd_arr.size else np.nan,
        "ratio_p10": float(np.percentile(spd_arr, 10)) if spd_arr.size else np.nan,
        "ratio_p90": float(np.percentile(spd_arr, 90)) if spd_arr.size else np.nan,
        "run_ids": run_ids,
        "ilp_times": ilp_times,
        "bnb_times": bnb_times,
        "speedups": speedups,
        "csv_main": str(csv_main),
        "csv_diag": str(csv_diag),
    }
    return summary


# -----------------------------
# Scaling experiment across shapes
# -----------------------------
def run_scaling_experiment(
    shapes: List[Tuple[int, int]],
    num_instances: int = 100,
    out_dir: str = "results",
    base_seed: int = 42,
    tolerance: float = 1e-6,
    do_warmup: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple shapes and collect summaries. Saves a summary CSV and two plots:
      - runtime_comparison_summary.png (avg ILP/BnB vs size)
      - speedup_vs_size.png (ILP/BnB ratio vs node count)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for shape in shapes:
        # (Optional) advance base_seed per shape for cleanliness
        # but we also generate independent seeds inside each per-shape runner.
        summary = run_benchmark_one_shape(
            shape=shape,
            num_instances=num_instances,
            tolerance=tolerance,
            base_seed=base_seed,
            out_dir=str(out_dir),
            do_warmup=do_warmup,
        )
        summaries.append(summary)

    # Write summary CSV
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape", "nodes", "edges", "runs", "mismatches",
            "ilp_avg_s", "bnb_avg_s",
            "ratio_avg", "ratio_median", "ratio_p10", "ratio_p90"
        ])
        for s in summaries:
            sh = s["shape"]
            w.writerow([
                f"{sh[0]}x{sh[1]}",
                s["nodes"], s["edges"], s["runs"], s["mismatches"],
                f"{s['ilp_avg']:.6f}", f"{s['bnb_avg']:.6f}",
                f"{s['ratio_avg']:.6f}", f"{s['ratio_median']:.6f}",
                f"{s['ratio_p10']:.6f}", f"{s['ratio_p90']:.6f}",
            ])

    # Plots
    # 1) Average runtime vs size
    fig1 = plt.figure(figsize=(9, 5.5))
    ax1 = plt.gca()
    sizes = [s["nodes"] for s in summaries]
    ilp_avg = [s["ilp_avg"] for s in summaries]
    bnb_avg = [s["bnb_avg"] for s in summaries]
    labels = [f"{s['shape'][0]}x{s['shape'][1]}" for s in summaries]

    ax1.plot(sizes, ilp_avg, marker='o', label='ILP avg time (s)')
    ax1.plot(sizes, bnb_avg, marker='o', label='BnB avg time (s)')
    for x, txt in zip(sizes, labels):
        ax1.annotate(txt, (x, np.interp(x, sizes, ilp_avg)), xytext=(0, 6), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Number of nodes')
    ax1.set_ylabel('Average runtime (seconds)')
    ax1.set_title('ILP vs BnB Average Runtime vs Problem Size (random cost graphs)')
    ax1.grid(True)
    ax1.legend(loc='best', fontsize=11, frameon=True)
    (out_dir / "runtime_comparison_summary.png").parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(out_dir / "runtime_comparison_summary.png", dpi=160, bbox_inches='tight')
    plt.close(fig1)

    # 2) Speedup (ILP/BnB) vs size
    fig2 = plt.figure(figsize=(9, 5.5))
    ax2 = plt.gca()
    ratio_avg = [s["ratio_avg"] for s in summaries]
    ratio_p10 = [s["ratio_p10"] for s in summaries]
    ratio_p90 = [s["ratio_p90"] for s in summaries]

    ax2.plot(sizes, ratio_avg, marker='o', label='Avg speedup (ILP/BnB)')
    # optional band using p10~p90 to hint variability
    ax2.fill_between(sizes, ratio_p10, ratio_p90, alpha=0.15, label='P10–P90 band')
    for x, txt in zip(sizes, labels):
        ax2.annotate(txt, (x, np.interp(x, sizes, ratio_avg)), xytext=(0, 6), textcoords='offset points', fontsize=8)

    ax2.axhline(1.0, linestyle='--')  # where ILP == BnB
    ax2.set_xlabel('Number of nodes')
    ax2.set_ylabel('Speedup (ILP/BnB)')
    ax2.set_title('Speedup vs Size (values > 1 mean ILP slower than BnB)')
    ax2.grid(True)
    ax2.legend(loc='best', fontsize=11, frameon=True)
    fig2.savefig(out_dir / "speedup_vs_size.png", dpi=160, bbox_inches='tight')
    plt.close(fig2)

    print(f"\nSaved summary CSV: {summary_csv.resolve()}")
    print(f"Saved plots: {(out_dir / 'runtime_comparison_summary.png').resolve()}")
    print(f"            {(out_dir / 'speedup_vs_size.png').resolve()}")

    return summaries


if __name__ == "__main__":
    # You can modify the shapes to push size:
    shapes_to_test = [
        (5, 3),   # 15 nodes
        (8, 4),   # 32 nodes
        (10, 5),  # 50 nodes
        # Add more if needed:
        # (12, 6),  # 72 nodes
        # (14, 7),  # 98 nodes
        # (16, 8), (18, 9), (20, 10),
    ]

    # Run: each shape uses 1 warm-up + num_instances measured runs
    run_scaling_experiment(
        shapes=shapes_to_test,
        num_instances=100,   # per shape (after warm-up)
        out_dir="results",
        base_seed=42,
        tolerance=1e-6,
        do_warmup=True,
    )