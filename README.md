# 🧠 BnB Multicut Solver

This project implements a **Branch and Bound algorithm** to solve the Multicut / Clique Partitioning problem on weighted graphs. It supports comparison against ILP-based solutions and known optimal solutions from the CP-Lib dataset.

---

## 🚀 Installation

Recommended: Python 3.11

```bash
pip install -r requirements.txt
```

Create a virtual environment using `venv` or `conda` for isolation.

---

## 📂 Project Structure

```
bnb_multicut/
├── bnb_solver.py          # Branch-and-Bound implementation
├── ilp_solver.py          # ILP-based baseline solver
├── cp_loader.py           # Load graphs in CP-Lib format
├── graph_generators.py    # Synthetic/random/test graph generators
├── visualizer.py          # Multicut result visualization
├── evaluator.py           # Objective computation and cluster parsing
├── scripts/               # Entry point scripts
│   ├── run_demo.py
│   ├── run_benchmark.py
│   └── run_cp_instance.py
├── cp_lib/                # Folder for CP-Lib benchmark instances
└── main.py                # Optional unified CLI launcher
```

---

## 🔧 Usage

### ▶ Run a demo with random graph + ILP/BnB visualization

```bash
python scripts/run_demo.py
```

### ▶ Run benchmark to verify BnB matches ILP on multiple random graphs

```bash
python scripts/run_benchmark.py
```

### ▶ Solve a CP-Lib instance (e.g., `cars.txt`)

```bash
python scripts/run_cp_instance.py
```

or use unified CLI:

```bash
python main.py --mode cp --instance cp_lib/ABR/cars.txt
```

---

## 📁 CP-Lib Dataset

Clone the full instance dataset from:

📎 https://github.com/MMSorensen/CP-Lib

```bash
git clone https://github.com/MMSorensen/CP-Lib.git cp_lib
```

Make sure the `cp_lib/` folder is at the root of this project.

---

## 🧪 Example Output

```text
[BnB] obj = 964, nodes = 123
Clusters (BnB):
{ 1 2 5 }
{ 3 4 }
...

[OPT] known = 964
[MATCH] ✅ True
Clusters (Optimal):
{ 1 2 5 }
{ 3 4 }
...
```

---

## 📄 License

MIT License or as specified.

---

## 🙋‍♀️ Author

Developed by @annawang for multicut graph clustering experiments and educational purposes. Contributions or collaboration welcome!
