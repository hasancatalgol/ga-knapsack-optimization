# Knapsack Optimization with PyGAD

This project demonstrates solving a **0/1 Knapsack Problem** (a classic optimization problem) using a **Genetic Algorithm (GA)** implemented with the [PyGAD](https://pygad.readthedocs.io/) library.

---

## 📦 Problem Statement

You have a set of items, each with a **volume (m³)** and **price ($)**.  
Your goal is to **maximize the total value** of selected items while keeping the **total volume ≤ 3.0 m³**.

The dataset is stored in [`items.csv`](items.csv) with columns:

- `Item` : Name of the item
- `Volume_m3` : Volume in cubic meters
- `Price` : Price in dollars

This is a version of the **0/1 knapsack problem**, where each item can be either taken (`1`) or not taken (`0`).  
Formally:


**Maximize (objective):**

$$
\text{Total Value in dollars} = \sum_{i=1}^{n} v_i x_i
$$

**Subject to:**

$$
\sum_{i=1}^{n} w_i x_i \le C, \quad x_i \in \{0,1\}
$$

**Where**  
- $v_i$: value of item *i* in **dollars**  
- $w_i$: volume of item *i* in **m³**  
- $C$: capacity (3.0 m³ in this case)  
- $x_i$: decision variable (0/1)  

---

## 🧬 Why Genetic Algorithms?

The knapsack problem is **NP-hard**:  
- Exact algorithms (Dynamic Programming, Integer Linear Programming) exist and are efficient for small/medium sizes.  
- But for **large-scale**, **multi-objective**, or **non-linear** constraints, these methods become less practical.  

**Genetic Algorithms (GAs)** are metaheuristics inspired by natural selection. They don’t guarantee optimality but can find **high-quality solutions** in complex search spaces.

### GA Key Concepts
- **Chromosome**: a candidate solution (binary vector representing selected items).
- **Gene**: each item (0 = excluded, 1 = included).
- **Population**: a collection of chromosomes.
- **Fitness function**: evaluates solution quality (total value, with penalties for overweight).
- **Selection**: chooses parents based on fitness.
- **Crossover**: combines parents to form new solutions.
- **Mutation**: introduces randomness to maintain diversity.
- **Termination**: stop after a number of generations or convergence.

In this repo:
- **Fitness = total item value + small bonus for fuller capacity**.  
- **Penalty** is applied if the solution exceeds the capacity.  
- PyGAD handles selection, crossover, and mutation automatically.

---

## ⚙️ How it Works (in Code)

1. **Load dataset** (`items.csv`) with items, volumes, and prices.  
2. **Define fitness function**:
   - If total volume ≤ 3.0 → reward based on total value.  
   - If total volume > 3.0 → subtract large penalty.  
   - Add a small utilization bonus for near-full truck usage.  
3. **Run GA** with PyGAD:
   - `gene_space=[0,1]` ensures binary encoding.  
   - Runs for 800 generations with mutation and crossover.  
4. **Output** best solution: items selected, total volume, total value, and fitness.

---

## ⚙️ GA Parameters Explained

Here are the key parameters used in this project’s GA configuration:

- **`num_generations=800`** → number of evolutionary cycles to run. More generations allow better convergence but take longer.  
- **`sol_per_pop=60`** → population size (number of candidate solutions per generation). Larger = more diversity, but more computation.  
- **`num_genes=len(names)`** → chromosome length, equal to the number of items in the dataset.  

### Genes & Representation
- **`gene_space=[0, 1]`** → restricts each gene to 0/1 (don’t take / take item).  
- **`gene_type=int`** → ensures genes are stored as integers.  
- **`allow_duplicate_genes=True`** → irrelevant for this binary problem, but required for consistency.  

### Fitness
- **`fitness_func=fitness_func`** → evaluates solutions based on total value, with penalties for exceeding capacity.  

### Parent Selection & Reproduction
- **`num_parents_mating=20`** → number of parents selected each generation.  
- **`parent_selection_type="sss"`** → “stochastic universal sampling”, a fair selection proportional to fitness.  
- **`keep_parents=4`** → elitism: best solutions are carried forward unchanged.  

### Crossover & Mutation
- **`crossover_type="two_points"`** → two crossover points swap genes between parents.  
- **`mutation_type="random"`** → randomly flips genes.  
- **`mutation_percent_genes=12`** → percentage of genes mutated in each offspring (≈12%).  

### Reproducibility
- **`random_seed=SEED`** → ensures results are repeatable across runs.  

---

## 🚀 Running the Code

### 1. Install dependencies
```bash
uv add pygad pandas matplotlib
```

### 2. Run the script
```bash
uv run main.py
```

### 3. Example Output
```
Picked items:
  - Refrigerator A        vol=0.751 m^3  price=$999.90
  - Notebook A            vol=0.004 m^3  price=$2,499.90
  ...

Total volume: 2.798 m^3 (capacity 3.0 m^3)
Total value:  $24,082.46
Fitness:      24,307.09
```

---

## 📂 Repo Structure
```
.
├── items.csv        # Input dataset
├── main.py          # PyGAD solution
└── README.md        # Project documentation
```

---

## 🔬 Theoretical Comparison

- **Exact Methods (OR):**
  - Integer Linear Programming (ILP) can solve this problem optimally in milliseconds for small item counts.
  - Guarantees **optimal solution**.
- **Heuristic Methods (GA):**
  - Useful when the problem grows large, has multiple constraints, or objectives are not linear.
  - Provides **near-optimal solution** quickly, without complex modeling.
  - Flexible to adapt to new rules (e.g., category constraints, synergies).

---

## 📈 Extensions
- Add an **OR baseline** using PuLP/OR-Tools for comparison.  
- Explore **multi-objective optimization** (maximize value, minimize unused volume, diversify item categories).  
- Experiment with **different crossover/mutation strategies** in GA.  
- Scale dataset with 100s or 1000s of items to test GA’s robustness.  
