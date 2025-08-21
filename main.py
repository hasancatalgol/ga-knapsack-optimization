# pip install pygad pandas matplotlib

import numpy as np
import pandas as pd
import pygad

# ---------- simple knobs ----------
CSV_PATH = "data/items.csv"   # put your CSV (Item,Volume_m3,Price) here
CAPACITY = 3.0           # m^3
SEED = 42
# ----------------------------------

# load data
df = pd.read_csv(CSV_PATH)
for col in ("Item", "Volume_m3", "Price"):
    if col not in df.columns:
        raise ValueError(f"CSV must contain columns: Item, Volume_m3, Price (missing: {col})")

names = df["Item"].astype(str).tolist()
volumes = df["Volume_m3"].astype(float).to_numpy()
values  = df["Price"].astype(float).to_numpy()

# fitness
def make_fitness(volumes, values, capacity):
    volumes = np.asarray(volumes, dtype=float)
    values  = np.asarray(values, dtype=float)

    # now with 3 parameters
    def fitness(ga_instance, solution, solution_idx):
        chosen = np.array(solution, dtype=int)
        tot_vol = np.sum(chosen * volumes)
        tot_val = np.sum(chosen * values)

        if tot_vol <= capacity:
            return float(tot_val * (1.0 + (tot_vol / capacity) * 0.01))

        overflow = tot_vol - capacity
        penalty = 1e6 * (overflow ** 2)
        return float(tot_val - penalty)

    return fitness

fitness_func = make_fitness(volumes, values, CAPACITY)

ga = pygad.GA(
    num_generations=800,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=60,
    num_genes=len(names),
    gene_space=[0, 1],
    parent_selection_type="sss",
    keep_parents=4,
    crossover_type="two_points",
    mutation_type="random",
    mutation_percent_genes=12,
    gene_type=int,
    allow_duplicate_genes=True,
    random_seed=SEED,
)

ga.run()

solution, best_fitness, _ = ga.best_solution()
chosen = np.array(solution, dtype=int)

picked = [(names[i], float(volumes[i]), float(values[i])) for i, g in enumerate(chosen) if g == 1]
total_vol = float(np.sum(chosen * volumes))
total_val = float(np.sum(chosen * values))

print("Picked items:")
for name, vol, price in picked:
    print(f"  - {name:20s}  vol={vol:.3f} m^3  price=${price:,.2f}")

print(f"\nTotal volume: {total_vol:.3f} m^3 (capacity {CAPACITY} m^3)")
print(f"Total value:  ${total_val:,.2f}")
print(f"Fitness:      {best_fitness:,.2f}")

# optional: show convergence curve (comment out if you don't want a window)
# ga.plot_fitness()
