# pip install pygad pandas matplotlib

import numpy as np
import pandas as pd
import pygad
import matplotlib.pyplot as plt

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


# ---- history collectors ----
best_gene_history = []        # best chromosome each generation (binary)
selection_rate_history = []   # fraction of 1s per gene each generation (continuous [0,1])

def on_generation(ga_instance):
    # best chromosome (still 0/1)
    best_sol, _, _ = ga_instance.best_solution()
    best_gene_history.append(np.array(best_sol, dtype=int))

    # selection rate across the whole population this generation
    # population shape: (sol_per_pop, num_genes)
    pop = np.array(ga_instance.population, dtype=int)
    selection_rate = pop.mean(axis=0)  # fraction of 1s per gene
    selection_rate_history.append(selection_rate)

ga = pygad.GA(
    num_generations=20,
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
    on_generation=on_generation, 
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
ga.plot_fitness()


import matplotlib.pyplot as plt

# Final chromosome (1xN heatmap)
plt.figure(figsize=(max(8, len(names)*0.6), 2.6))
im = plt.imshow(chosen.reshape(1, -1), aspect="auto")
plt.xticks(range(len(names)), names, rotation=75, ha="right")
plt.yticks([0], ["Solution"])
plt.title("Genes in Final Solution (1 = selected, 0 = not)")
cbar = plt.colorbar(im)                         # legend on the right
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Not selected (0)", "Selected (1)"])
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 1) Best-chromosome evolution (binary per row)
evo_best = np.vstack(best_gene_history)  # shape: generations x genes
plt.figure(figsize=(max(8, len(names)*0.6), 2 + 0.15*len(best_gene_history)))
im1 = plt.imshow(evo_best, aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(names)), names, rotation=75, ha="right")
plt.yticks(
    np.linspace(0, len(best_gene_history)-1, num=min(10, len(best_gene_history))).astype(int),
    np.linspace(1, len(best_gene_history), num=min(10, len(best_gene_history))).astype(int),
)
plt.ylabel("Generation")
plt.title("Best Solution per Generation (genes 0/1)")
cbar1 = plt.colorbar(im1)
cbar1.set_ticks([0, 1])
cbar1.set_ticklabels(["Not selected (0)", "Selected (1)"])
plt.tight_layout()
plt.show()

# 2) Selection-rate evolution (continuous [0,1] → colorful!)
# selection_rate_history collected already
rates = np.vstack(selection_rate_history)  # shape: generations x genes

# transpose so genes are on Y, generations on X
rates_T = rates.T  # shape: genes x generations

import matplotlib.pyplot as plt

plt.figure(figsize=(12, max(4, len(names)*0.4)))
im = plt.imshow(rates_T, aspect="auto", vmin=0, vmax=1, cmap="viridis")

plt.xticks(
    np.linspace(0, rates.shape[0]-1, num=10).astype(int),
    np.linspace(1, rates.shape[0], num=10).astype(int)
)
plt.xlabel("Generation")

plt.yticks(range(len(names)), names)
plt.ylabel("Items")

plt.title("Selection Rate of Each Item Over Generations")
cbar = plt.colorbar(im)
cbar.set_label("Fraction of population with gene = 1")
plt.tight_layout()
plt.show()
