import numpy as np
import matplotlib.pyplot as plt
from superposition import train, feature_dimensionality, gen_batch

# ============================================================
# PLOT 1: Feature dimensionality spectrum vs sparsity
#         (reproduces the "stacked bar" plot from the paper)
# ============================================================
n_features = 20
m_hidden = 5
importance = 0.7 ** np.arange(n_features)
sparsities = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

Ds = []
n_used = []
final_losses = []
for S in sparsities:
    W, b, losses = train(n_features, m_hidden, S, importance, steps=6000, seed=1)
    D, G = feature_dimensionality(W)
    Ds.append(np.sort(D)[::-1])   # sort descending for clean stacked bars
    n_used.append(np.sum(D > 0.25))
    final_losses.append(losses[-1])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Stacked bar: for each sparsity level, stack the D_i (sorted) as a bar of total height
ax = axes[0]
x = np.arange(len(sparsities))
cmap = plt.get_cmap("tab20")
bottoms = np.zeros(len(sparsities))
for feat_rank in range(n_features):
    heights = [Ds[s][feat_rank] for s in range(len(sparsities))]
    ax.bar(x, heights, bottom=bottoms, width=0.6, color=cmap(feat_rank % 20), edgecolor="white", linewidth=0.3)
    bottoms += heights
ax.axhline(m_hidden, color="black", linestyle="--", linewidth=1, label=f"m = {m_hidden} (hidden dims)")
ax.set_xticks(x)
ax.set_xticklabels([f"{1-s:.3f}" if s < 0.999 else "0.001" for s in sparsities], rotation=45)
ax.set_xlabel("feature density (1 - sparsity)")
ax.set_ylabel("sum of per-feature dimensionality  Σ D_i")
ax.set_title("Total 'dimensionality budget' used\n(bars stack the 20 individual D_i values)")
ax.legend()

# Number of features with a meaningfully-used direction (D_i > 0.25) vs sparsity
ax = axes[1]
density = [1 - s for s in sparsities]
ax.plot(density, n_used, marker="o", color="crimson")
ax.axhline(m_hidden, color="black", linestyle="--", linewidth=1, label=f"m = {m_hidden} (hidden dims)")
ax.set_xscale("log")
ax.set_xlabel("feature density (1 - sparsity), log scale")
ax.set_ylabel("# features with D_i > 0.25")
ax.set_title("Superposition: more features represented\nthan hidden dimensions as sparsity increases")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/plot1_dimensionality_vs_sparsity.png", dpi=150)
plt.close()
print("saved plot1")
print("n_used per sparsity:", list(zip(sparsities, n_used)))
print("final losses:", list(zip(sparsities, [round(l,5) for l in final_losses])))
