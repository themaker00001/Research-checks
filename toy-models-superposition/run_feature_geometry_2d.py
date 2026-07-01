import numpy as np
import matplotlib.pyplot as plt
from superposition import train

n_features = 8
m_hidden = 2
importance = np.ones(n_features)  # equal importance to see clean polygon geometry
sparsities = [0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, S in enumerate(sparsities):
    W, b, losses = train(n_features, m_hidden, S, importance, steps=8000, seed=3)
    ax = axes[idx]
    circle = plt.Circle((0, 0), 1, fill=False, linestyle=":", color="gray", alpha=0.5)
    ax.add_patch(circle)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_features))
    for i in range(n_features):
        vx, vy = W[0, i], W[1, i]
        ax.arrow(0, 0, vx, vy, head_width=0.05, length_includes_head=True,
                  color=colors[i], alpha=0.9, linewidth=2)
    lim = 1.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"density={1-S:.3f}\nfinal loss={losses[-1]:.4f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Feature geometry in 2 hidden dimensions, 8 equally-important features\n"
             "(each arrow = one feature's learned direction W[:,i])", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/plot2_feature_geometry_2d.png", dpi=150)
plt.close()
print("saved plot2")
