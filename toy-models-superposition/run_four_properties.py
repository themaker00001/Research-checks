import numpy as np
import matplotlib.pyplot as plt
from superposition import train, feature_dimensionality, gen_batch

rng = np.random.default_rng(42)

# ============================================================
# EXPERIMENT A: LINEARITY probe
# If features are linear directions, then activating feature i
# alone at magnitude x should move H along a straight line,
# H(x) = x * W[:,i], for all x in [0,1].
# ============================================================
n_features, m_hidden = 20, 5
importance = 0.7 ** np.arange(n_features)
S = 0.95
W, b, _ = train(n_features, m_hidden, S, importance, steps=6000, seed=7)

probe_feature = 3
xs = np.linspace(0, 1, 11)
H_actual = []
for xv in xs:
    x = np.zeros(n_features); x[probe_feature] = xv
    H_actual.append(x @ W.T)
H_actual = np.array(H_actual)              # (11, m)
H_predicted = np.outer(xs, W[:, probe_feature])  # x * W_i, the "linear" prediction

max_dev = np.max(np.abs(H_actual - H_predicted))
print(f"[Linearity] max deviation between actual H and x*W_i prediction: {max_dev:.2e}  (should be ~0)")

# ============================================================
# EXPERIMENT B: DECOMPOSABILITY / sparse recovery
# If activations decompose into independent, sparse features,
# we should be able to invert H back to the sparse code X using
# only the learned dictionary W (i.e. do sparse recovery / matching
# pursuit). Recovery quality should degrade as density increases
# and features start interfering with each other.
# ============================================================
def recover_active_features(H, W, threshold=0.1):
    """Greedy matching pursuit: repeatedly pick the feature whose
    direction best explains the residual activation."""
    n = W.shape[1]
    residual = H.copy()
    recovered = np.zeros(n)
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    for _ in range(n):
        scores = Wn.T @ residual
        i = np.argmax(scores)
        if scores[i] < threshold:
            break
        coef = (W[:, i] @ residual) / (W[:, i] @ W[:, i] + 1e-8)
        coef = max(coef, 0)
        recovered[i] = coef
        residual = residual - coef * W[:, i]
    return recovered

def recovery_accuracy(n_features, m_hidden, S, importance, n_test=300, seed=0):
    W, b, _ = train(n_features, m_hidden, S, importance, steps=6000, seed=seed)
    Xtest = gen_batch(n_test, n_features, S, rng)
    hits, total_active = 0, 0
    for x in Xtest:
        h = x @ W.T
        rec = recover_active_features(h, W)
        true_active = set(np.nonzero(x > 0.05)[0])
        rec_active = set(np.nonzero(rec > 0.05)[0])
        hits += len(true_active & rec_active)
        total_active += len(true_active)
    return hits / max(total_active, 1)

sparsities_rec = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
recovery = [recovery_accuracy(n_features, m_hidden, S, importance, seed=11) for S in sparsities_rec]
for S, r in zip(sparsities_rec, recovery):
    print(f"[Decomposability] density={1-S:.3f}  feature-recovery accuracy={r:.3f}")

# ============================================================
# EXPERIMENT C: BASIS ALIGNMENT
# Compare a hidden layer WITHOUT a privileged basis (linear, no
# elementwise nonlinearity) against one WITH a privileged basis
# (ReLU applied to the hidden layer). Measure how tightly learned
# feature directions cluster onto the standard neuron axes, and
# whether that alignment is stable across random seeds.
# ============================================================
def axis_alignment_score(W):
    """For each feature direction (column of W, normalized), take
    the max |cosine similarity| to any standard basis vector.
    1.0 = perfectly aligned with a single neuron; lower = spread
    across multiple neurons (no privileged basis)."""
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
    return np.mean(np.max(np.abs(Wn), axis=0))

n_feat2, m2 = 8, 2
importance2 = np.ones(n_feat2)
S2 = 0.9
seeds = [1, 2, 3]

scores_linear, scores_priv = [], []
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for col, seed in enumerate(seeds):
    W_lin, _, _ = train(n_feat2, m2, S2, importance2, steps=8000, seed=seed, privileged_basis=False)
    W_priv, _, _ = train(n_feat2, m2, S2, importance2, steps=8000, seed=seed, privileged_basis=True)
    scores_linear.append(axis_alignment_score(W_lin))
    scores_priv.append(axis_alignment_score(W_priv))

    for row, (W_, label) in enumerate([(W_lin, "no privileged basis\n(linear hidden layer)"),
                                        (W_priv, "privileged basis\n(ReLU hidden layer)")]):
        ax = axes[row, col]
        circle = plt.Circle((0, 0), 1, fill=False, linestyle=":", color="gray", alpha=0.5)
        ax.add_patch(circle)
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_feat2))
        for i in range(n_feat2):
            vx, vy = W_[0, i], W_[1, i]
            ax.arrow(0, 0, vx, vy, head_width=0.05, length_includes_head=True,
                      color=colors[i], alpha=0.9, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.5); ax.axvline(0, color="black", linewidth=0.5)
        lim = 1.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"seed {seed}\n{label}" if row == 0 else f"seed {seed}", fontsize=9)

axes[0, 0].set_ylabel("no privileged basis", fontsize=11)
axes[1, 0].set_ylabel("privileged basis (ReLU)", fontsize=11)
plt.suptitle("Basis alignment: with a privileged basis, feature directions\n"
             "snap onto the neuron axes (x/y) and stay consistent across seeds", fontsize=13)
plt.tight_layout()
plt.savefig("/home/claude/plot3_basis_alignment.png", dpi=150)
plt.close()

print(f"\n[Basis alignment] mean axis-alignment score, no privileged basis: {np.mean(scores_linear):.3f} (seeds {scores_linear})")
print(f"[Basis alignment] mean axis-alignment score, privileged basis (ReLU): {np.mean(scores_priv):.3f} (seeds {scores_priv})")
print("saved plot3")
