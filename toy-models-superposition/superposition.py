import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ---------------------------------------------------------------
# Model: tied-weight ReLU autoencoder
#   h   = W @ x                (m,)   -- compress n features into m dims
#   out = relu(W.T @ h + b)    (n,)   -- reconstruct
# Loss: sum_i I_i * (x_i - out_i)^2   -- importance-weighted MSE
# ---------------------------------------------------------------

def gen_batch(batch_size, n_features, sparsity, rng):
    """Each feature is nonzero with prob (1-sparsity), value ~ Uniform(0,1)."""
    values = rng.uniform(0, 1, size=(batch_size, n_features))
    mask = rng.uniform(0, 1, size=(batch_size, n_features)) > sparsity
    return values * mask

def forward(W, b, X):
    # X: (batch, n)
    H = X @ W.T          # (batch, m)
    Z = H @ W + b        # (batch, n)
    out = np.maximum(Z, 0)
    return H, out

def train(n_features, m_hidden, sparsity, importance, steps=8000, batch_size=1024,
          lr=1e-2, seed=0, privileged_basis=False):
    """privileged_basis=True applies a ReLU to the hidden layer too, giving the
    hidden neurons an elementwise nonlinearity (and hence a privileged basis).
    privileged_basis=False (default) keeps the hidden layer purely linear,
    matching the original Anthropic toy model (no privileged basis in H)."""
    rng_local = np.random.default_rng(seed)
    W = rng_local.normal(0, 1 / np.sqrt(n_features), size=(m_hidden, n_features))
    b = np.zeros(n_features)

    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    losses = []
    for t in range(1, steps + 1):
        X = gen_batch(batch_size, n_features, sparsity, rng_local)

        Wpre = X @ W.T                                   # (batch, m) hidden pre-activation
        H = np.maximum(Wpre, 0) if privileged_basis else Wpre
        Z = H @ W + b                                     # (batch, n) output pre-activation
        out = np.maximum(Z, 0)

        err = out - X
        weighted_err = err * importance
        loss = np.mean(np.sum(weighted_err * err, axis=1))
        losses.append(loss)

        relu_mask_out = (out > 0).astype(float)
        g_Z = (2.0 * weighted_err * relu_mask_out) / X.shape[0]     # dL/dZ, (batch, n)
        g_b = g_Z.sum(axis=0)

        g_H = g_Z @ W.T                                    # (batch, m)
        if privileged_basis:
            g_Wpre = g_H * (Wpre > 0).astype(float)
        else:
            g_Wpre = g_H

        gradW = H.T @ g_Z + g_Wpre.T @ X                    # (m, n) total

        for (param, grad, m_, v_) in [(W, gradW, mW, vW), (b, g_b, mb, vb)]:
            m_ *= beta1; m_ += (1 - beta1) * grad
            v_ *= beta2; v_ += (1 - beta2) * (grad ** 2)
            m_hat = m_ / (1 - beta1 ** t)
            v_hat = v_ / (1 - beta2 ** t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return W, b, losses


def feature_dimensionality(W):
    """D_i = ||W_i||^2 / sum_j (W_i . W_j)^2   (from Elhage et al. 2022)
    W columns are feature directions (m, n) -> feature i is W[:, i]."""
    norms_sq = np.sum(W ** 2, axis=0)              # (n,)
    G = W.T @ W                                     # (n, n) gram matrix of feature dirs
    denom = np.sum(G ** 2, axis=0)                  # sum_j (Wi.Wj)^2
    D = norms_sq ** 2 / np.maximum(denom, 1e-12)
    return D, G


if __name__ == "__main__":
    n_features = 20
    m_hidden = 5
    importance = 0.7 ** np.arange(n_features)   # decreasing importance

    sparsities = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
    results = {}
    for S in sparsities:
        W, b, losses = train(n_features, m_hidden, S, importance, steps=6000)
        D, G = feature_dimensionality(W)
        results[S] = dict(W=W, b=b, loss=losses[-1], D=D, G=G)
        print(f"sparsity={S:.3f}  final_loss={losses[-1]:.5f}  "
              f"avg_dimensionality={D.mean():.3f}  "
              f"n_features_with_D>0.3={np.sum(D>0.3)}")

    np.savez("/home/claude/superposition_results.npz",
             sparsities=np.array(sparsities),
             **{f"D_{S}": results[S]["D"] for S in sparsities},
             **{f"G_{S}": results[S]["G"] for S in sparsities})
    print("done")
