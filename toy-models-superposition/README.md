# Toy Models of Superposition — a from-scratch replication

A minimal NumPy-only reproduction of the core experiments from Anthropic's
[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
(Elhage et al., 2022). No PyTorch, no GPU, no downloaded datasets — everything
is a tied-weight ReLU autoencoder trained on synthetic sparse features, with
gradients implemented by hand.

## The model

```
h   = W @ x                (m,)   compress n features into m < n dimensions
out = relu(Wᵀ @ h + b)     (n,)   reconstruct
loss = Σ_i  I_i · (x_i − out_i)²   (importance-weighted MSE)
```

Each input feature `x_i` is independently zero with probability `sparsity`,
otherwise drawn from `Uniform(0, 1)`.

## What's tested

The paper frames interpretability around four properties. Each script here
targets one:

| Property | Script | Finding |
|---|---|---|
| **Linearity** — are features straight-line directions? | `run_four_properties.py` | Confirmed exactly: activation scales perfectly linearly with feature value. |
| **Decomposability** — can activations be unpacked into independent features? | `run_four_properties.py` | Sparse recovery (matching pursuit) accuracy rises from 0.22 → 0.67 as density drops from 1.0 → 0.001. |
| **Superposition** — more features than dimensions? | `run_dimensionality_experiment.py`, `run_feature_geometry_2d.py` | With m=5 hidden dims, the model represents up to 12/20 features at high sparsity (vs. exactly 5 with no sparsity). |
| **Basis-alignment** — do directions align with individual neurons? | `run_four_properties.py` | Inconclusive with this simple probe — see caveats below. |

## Setup

```bash
pip install -r requirements.txt
bash run_all.sh
```

Figures land in `results/`.

## Files

- `superposition.py` — core model: `train()`, `feature_dimensionality()`, batch generator. Also runs a sparsity sweep and prints summary stats when run directly.
- `run_dimensionality_experiment.py` — reproduces the paper's "dimensionality budget" plot: stacked per-feature D_i values and count of represented features vs. sparsity.
- `run_feature_geometry_2d.py` — 2D visualization of learned feature directions (the classic "polygon" figure), across a sparsity sweep.
- `run_four_properties.py` — linearity probe, sparse-recovery decomposability test, and a privileged-basis vs. no-privileged-basis alignment comparison.

## Key results

- **Superposition is real and measurable at toy scale.** Going from dense (density=1.0) to very sparse (density=0.001) inputs takes the number of "meaningfully represented" features from exactly `m` (5) to more than double that (12), while the total dimensionality budget (Σ D_i) stays conserved at ~m throughout.
- **Feature geometry is visibly non-orthogonal under superposition** — 2D projections show features spreading into symmetric arrangements (antipodal pairs, evenly-spaced configurations) as sparsity increases, consistent with the original paper's polygon/tegum-product figures.
- **Decomposability tracks sparsity directly**, and this is the mechanistic reason superposition doesn't just destroy interpretability: a crude matching-pursuit decoder recovers active features far more reliably as inputs get sparser.

## Caveats

- This is a personal reimplementation for learning/demonstration purposes, not the original paper's code — exact hyperparameters and results will differ.
- The basis-alignment experiment (ReLU vs. linear hidden layer) did **not** cleanly confirm the expected effect at this scale; adding a ReLU to the hidden layer also restricts it to the non-negative orthant, which reshapes the optimization landscape beyond just adding an axis preference. Left in as an honest negative result — a cleaner test would likely need correlated feature structure and more neurons, closer to the original paper's setup.
- No internet/GPU was used — everything trains in well under a minute on CPU.

## Reference

Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread, Anthropic.
