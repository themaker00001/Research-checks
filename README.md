# Research

A personal sandbox for working through influential ML/AI research papers —
not by summarizing them, but by reimplementing their core experiments from
scratch and checking whether we get the same results, and understand why.

Each subdirectory picks one paper and builds the smallest possible version of
its central experiment, so the underlying idea can be verified and explored
directly instead of taken on faith.

## Papers covered

| Paper | Directory | Status |
|---|---|---|
| [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage et al., 2022) | [toy-models-superposition/](toy-models-superposition/) | Done |
| [AutoRedTrader: Autonomous Red Teaming of Trading Agents through Synthetic Misinformation Injection](https://arxiv.org/abs/2605.09185) (arXiv:2605.09185) | [AutoRedTrader/](AutoRedTrader/) | Done |

## Structure

Each paper directory is self-contained: its own README, requirements, and
runnable scripts that reproduce the paper's key figures/claims. See the
directory's README for details and results.
