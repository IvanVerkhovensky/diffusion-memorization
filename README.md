# Why Diffusion Models Don't Memorize (NeurIPS 2025) — Replication + PyTorch (Apple Silicon / MPS)

Unofficial, research-oriented implementation inspired by the NeurIPS 2025 paper:

**“Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization”**

This repo reproduces and analyzes diffusion training behavior on **synthetic high-dimensional Gaussian mixture data**, with a focus on clean separation between **generalization** and **memorization**, and a paper-aligned measurement protocol (time in **SGD steps** + memorization fraction).

---

## Project goal

Show that diffusion models can exhibit a **generalization-first regime** (learn the distribution structure) before entering a later-time **memorization regime** (generated samples become unusually close to training points), and provide a reproducible experimental pipeline to study this behavior under different dataset sizes.

---

## What’s implemented

### Phase 1 — Synthetic dynamics: Gen vs Mem gap
We train a DDPM-like denoiser (MLP) on a high-dimensional GMM (default `D=128`) and track:

- **Generalization error**: distance from generated samples to the nearest true cluster centroid.
- **Memorization error**: distance from generated samples to the nearest training example.

A persistent separation between these curves corresponds to a “generalization-first” regime.

### Phase 2 — Paper-style protocol: two timescales + memorization fraction
To match the paper’s experimental logic more closely:

- We measure training time as **SGD steps (optimizer updates)**, not epochs.
- We log **memorization fraction** `f_mem(step)` using a 1-NN vs 2-NN ratio criterion.

**Memorization fraction definition (kNN ratio):**
For a generated sample `x`, let `d1^2` be the squared distance to its nearest training point, and `d2^2` to its second nearest.  
Define `r = d1^2 / (d2^2 + eps)`.  
A sample is considered “memorized” if `r < k` (default `k = 1/3`).  
Then `f_mem` is the fraction of generated samples classified as memorized.

From these curves we extract:
- `tau_gen`: step when generalization stabilizes (near-minimum gen error).
- `tau_mem`: step when memorization fraction crosses a threshold and stays above it.

We also produce a “collapse plot” in the spirit of the paper:
- `f_mem` vs `steps / N`.

---

## Repository structure

- `src/data.py` — synthetic GMM data generation
- `src/model.py` — MLP denoiser
- `train.py` — training + sampling + metrics
  - `--mode single` for a single run
  - `--mode scaling` for sweeps over dataset size `N` and seeds

Outputs are written to `results/` by default.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```