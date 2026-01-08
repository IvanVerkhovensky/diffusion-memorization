# Why Diffusion Models Don't Memorize (NeurIPS 2025) — Replication in PyTorch (Apple Silicon / MPS)

I’m replicating and extending experiments inspired by the NeurIPS 2025 paper:

**“Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization”**

This repo focuses on diffusion training dynamics on **synthetic high‑dimensional Gaussian mixture data**, with a clean split between **generalization** and **memorization** and a paper-aligned measurement protocol (**time in SGD steps** + **memorization fraction**).

---

## What I’m trying to show

Diffusion models can go through a **generalization-first regime** (they learn the underlying data structure) before a later-time **memorization regime** (generated samples become unusually close to specific training points).  
My goal is to make this behavior measurable, reproducible, and easy to run on a Mac (PyTorch MPS).

---

## What’s implemented

### Phase 1 — Synthetic dynamics: Generalization vs Memorization gap
I train a DDPM-like denoiser (MLP) on a high-dimensional GMM (default `D=128`) and track:

- **Generalization error**: distance from generated samples to the nearest true cluster centroid.
- **Memorization error**: distance from generated samples to the nearest training example.

A persistent separation between these curves corresponds to a “generalization-first” regime.

### Phase 2 — Paper-style protocol: two timescales + memorization fraction
To match the paper’s experimental logic more closely:

- I measure training time as **SGD steps (optimizer updates)**, not epochs.
- I log **memorization fraction** `f_mem(step)` using a 1-NN vs 2-NN ratio criterion.

**Memorization fraction definition (kNN ratio)**  
For a generated sample `x`, let `d1^2` be the squared distance to its nearest training point and `d2^2` to its second nearest.  
Define `r = d1^2 / (d2^2 + eps)`.  
A sample is considered “memorized” if `r < k` (default `k = 1/3`).  
Then `f_mem` is the fraction of generated samples classified as memorized.

From the curves I extract:
- `tau_gen`: step when generalization stabilizes (near-minimum gen error).
- `tau_mem`: step when memorization fraction crosses a threshold and stays above it.

I also generate a paper-style “collapse plot”:
- `f_mem` vs `steps / N`.

---

## Results (Mac compute-limited)

### Single-run dynamics (Gen/Mem + memorization fraction)
![](figures/dynamics_single_mac.png)

### Scaling of time scales (tau_gen / tau_mem)
![](figures/scaling_tau_mac.png)

### Collapse plot (f_mem vs steps/N)
![](figures/collapse_fmem_mac.png)

Summary table: `artifacts/summary_mac_compute_limited.csv`

> Note: On Apple Silicon / MPS, reaching very late-time memorization for larger `N` may require more steps than practical locally.  
> In those cases `tau_mem` can remain undefined within the current budget (`tau_mem > max_steps`), which should be interpreted as a right-censored estimate.

---

## Repository structure

- `src/data.py` — synthetic GMM data generation
- `src/model.py` — MLP denoiser
- `train.py` — training + sampling + metrics  
  - `--mode single` for a single run  
  - `--mode scaling` for sweeps over dataset size `N` and seeds  

Outputs are written to `results/` by default (not intended for committing).

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
