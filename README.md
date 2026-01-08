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

- I measure training time as **SGD steps** (optimizer updates), not epochs.
- I log the **memorization fraction** $f_{\mathrm{mem}}(\tau)$ using a 1-NN vs 2-NN ratio criterion.

#### Memorization fraction (kNN ratio)

For a generated sample $x$, let:
- $d_1^2$ be the squared distance to its nearest training point,
- $d_2^2$ be the squared distance to its second-nearest training point.

Define the ratio:
<p align="center"><img src="figures/eq_ratio.svg" width="240"></p>

<p align="center"><img src="figures/eq_threshold.svg" width="260"></p>

<p align="center"><img src="figures/eq_fmem.svg" width="350"></p>

---

## Results (Mac compute-limited)

### Single-run dynamics (Gen/Mem + memorization fraction)
![](figures/dynamics_single_mac.png)

### Scaling of time scales ($\tau_{\mathrm{gen}}$ / $\tau_{\mathrm{mem}}$)
![](figures/scaling_tau_mac.png)

### Collapse plot ($f_{\mathrm{mem}}(\tau)$ vs $\tau/N$)
![](figures/collapse_fmem_mac.png)

Summary table: `artifacts/summary_mac_compute_limited.csv`

> Note: On Apple Silicon / MPS, reaching very late-time memorization for larger $N$ may require more steps than practical locally.  
> In those cases $\tau_{\mathrm{mem}}$ can remain undefined within the current budget (i.e., $\tau_{\mathrm{mem}} > \tau_{\max}$), which should be interpreted as a right-censored estimate.

---

## Repository structure

- `src/data.py` — synthetic GMM data generation
- `src/model.py` — MLP denoiser
- `train.py` — training + sampling + metrics  
  - `--mode single` for a single run  
  - `--mode scaling` for sweeps over dataset size $N$ and seeds  

Outputs are written to `results/` by default (not intended for committing). Final plots/summary for the README live in `figures/` and `artifacts/`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
---
## How to run

### Single run 

Produces one run with:
- Gen/Mem dynamics (log-scale)
- `f_mem(step)` dynamics

```bash
python train.py --mode single \
  --n_train 100 \
  --train_seed 0 --data_seed 0 \
  --max_steps 20000 \
  --eval_every_steps 5000 \
  --n_eval_samples 16 \
  --train_chunk_size 1024 \
  --device auto
```
---

## Scaling sweep

```bash
python train.py --mode scaling \
  --n_list 100 500 1000 \
  --seeds 0 1 2 \
  --max_steps 20000 \
  --eval_every_steps 5000 \
  --n_eval_samples 16 \
  --train_chunk_size 1024 \
  --device auto
```


