import os
import json
import csv
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.data import generate_data
from src.model import SimpleDenoiser



# Config

@dataclass
class ExpConfig:
    # data
    D: int = 128
    K: int = 5
    SIGMA_DATA: float = 2.0

    # optimization
    BATCH_SIZE: int = 128
    LR: float = 2e-4
    MAX_STEPS: int = 20000                 # SGD steps
    EVAL_EVERY_STEPS: int = 1000

    # diffusion
    NUM_STEPS: int = 1000                  # diffusion timesteps
    N_EVAL_SAMPLES: int = 100              # samples generated at each eval

    # memorization fraction
    MEM_K_RATIO: float = 1.0 / 3.0         # k in ratio criterion
    MEM_THR: float = 0.05                  # threshold to define tau_mem
    MEM_CONSECUTIVE: int = 2               # require f_mem >= thr for N evals in a row
    TRAIN_CHUNK_SIZE: int = 2048           # chunk for NN distances to avoid memory spikes

    # reproducibility 
    DEVICE: str = "auto"
    OUT_DIR: str = "results"
    RUN_NAME: str = ""
    TRAIN_SEED: int = 0
    DATA_SEED: int = 0                     # separate seed for dataset generation

    # scaling sweep
    N_TRAIN: int = 1000


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device_str


def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_dir(base_out_dir: str, group: str, run_name: str) -> Path:
    run_dir = Path(base_out_dir) / group / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch



# Diffusion schedule / loss / sampling

def make_schedule(num_steps: int, device: str):
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def get_loss(model, x_0, num_steps, alphas_cumprod, device):
    batch_size = x_0.shape[0]
    t = torch.randint(0, num_steps, (batch_size,), device=device).long()
    noise = torch.randn_like(x_0)
    a_bar = alphas_cumprod[t].view(-1, 1)
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1.0 - a_bar) * noise
    predicted_noise = model(x_t, t)
    return nn.functional.mse_loss(predicted_noise, noise)


@torch.no_grad()
def sample(model, n_samples, D, num_steps, betas, alphas, alphas_cumprod, device):
    model.eval()
    x = torch.randn(n_samples, D, device=device)

    for i in reversed(range(0, num_steps)):
        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        pred_noise = model(x, t)

        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]

        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)

        if i > 0:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(beta_t) * noise
        else:
            x = model_mean

    model.train()
    return x



# Fast distances robust fallback if MPS cdist is problematic

@torch.no_grad()
def safe_cdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    
    try:
        return torch.cdist(a, b)
    except Exception:
        a_cpu = a.detach().cpu()
        b_cpu = b.detach().cpu()
        d_cpu = torch.cdist(a_cpu, b_cpu)
        return d_cpu.to(a.device)


@torch.no_grad()
def compute_gen_mem_errors(generated, train_data, centroids, train_chunk_size: int = 2048):
    # Gen error: min distance to centroids
    d_cent = safe_cdist(generated, centroids)         
    gen = d_cent.min(dim=1).values.mean().item()

    # Mem error: min distance to training points
    n_gen = generated.shape[0]
    best = torch.full((n_gen,), float("inf"), device=generated.device)

    n_train = train_data.shape[0]
    for start in range(0, n_train, train_chunk_size):
        end = min(start + train_chunk_size, n_train)
        chunk = train_data[start:end]
        d = safe_cdist(generated, chunk)               # [n_gen, chunk]
        best = torch.minimum(best, d.min(dim=1).values)

    mem = best.mean().item()
    return float(gen), float(mem)


@torch.no_grad()
def compute_memorization_fraction(
    generated: torch.Tensor,
    train_data: torch.Tensor,
    k_ratio: float = 1/3,
    train_chunk_size: int = 2048,
    eps: float = 1e-12,
):
    
    n_gen = generated.shape[0]
    best1 = torch.full((n_gen,), float("inf"), device=generated.device)
    best2 = torch.full((n_gen,), float("inf"), device=generated.device)

    n_train = train_data.shape[0]
    for start in range(0, n_train, train_chunk_size):
        end = min(start + train_chunk_size, n_train)
        chunk = train_data[start:end]

        # squared distances
        d2 = safe_cdist(generated, chunk) ** 2  # [n_gen, chunk]

        # top-2 smallest within chunk
        
        v, _ = torch.topk(d2, k=2, dim=1, largest=False)

        merged = torch.stack([best1, best2, v[:, 0], v[:, 1]], dim=1)
        newv, _ = torch.topk(merged, k=2, dim=1, largest=False)
        best1, best2 = newv[:, 0], newv[:, 1]

    ratio = best1 / (best2 + eps)
    f_mem = (ratio < k_ratio).float().mean().item()
    ratio_mean = ratio.mean().item()
    return float(f_mem), float(ratio_mean)


def gap_log10(gen: float, mem: float, eps: float = 1e-12) -> float:
    return float(np.log10(gen + eps) - np.log10(mem + eps))



# tau extraction 

def extract_tau_gen(steps, gen, delta: float = 0.05):
    
    if len(gen) == 0:
        return None
    gmin = float(np.min(gen))
    thr = (1.0 + delta) * gmin
    for s, g in zip(steps, gen):
        if g <= thr:
            return int(s)
    return None


def extract_tau_mem(steps, f_mem, thr: float = 0.05, consecutive: int = 2):
    
    cnt = 0
    for s, fm in zip(steps, f_mem):
        if fm >= thr:
            cnt += 1
            if cnt >= consecutive:
                return int(s)
        else:
            cnt = 0
    return None



# plotting

def plot_dynamics(history, out_path: Path, title: str):
    steps = history["step"]
    gen = history["gen"]
    mem = history["mem"]
    fmem = history["f_mem"]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(steps, gen, label="Gen (dist to centroid)", linewidth=2)
    ax[0].plot(steps, mem, label="Mem (dist to train)", linewidth=2, linestyle="--")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Distance (log)")
    ax[0].grid(True, which="both", alpha=0.25)
    ax[0].legend()

    ax[1].plot(steps, fmem, label="f_mem (NN ratio)", linewidth=2)
    ax[1].set_xlabel("Training steps (SGD updates)")
    ax[1].set_ylabel("Memorization fraction")
    ax[1].grid(True, alpha=0.25)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_scaling_tau(rows, out_path: Path, title: str):
    """
    rows: list of dict with keys N, seed, tau_gen, tau_mem
    """
    byN = {}
    for r in rows:
        byN.setdefault(int(r["N"]), []).append(r)

    Ns = sorted(byN.keys())

    tau_gen_mean, tau_gen_std = [], []
    tau_mem_mean, tau_mem_std = [], []

    for N in Ns:
        tg = np.array([r["tau_gen_steps"] for r in byN[N] if r["tau_gen_steps"] != ""], dtype=float)
        tm = np.array([r["tau_mem_steps"] for r in byN[N] if r["tau_mem_steps"] != ""], dtype=float)

        # if missing (e.g., tau_mem not reached), keep NaN
        tau_gen_mean.append(np.nan if tg.size == 0 else tg.mean())
        tau_gen_std.append(np.nan if tg.size == 0 else tg.std(ddof=0))

        tau_mem_mean.append(np.nan if tm.size == 0 else tm.mean())
        tau_mem_std.append(np.nan if tm.size == 0 else tm.std(ddof=0))

    plt.figure(figsize=(10, 6))
    plt.errorbar(Ns, tau_gen_mean, yerr=tau_gen_std, fmt="o-", capsize=4, label="tau_gen (steps)")
    plt.errorbar(Ns, tau_mem_mean, yerr=tau_mem_std, fmt="o-", capsize=4, label="tau_mem (steps)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dataset size N (log)")
    plt.ylabel("Steps (log)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_collapse_fmem(curves, out_path: Path, title: str):
    """
    curves: list of dicts: {N, seed, steps: [...], f_mem: [...]}
    Plot f_mem vs steps/N to see collapse (paper-style idea).
    """
    plt.figure(figsize=(10, 6))
    for c in curves:
        N = c["N"]
        steps = np.array(c["steps"], dtype=float)
        fmem = np.array(c["f_mem"], dtype=float)
        x = steps / float(N)
        plt.plot(x, fmem, linewidth=2, alpha=0.8, label=f"N={N}, seed={c['seed']}")

    plt.xlabel("steps / N")
    plt.ylabel("f_mem")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



# data prep

def prepare_data_once(cfg: ExpConfig, N: int, device: str):
    
    set_all_seeds(cfg.DATA_SEED)
    raw_data, raw_centroids = generate_data(N, cfg.D, cfg.K, cfg.SIGMA_DATA, device)

    data_mean = raw_data.mean(dim=0)
    data_std = raw_data.std(dim=0)

    train_data = (raw_data - data_mean) / (data_std + 1e-8)
    centroids = (raw_centroids - data_mean) / (data_std + 1e-8)
    return train_data, centroids



# single run

def run_one_training(cfg: ExpConfig, group: str, train_data: torch.Tensor, centroids: torch.Tensor) -> dict:
    device = cfg.DEVICE

    dataset = TensorDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    it = infinite_loader(dataloader)

    model = SimpleDenoiser(input_dim=cfg.D, device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)

    betas, alphas, alphas_cumprod = make_schedule(cfg.NUM_STEPS, device=device)

    history = {"step": [], "gen": [], "mem": [], "gap": [], "f_mem": [], "ratio_mean": []}

    t0 = time.time()
    pbar = tqdm(total=cfg.MAX_STEPS, desc=f"train N={cfg.N_TRAIN} seed={cfg.TRAIN_SEED}")
    global_step = 0

    while global_step < cfg.MAX_STEPS:
        x = next(it)[0]
        optimizer.zero_grad(set_to_none=True)
        loss = get_loss(model, x, cfg.NUM_STEPS, alphas_cumprod, device)
        loss.backward()
        optimizer.step()

        global_step += 1
        pbar.update(1)

        if global_step % cfg.EVAL_EVERY_STEPS == 0 or global_step == cfg.MAX_STEPS:
            generated = sample(
                model=model,
                n_samples=cfg.N_EVAL_SAMPLES,
                D=cfg.D,
                num_steps=cfg.NUM_STEPS,
                betas=betas,
                alphas=alphas,
                alphas_cumprod=alphas_cumprod,
                device=device,
            )

            gen, mem = compute_gen_mem_errors(
                generated, train_data, centroids, train_chunk_size=cfg.TRAIN_CHUNK_SIZE
            )
            f_mem, ratio_mean = compute_memorization_fraction(
                generated, train_data,
                k_ratio=cfg.MEM_K_RATIO,
                train_chunk_size=cfg.TRAIN_CHUNK_SIZE
            )
            gap = gap_log10(gen, mem)

            history["step"].append(int(global_step))
            history["gen"].append(float(gen))
            history["mem"].append(float(mem))
            history["gap"].append(float(gap))
            history["f_mem"].append(float(f_mem))
            history["ratio_mean"].append(float(ratio_mean))

            pbar.set_postfix(gen=f"{gen:.3f}", mem=f"{mem:.3f}", gap=f"{gap:.3f}", f_mem=f"{f_mem:.3f}")

    pbar.close()
    wall = time.time() - t0

    # extract taus
    steps_np = np.array(history["step"], dtype=int)
    gen_np = np.array(history["gen"], dtype=float)
    fmem_np = np.array(history["f_mem"], dtype=float)

    tau_gen = extract_tau_gen(steps_np, gen_np, delta=0.05)
    tau_mem = extract_tau_mem(steps_np, fmem_np, thr=cfg.MEM_THR, consecutive=cfg.MEM_CONSECUTIVE)

    # save run
    if cfg.RUN_NAME == "":
        cfg.RUN_NAME = f"N_{cfg.N_TRAIN:05d}_trainseed_{cfg.TRAIN_SEED}_dataseed_{cfg.DATA_SEED}"

    run_dir = make_run_dir(cfg.OUT_DIR, group=group, run_name=cfg.RUN_NAME)

    meta = {
        "torch_version": torch.__version__,
        "device": cfg.DEVICE,
        "mps_available": bool(torch.backends.mps.is_available()),
    }
    (run_dir / "config.json").write_text(json.dumps({"config": asdict(cfg), "meta": meta}, indent=2))

    np.savez(
        run_dir / "history.npz",
        step=np.array(history["step"], dtype=np.int32),
        gen=np.array(history["gen"], dtype=np.float64),
        mem=np.array(history["mem"], dtype=np.float64),
        gap=np.array(history["gap"], dtype=np.float64),
        f_mem=np.array(history["f_mem"], dtype=np.float64),
        ratio_mean=np.array(history["ratio_mean"], dtype=np.float64),
    )

    plot_dynamics(
        history,
        out_path=run_dir / "dynamics.png",
        title=f"Dynamics | N={cfg.N_TRAIN} | train_seed={cfg.TRAIN_SEED} | data_seed={cfg.DATA_SEED}"
    )

    # final metrics
    gen_final = history["gen"][-1]
    mem_final = history["mem"][-1]
    gap_final = history["gap"][-1]
    fmem_final = history["f_mem"][-1]

    metrics = {
        "N": int(cfg.N_TRAIN),
        "train_seed": int(cfg.TRAIN_SEED),
        "data_seed": int(cfg.DATA_SEED),
        "gen_final": float(gen_final),
        "mem_final": float(mem_final),
        "gap_log10_final": float(gap_final),
        "f_mem_final": float(fmem_final),
        "tau_gen_steps": "" if tau_gen is None else int(tau_gen),
        "tau_mem_steps": "" if tau_mem is None else int(tau_mem),
        "wall_time_sec": float(wall),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    curve = {"N": int(cfg.N_TRAIN), "seed": int(cfg.TRAIN_SEED), "steps": history["step"], "f_mem": history["f_mem"]}
    return metrics, curve


def save_summary_csv(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)



# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "scaling"])

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--out_dir", type=str, default="results")

    # data
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--sigma_data", type=float, default=2.0)

    # diffusion
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--n_eval_samples", type=int, default=100)

    # optimization (steps-based)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_every_steps", type=int, default=1000)

    # memorization fraction params
    parser.add_argument("--mem_k_ratio", type=float, default=1/3)
    parser.add_argument("--mem_thr", type=float, default=0.05)
    parser.add_argument("--mem_consecutive", type=int, default=2)
    parser.add_argument("--train_chunk_size", type=int, default=2048)

    # single
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)

    # scaling
    parser.add_argument("--n_list", type=int, nargs="*", default=[100, 500, 1000, 5000, 10000])
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--nested_data", type=int, default=1)  # 1 = generate Nmax once per seed, take prefixes

    args = parser.parse_args()

    cfg = ExpConfig(
        D=args.D,
        K=args.K,
        SIGMA_DATA=args.sigma_data,
        BATCH_SIZE=args.batch_size,
        LR=args.lr,
        MAX_STEPS=args.max_steps,
        EVAL_EVERY_STEPS=args.eval_every_steps,
        NUM_STEPS=args.num_steps,
        N_EVAL_SAMPLES=args.n_eval_samples,
        MEM_K_RATIO=args.mem_k_ratio,
        MEM_THR=args.mem_thr,
        MEM_CONSECUTIVE=args.mem_consecutive,
        TRAIN_CHUNK_SIZE=args.train_chunk_size,
        DEVICE=resolve_device(args.device),
        OUT_DIR=args.out_dir,
        TRAIN_SEED=args.train_seed,
        DATA_SEED=args.data_seed,
        N_TRAIN=args.n_train,
    )

    Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Running on device: {cfg.DEVICE}")

    if args.mode == "single":
        # training seed affects model init + SGD noise
        set_all_seeds(cfg.TRAIN_SEED)

        train_data, centroids = prepare_data_once(cfg, N=cfg.N_TRAIN, device=cfg.DEVICE)
        cfg.RUN_NAME = f"N_{cfg.N_TRAIN:05d}_trainseed_{cfg.TRAIN_SEED}_dataseed_{cfg.DATA_SEED}"
        metrics, _curve = run_one_training(cfg, group="single", train_data=train_data, centroids=centroids)
        print(json.dumps(metrics, indent=2))
        return

    # scaling
    all_rows = []
    all_curves = []
    sweep_dir = Path(cfg.OUT_DIR) / "scaling"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    n_list = list(map(int, args.n_list))
    Nmax = max(n_list)

    for seed in args.seeds:
        # each seed defines both data and training seeds unless you want to decouple
        train_seed = int(seed)
        data_seed = int(seed)

        cfg.TRAIN_SEED = train_seed
        cfg.DATA_SEED = data_seed

        set_all_seeds(cfg.TRAIN_SEED)

        if args.nested_data == 1:
            # generate Nmax once per seed, normalize once, then take prefixes for each N
            train_full, centroids = prepare_data_once(cfg, N=Nmax, device=cfg.DEVICE)
        else:
            train_full, centroids = None, None  # regenerated per N below

        for N in n_list:
            cfg.N_TRAIN = int(N)
            cfg.RUN_NAME = f"N_{cfg.N_TRAIN:05d}_trainseed_{cfg.TRAIN_SEED}_dataseed_{cfg.DATA_SEED}"

            if args.nested_data == 1:
                train_data = train_full[:N]
                # centroids fixed
                metrics, curve = run_one_training(cfg, group="scaling", train_data=train_data, centroids=centroids)
            else:
                train_data, centroidsN = prepare_data_once(cfg, N=N, device=cfg.DEVICE)
                metrics, curve = run_one_training(cfg, group="scaling", train_data=train_data, centroids=centroidsN)

            all_rows.append(metrics)
            all_curves.append(curve)

    summary_csv = sweep_dir / "summary.csv"
    save_summary_csv(all_rows, summary_csv)

    # plots: tau scaling + collapse
    plot_scaling_tau(
        all_rows,
        out_path=sweep_dir / "scaling_tau.png",
        title=f"Scaling of tau_gen / tau_mem | D={cfg.D}, K={cfg.K}, mem_thr={cfg.MEM_THR}"
    )
    plot_collapse_fmem(
        all_curves,
        out_path=sweep_dir / "collapse_fmem.png",
        title="Collapse plot: f_mem vs steps/N"
    )

    print(f"Saved: {summary_csv}")
    print(f"Saved: {sweep_dir / 'scaling_tau.png'}")
    print(f"Saved: {sweep_dir / 'collapse_fmem.png'}")


if __name__ == "__main__":
    main()