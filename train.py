import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
from src.data import generate_data
from src.model import SimpleDenoiser

# --- CONFIG (STABLE + NORMALIZED) ---
CONFIG = {
    'D': 128,             # Высокая размерность
    'K': 5,               # 1 Кластер (Тест на стабильность)
    'N_TRAIN': 1000,      # Достаточно данных
    'SIGMA_DATA': 2.0,    # Изначальный разброс
    'BATCH_SIZE': 128,    # Большой батч = гладкий график
    'LR': 2e-4,           # Аккуратный Learning Rate
    'EPOCHS': 2000,       # Хватит для проверки
    'CHECK_INTERVAL': 100,
    'DEVICE': 'mps' if torch.backends.mps.is_available() else 'cpu'
}

print(f"🚀 Running on {CONFIG['DEVICE']}")

# --- 1. DATA GENERATION & NORMALIZATION ---
# Генерируем "сырые" данные
raw_data, raw_centroids = generate_data(
    CONFIG['N_TRAIN'], CONFIG['D'], CONFIG['K'], CONFIG['SIGMA_DATA'], CONFIG['DEVICE']
)

# Считаем статистику (Mean, Std)
data_mean = raw_data.mean(dim=0)
data_std = raw_data.std(dim=0)

# НОРМАЛИЗАЦИЯ: (x - mu) / sigma
train_data = (raw_data - data_mean) / (data_std + 1e-8)
normalized_centroids = (raw_centroids - data_mean) / (data_std + 1e-8)

dataset = TensorDataset(train_data)
dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)

print(f"Data Normalized. Mean: {train_data.mean().item():.3f}, Std: {train_data.std().item():.3f}")

# --- 2. MODEL & OPTIMIZER ---
model = SimpleDenoiser(input_dim=CONFIG['D'], device=CONFIG['DEVICE'])
optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])

# Diffusion Schedule
num_steps = 1000
betas = torch.linspace(1e-4, 0.02, num_steps).to(CONFIG['DEVICE'])
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def get_loss(model, x_0):
    batch_size = x_0.shape[0]
    t = torch.randint(0, num_steps, (batch_size,), device=CONFIG['DEVICE']).long()
    noise = torch.randn_like(x_0)
    a_bar = alphas_cumprod[t].view(-1, 1)
    x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise
    predicted_noise = model(x_t, t)
    return nn.functional.mse_loss(predicted_noise, noise)

@torch.no_grad()
def sample(model, n_samples):
    model.eval()
    x = torch.randn(n_samples, CONFIG['D'], device=CONFIG['DEVICE'])
    
    for i in reversed(range(0, num_steps)):
        t = torch.full((n_samples,), i, device=CONFIG['DEVICE'], dtype=torch.long)
        pred_noise = model(x, t)
        
        beta_t = betas[i]
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        
        model_mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        
        if i > 0:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(beta_t) * noise
        else:
            x = model_mean
    model.train()
    return x

def compute_metrics(generated, train_data, centroids):
    gen_np = generated.cpu().numpy()
    train_np = train_data.cpu().numpy()
    cent_np = centroids.cpu().numpy()
    
    # Считаем метрики относительно НОРМАЛИЗОВАННЫХ данных
    d2c = [np.min(np.linalg.norm(cent_np - p, axis=1)) for p in gen_np]
    d2t = [np.min(np.linalg.norm(train_np - p, axis=1)) for p in gen_np]
    
    return np.mean(d2c), np.mean(d2t)

# --- 3. TRAIN LOOP ---
history = {'epoch': [], 'gen': [], 'mem': []}
print("Starting training...")

if not os.path.exists('results'):
    os.makedirs('results')

try:
    for epoch in tqdm(range(CONFIG['EPOCHS'])):
        for x_batch in dataloader:
            optimizer.zero_grad()
            loss = get_loss(model, x_batch[0])
            loss.backward()
            optimizer.step()
        
        if epoch % CONFIG['CHECK_INTERVAL'] == 0:
            generated = sample(model, 100)
            gen_score, mem_score = compute_metrics(generated, train_data, normalized_centroids)
            
            history['epoch'].append(epoch)
            history['gen'].append(gen_score)
            history['mem'].append(mem_score)
            
            tqdm.write(f"Ep {epoch}: Gen={gen_score:.2f} | Mem={mem_score:.2f}")

except KeyboardInterrupt:
    print("Stopped manually")

# --- IMPROVED PLOTTING (LOG SCALE) ---
print("Generating Log-Scale Plot...")

# Убираем первые несколько точек, где ошибка космос
start_idx = 1 

if len(history['epoch']) > start_idx:
    epochs = history['epoch'][start_idx:]
    gen_err = history['gen'][start_idx:]
    mem_err = history['mem'][start_idx:]
else:
    epochs = history['epoch']
    gen_err = history['gen']
    mem_err = history['mem']

plt.figure(figsize=(10, 6))

# Рисуем линии
plt.plot(epochs, gen_err, label='Generalization (Dist to Center)', linewidth=3)
plt.plot(epochs, mem_err, label='Memorization (Dist to Train)', linestyle='--', linewidth=2)

# Включаем логарифмическую шкалу по Y
plt.yscale('log') 

plt.title(f"Diffusion Dynamics (Log Scale, K={CONFIG['K']})")
plt.xlabel("Epochs")
plt.ylabel("Distance (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
# Сохраняем под новым именем, чтобы ты сразу увидел
plt.savefig('results/final_result.png')
print("✅ Graph saved to results/final_result.png! Check it now.")
# plt.show()