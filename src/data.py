import torch

def generate_data(n_samples, d, k, sigma, device='cpu'):
   
    # Центры кластеров 
    centroids = torch.randn(k, d, device=device)
    centroids = torch.nn.functional.normalize(centroids, dim=1) * 5.0 
    
    # Выбор кластера для каждой точки
    cluster_indices = torch.randint(0, k, (n_samples,), device=device)
    
    # Смещение + Шум
    noise = torch.randn(n_samples, d, device=device) * sigma
    data = centroids[cluster_indices] + noise
    
    return data, centroids