import torch
import numpy as np
from src.utils import load_configs


def hybrid_loss_function(model, X_data, y_data, X_phys):
    cfg = load_configs()

    # Loss weights
    lambda_data = cfg['training']['lambda_data']
    lambda_phys = cfg['training']['lambda_phys']
    lambda_ic = cfg['training']['lambda_ic']

    # Data loss
    MR_data_pred, _ = model(X_data)
    loss_data = torch.mean((MR_data_pred - y_data)**2)
    
    # Physics loss
    X_phys.requires_grad_(True)
    MR_phys_pred, Deff_pred = model(X_phys)
    
    dMR_dt = torch.autograd.grad(
        outputs=MR_phys_pred,
        inputs=X_phys,
        grad_outputs=torch.ones_like(MR_phys_pred),
        create_graph=True
    )[0][:, 0:1]
    
    L_vals = X_phys[:, 1:2]
    
    # Estimate k
    k_sec = (np.pi**2 * Deff_pred) / (L_vals**2)
    k_vals = k_sec * 60.0
    
    f_physics = dMR_dt + k_vals * MR_phys_pred
    loss_physics = torch.mean(f_physics**2)
    
    # IC loss
    X_ic = X_phys.clone()
    X_ic[:, 0] = 0.0
    MR_ic_pred, _ = model(X_ic)
    loss_ic = torch.mean((MR_ic_pred - 1.0)**2)
    
    total_loss = lambda_data * loss_data + lambda_phys * loss_physics + lambda_ic * loss_ic
    return total_loss, loss_data, loss_physics, loss_ic