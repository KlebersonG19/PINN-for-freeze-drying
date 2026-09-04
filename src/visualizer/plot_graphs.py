import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
import time


def plot_validation_grid(espessuras_m, pressoes_pa, results_dict, save_image=True):
    n_rows = len(espessuras_m)
    n_cols = len(pressoes_pa)
    
    fig_width = max(5 * n_cols, 8) 
    fig_height = max(4 * n_rows, 5)
    
    # Matriz 2D (squeeze)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True, squeeze=False)
    box_style = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')

    for i, L in enumerate(espessuras_m):
        for j, Pc in enumerate(pressoes_pa):
            ax = axes[i, j]
            
            if (L, Pc) not in results_dict:
                ax.set_visible(False)
                continue
                
            res = results_dict[(L, Pc)]
            
            ax.scatter(res['t_exp'], res['mr_exp'], color='red', s=30, label='Dados Experimentais', zorder=3)
            ax.plot(res['t_continuous'], res['mr_pred'], '-', color='blue', linewidth=2, label='RNIF')

            # Subplots
            ax.set_title(f"L = {L*1000:.0f} mm, $P_c$ = {Pc:.1f} Pa", fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Metrics
            metrics_text = f"RMSE: {res['rmse']:.5f}\n$R^2$: {res['r2']:.5f}"
            ax.text(0.75, 0.80, metrics_text, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='bottom', bbox=box_style)

    # Axis
    for ax in axes[-1, :]: 
        ax.set_xlabel("Tempo (min)", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Razão de Umidade $MR(t)$", fontsize=11)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=12)

    # plt.suptitle("Rede Neural Informada pela Física - Liofilização de Frutas", fontsize=14, y=1.08)
    plt.tight_layout()

    if save_image:
        save_time = str(time.strftime("%y_%m_%d_%H_%M_%S"))
        save_img_path = Path(f"files/images/predictions_{save_time}.png")
        save_img_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico ({n_rows}x{n_cols}) salvo em '{save_img_path}'.")
    
    plt.show()


def plot_custom_scenario(t_continuous, mr_pred, L_mm, Pc_pa, t_target, target_mr, curves_to_compare=None, save_plot=True):
    plt.figure(figsize=(8, 5))
    
    if curves_to_compare:
        for ref in curves_to_compare:
            t_ref = np.asarray(ref['t'])
            mr_ref = np.asarray(ref['mr'])

            idx = np.argsort(t_ref)
            t_ref = t_ref[idx]
            mr_ref = mr_ref[idx]

            # Suavizar curvas de referências (não ficou bom)
            t_smooth = np.linspace(t_ref.min(), t_ref.max(), 200)
            mr_smooth = PchipInterpolator(t_ref, mr_ref)(t_smooth)

            plt.plot(t_smooth, mr_smooth, color='dimgray', linestyle='--', linewidth=1.8, alpha=0.75)
            # ,label=f"Exp. Ref (L = {ref['L']*1000:.0f} mm, $P_c$ = {ref['Pc']} Pa)", zorder=2

    plt.plot(t_continuous, mr_pred, color='darkmagenta', linewidth=2.5, 
             label=f'RNIF (L = {L_mm} mm, $P_c$ = {Pc_pa} Pa)', zorder=4)
    
    plt.axvline(x=t_target, color='darkorange', linestyle='--', alpha=0.8, 
                label=f'Tempo de secagem: {t_target:.1f} min', zorder=3)
    plt.axhline(y=target_mr, color='gray', linestyle=(0, (1, 4)), alpha=0.6, zorder=3, dash_capstyle='round')
    plt.scatter([t_target], [target_mr], color='darkorange', s=80, zorder=5)

    plt.xlabel('Tempo (min)', fontsize=11)
    plt.ylabel('Razão de Umidade $MR(t)$', fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()

    if save_plot:
        save_path = Path(f"files/images/simulations_{L_mm}mm_{Pc_pa}Pa.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
