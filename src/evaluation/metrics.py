# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# class MetricTracker:
#     def __init__(self):
#         self.ssim_scores = []
#         self.prev_depth = None
#         self.temporal_errors = []

#     def update(self, current_frame, current_depth):
#         # 1. Stabilité Temporelle (Approximation)
#         if self.prev_depth is not None:
#             # Calcule la différence moyenne absolue entre deux depth maps consécutives MAE
#             diff = np.mean(np.abs(current_depth - self.prev_depth))
#             self.temporal_errors.append(diff)

#             #calculer le SSIM entre les deux depth maps
#             ssim_score = ssim(self.prev_depth, current_depth, data_range=current_depth.max() - current_depth.min())
#             self.ssim_scores.append(ssim_score)

#         self.prev_depth = current_depth

#     def get_summary(self):
#         avg_temp_error = np.mean(self.temporal_errors) if self.temporal_errors else 0.0
#         avg_ssim = np.mean(self.ssim_scores) if self.ssim_scores else 0.0
#         return {
#             "temporal_instability": float(avg_temp_error),
#             "avg_ssim": float(avg_ssim)
#         }

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import periodogram
from skimage.metrics import structural_similarity as ssim

class MetricTracker:
    def __init__(self):
        self.temporal_errors = []
        self.mean_depths = []
        self.ssim_scores = []
        self.prev_depth = None

    def update(self, current_depth):
        """Met à jour les signaux pour l'évaluation."""
        # 1. Évolution de la profondeur moyenne
        self.mean_depths.append(np.mean(current_depth))

        # 2. Instabilité temporelle (Différence frame-to-frame)
        if self.prev_depth is not None:
            diff = current_depth - self.prev_depth
            rmse = np.sqrt(np.mean(diff**2))
            self.temporal_errors.append(rmse)

            #calculer le SSIM entre les deux depth maps
            ssim_score = ssim(self.prev_depth, current_depth, data_range=current_depth.max() - current_depth.min())
            self.ssim_scores.append(ssim_score)
        else:
            self.temporal_errors.append(0.0) # Première frame
            self.ssim_scores.append(1.0) # Première frame, SSIM parfait
            
        self.prev_depth = current_depth.copy()

    def compute_psd_flicker(self):
        """
        Calcule l'énergie des hautes fréquences de la profondeur moyenne.
        Un score élevé indique un "scintillement" (flicker) rapide et désagréable.
        """
        if len(self.mean_depths) < 10:
            return 0.0
            
        signal = np.array(self.mean_depths)
        # Retrait de la composante continue (DC)
        signal = signal - np.mean(signal) 
        
        # Calcul du Périodogramme (Densité Spectrale de Puissance)
        freqs, psd = periodogram(signal)
        
        # On somme l'énergie dans la moitié supérieure du spectre (Hautes fréquences)
        mid_point = len(freqs) // 2
        high_freq_power = np.sum(psd[mid_point:])
        
        return float(high_freq_power)

    def get_summary(self):
        if not self.temporal_errors:
            return {"temporal_instability": 0.0, "high_freq_psd": 0.0, "avg_ssim": 0.0}
            
        return {
            "temporal_instability": float(np.mean(self.temporal_errors)),
            "avg_ssim" : float(np.mean(self.ssim_scores)),
            "high_freq_psd": self.compute_psd_flicker()
        }

    def plot_metrics(self, output_dir, filename_prefix):
        """Génère et sauvegarde un graphique de l'évolution des métriques."""
        if len(self.temporal_errors) < 2:
            return None
            
        plt.figure(figsize=(12, 8))

        # Graphique 1 : Instabilité Temporelle (RMSE)
        plt.subplot(2, 1, 1)
        plt.plot(self.temporal_errors, label='Erreur Temporelle (RMSE)', color='red', alpha=0.8)
        plt.axhline(np.mean(self.temporal_errors), color='black', linestyle='--', label='Moyenne')
        plt.title('Évolution de l\'Instabilité Temporelle (Frame par Frame)')
        plt.xlabel('Numéro de Frame')
        plt.ylabel('RMSE')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        # Graphique 2 : Profondeur Moyenne (Évolution de la scène)
        plt.subplot(2, 1, 2)
        plt.plot(self.mean_depths, label='Profondeur Moyenne de la Scène', color='blue', alpha=0.8)
        plt.title('Fluctuations de Profondeur Globale (Analyse du Scintillement)')
        plt.xlabel('Numéro de Frame')
        plt.ylabel('Profondeur (Normalisée)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{filename_prefix}_metrics_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return plot_path