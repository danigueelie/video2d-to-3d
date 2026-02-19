import numpy as np
from skimage.metrics import structural_similarity as ssim

class MetricTracker:
    def __init__(self):
        self.ssim_scores = []
        self.prev_depth = None
        self.temporal_errors = []

    def update(self, current_frame, current_depth):
        # 1. Stabilité Temporelle (Approximation)
        if self.prev_depth is not None:
            # Calcule la différence moyenne absolue entre deux depth maps consécutives MAE
            diff = np.mean(np.abs(current_depth - self.prev_depth))
            self.temporal_errors.append(diff)

            #calculer le SSIM entre les deux depth maps
            ssim_score = ssim(self.prev_depth, current_depth, data_range=current_depth.max() - current_depth.min())
            self.ssim_scores.append(ssim_score)

        self.prev_depth = current_depth

    def get_summary(self):
        avg_temp_error = np.mean(self.temporal_errors) if self.temporal_errors else 0.0
        avg_ssim = np.mean(self.ssim_scores) if self.ssim_scores else 0.0
        return {
            "temporal_instability": float(avg_temp_error),
            "avg_ssim": float(avg_ssim)
        }
