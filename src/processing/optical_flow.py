import cv2
import numpy as np

class OpticalFlowStabilizer:
    def __init__(self, factor=0.3):
        self.factor = factor
        self.prev_gray = None
        self.prev_depth = None

    def reset(self):
        self.prev_gray = None
        self.prev_depth = None

    def warp_flow(self, img, flow):
        h, w = img.shape[:2]
        flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
        # Création de la grille de coordonnées
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # On applique le vecteur de mouvement inverse pour aller chercher le pixel où il était avant
        coords = np.float32(np.dstack([x_coords, y_coords])) + flow
        
        # Remap (Warp) de l'image précédente vers la position actuelle
        warped = cv2.remap(img, coords, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped

    def process(self, frame_bgr, current_depth):
        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = current_gray
            self.prev_depth = current_depth
            return current_depth

        # 1. Calcul du Flot Optique (Farneback - Dense)
        # Calcule le mouvement de chaque pixel entre prev_frame et current_frame
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, current_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # 2. "Warp" de la depth précédente
        # On déforme l'ancienne depth map pour qu'elle s'aligne sur les objets de l'image actuelle
        # Note: Le flow de OpenCV est dx, dy. remap prend map_x, map_y.
        # On utilise une approximation simple ici : flow négatif pour remonter le temps
        h, w = current_depth.shape
        flow = -flow # Inverse flow to map current to prev
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        
        aligned_prev_depth = cv2.remap(self.prev_depth, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # 3. Fusion (EMA intelligent)
        # Maintenant que les pixels sont alignés, on peut moyenner sans faire de ghosting
        stabilized_depth = self.factor * current_depth + (1 - self.factor) * aligned_prev_depth

        # Mise à jour
        self.prev_gray = current_gray
        self.prev_depth = stabilized_depth
        
        return stabilized_depth