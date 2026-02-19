import cv2
import numpy as np

class StereoGen:
    def __init__(self, divergence=2.0, convergence=0.5):
        self.divergence = divergence / 100.0 
        self.convergence = convergence

    def normalize_depth(self, depth):
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return depth
        return (depth - d_min) / (d_max - d_min)

    def _get_latitude_correction(self, h, w):
        """
        Pour le VR360, la disparité doit diminuer à mesure qu'on approche des pôles.
        Sinon, on obtient un artefact de "pincement" au zénith et au nadir.
        Retourne une map de coefficients (0.0 aux pôles, 1.0 à l'équateur).
        """
        # On génère une map de latitudes (de -PI/2 à +PI/2)
        y_grid = np.linspace(-np.pi/2, np.pi/2, h)
        # La correction est le cosinus de la latitude
        correction = np.cos(y_grid)
        # On étend cette colonne à toute la largeur de l'image
        return np.tile(correction[:, np.newaxis], (1, w))

    def _build_vr180_map(self, h, w):
        """Projection simple pour simuler un fisheye VR180"""
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        theta = x * (np.pi / 2) 
        phi = y * (np.pi / 2)   
        
        # Limite pour éviter l'infini
        valid_mask = (np.abs(theta) < 1.5) & (np.abs(phi) < 1.5)
        
        x_src = np.tan(theta)
        y_src = np.tan(phi) / np.cos(theta)
        
        map_x = (x_src + 1) * w / 2
        map_y = (y_src + 1) * h / 2
        
        return map_x.astype(np.float32), map_y.astype(np.float32)

    def generate_sbs(self, left_img, depth_map, inpaint=False, mode='standard'):
        h, w = left_img.shape[:2]
        depth_norm = self.normalize_depth(depth_map)
        
        # --- PRÉ-TRAITEMENT SELON LE MODE ---
        if mode == 'vr360':
            # En 360, on redimensionne souvent en ratio 2:1 (ex: 4096x2048)
            # Ici on garde la taille source mais on applique la correction polaire
            lat_correction = self._get_latitude_correction(h, w)
            # On réduit la disparité aux pôles
            depth_norm = depth_norm * lat_correction

        elif mode == 'vr180':
            map_x_sphere, map_y_sphere = self._build_vr180_map(h, w)
            left_img = cv2.remap(left_img, map_x_sphere, map_y_sphere, cv2.INTER_LINEAR)
            depth_norm = cv2.remap(depth_norm, map_x_sphere, map_y_sphere, cv2.INTER_LINEAR)

        # --- CALCUL DU SHIFT ---
        shift = (depth_norm - self.convergence) * (self.divergence * w)
        
        x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x_grid + shift).astype(np.float32)
        map_y = y_grid.astype(np.float32)
        
        # --- REMAP & INPAINT ---
        border_mode = cv2.BORDER_CONSTANT if inpaint else cv2.BORDER_REPLICATE
        # En VR360, on préfère 'WRAP' pour que la gauche de l'image boucle avec la droite
        if mode == 'vr360':
            border_mode = cv2.BORDER_WRAP

        right_img = cv2.remap(left_img, map_x, map_y, cv2.INTER_LINEAR, 
                              borderMode=border_mode, borderValue=(0,0,0))
        
        if inpaint:
            gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            mask = (gray == 0).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=1)
            right_img = cv2.inpaint(right_img, mask, 3, cv2.INPAINT_TELEA)

        if mode == 'anaglyph':
            # OpenCV utilise le format BGR (Bleu=0, Vert=1, Rouge=2)
            # On copie l'image droite (qui a les bonnes perspectives pour le bleu/vert)
            anaglyph_img = right_img.copy()
            # On remplace le canal Rouge par celui de l'image gauche
            anaglyph_img[:, :, 2] = left_img[:, :, 2]
            return anaglyph_img

        return np.hstack((left_img, right_img))