import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from .base_model import DepthModel

class DepthAnythingV2(DepthModel):
    def __init__(self, version='small', device='cuda'):
        super().__init__(device)
        self.model_id = "depth-anything/Depth-Anything-V2-Small-hf" # Modifiable vers base/large
        if version == 'base': self.model_id = "depth-anything/Depth-Anything-V2-Base-hf"
        elif version == 'large':
            self.model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        self.processor = None; self.model = None

    def load(self):
        print(f"Chargement de {self.model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)

    def infer(self, frame):
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            pred = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            )
        return pred.squeeze().cpu().numpy()
