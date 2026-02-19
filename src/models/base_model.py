from abc import ABC, abstractmethod
import numpy as np

class DepthModel(ABC):
    def __init__(self, device='cuda'):
        self.device = device
    @abstractmethod
    def load(self): pass
    @abstractmethod
    def infer(self, frame: np.ndarray) -> np.ndarray: pass
