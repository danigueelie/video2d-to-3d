import numpy as np
from collections import deque

class TemporalFilter:
    def __init__(self, method='ema', factor=0.3):
        self.method = method; self.factor = factor
        self.history = deque(maxlen=5); self.ema_prev = None

    def reset(self):
        self.history.clear(); self.ema_prev = None

    def process(self, depth_map):
        if self.method == 'ema':
            if self.ema_prev is None:
                self.ema_prev = depth_map
                return depth_map
            smoothed = depth_map * self.factor + self.ema_prev * (1 - self.factor)
            self.ema_prev = smoothed
            return smoothed
        return depth_map
