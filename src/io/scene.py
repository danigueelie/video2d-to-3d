from scenedetect import detect, ContentDetector
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SceneManager:
    def __init__(self, threshold=27.0):
        self.threshold = threshold
        self.prev_frame_gray = None

    def is_cut(self, frame, frame_idx):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_small = cv2.resize(curr_gray, (64, 64))
        is_scene_change = False
        if self.prev_frame_gray is not None:
            score = np.mean(cv2.absdiff(curr_small, self.prev_frame_gray))
            if score > self.threshold:
                is_scene_change = True
                logger.info(f"Scene change detected at frame {frame_idx} (score: {score:.2f})")
        self.prev_frame_gray = curr_small
        return is_scene_change
