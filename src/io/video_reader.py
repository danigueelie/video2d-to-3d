import cmd

import cv2
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class VideoHandler:
    def __init__(self, input_path, output_dir, filename_prefix):
        self.input_path = input_path
        self.output_dir = output_dir

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo.")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.temp_video_path = os.path.join(output_dir, f"{filename_prefix}_temp.mp4")
        self.final_output_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
        self.writer = None
        self.frame_count = 0

    def get_info(self):
        return {'width': self.width, 'height': self.height, 'fps': self.fps, 'frames': self.total_frames}

    def start_writer(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.temp_video_path, fourcc, self.fps, (width, height))
        logger.info(f"Writer initialisé : {self.temp_video_path} ({width}x{height})")

    def read(self):
        ret, frame = self.cap.read()
        if ret: self.frame_count += 1
        return ret, frame

    def write(self, frame):
        if self.writer: self.writer.write(frame)

    def close(self):
        if self.cap: self.cap.release()
        if self.writer: self.writer.release()

    def mux_audio(self):
        """
        Remet l'audio original et ré-encode la vidéo pour compatibilité VR (H.264/yuv420p).
        """
        if not os.path.exists(self.temp_video_path):
            logger.warning("Pas de vidéo temporaire à mixer.")
            return
        
        # Sur Colab, ffmpeg est pré-installé dans le système
        # Utilisation de libx264 et yuv420p pour garantir la lecture sur Meta Quest / Casques VR
        # Le flag '-c:v copy' a été remplacé par un ré-encodage propre.
        cmd = [
            'ffmpeg', '-y', '-v', 'error',
            '-i', self.temp_video_path, # Vidéo générée (muette, codec OpenCV)
            '-i', self.input_path,      # Vidéo source (audio)
            '-c:v', 'libx264',          # Ré-encodage H.264 (Universel)
            '-pix_fmt', 'yuv420p',      # Format pixel indispensable pour les lecteurs vidéos standard
            '-preset', 'fast',          # Rapidité d'encodage
            '-crf', '23',               # Qualité visuelle (18-28 est bon)
            '-c:a', 'aac',              # Encodage audio
            '-map', '0:v:0?',            # Prendre flux vidéo du fichier 0
            '-map', '1:a:0?',            # Prendre flux audio du fichier 1
            '-shortest',                # Couper au plus court
            self.final_output_path
        ]
        
        try:
            logger.info("Muxing audio & Ré-encodage H.264 avec FFmpeg...")
            subprocess.run(cmd, check=True)
            print(f"Vidéo finale avec audio : {self.final_output_path}")
            if os.path.exists(self.temp_video_path): os.remove(self.temp_video_path)
        except Exception as e:
            logger.error(f"Erreur FFmpeg (L'audio n'a pas pu être ajouté) : {e}")
            print(f"Erreur Audio Muxing: {e}")
            if os.path.exists(self.temp_video_path): os.rename(self.temp_video_path, self.final_output_path)