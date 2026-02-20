import argparse
import logging
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.io.video_reader import VideoHandler
from src.io.scene import SceneManager
from src.models.depth_anything import DepthAnythingV2
from src.models.depthcrafter import DepthCrafter
from src.processing.temporal import TemporalFilter
from src.processing.optical_flow import OpticalFlowStabilizer
from src.stereo.generator import StereoGen
from src.evaluation.metrics import MetricTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Video3D")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Vidéo source")
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    # Mode de sortie mis à jour
    parser.add_argument('--mode', type=str, default='sbs', choices=['sbs', 'vr180', 'vr360', 'heatmap', 'heatmap_numbered', 'anaglyph'], 
                        help="sbs (Flat), vr180, vr360, heatmap (Debug), heatmap_numbered (Avec N° Frame), anaglyph (Lunettes Rouge/Cyan)")
    
    parser.add_argument('--model_type', type=str, default='depthanything', choices=['depthanything', 'depthcrafter'])
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'base', 'large'])
    
    parser.add_argument('--divergence', type=float, default=2.0)
    parser.add_argument('--inpaint', action='store_true')
    
    # Nouvelle option pour la sauvegarde des PNG
    parser.add_argument('--save_frames', action='store_true', help="Sauvegarder les frames, depths et outputs individuels en PNG")
    
    # Stabilisation
    parser.add_argument('--stabilize', action='store_true')
    parser.add_argument('--stabilizer_type', type=str, default='ema', choices=['ema', 'optical_flow'])
    parser.add_argument('--smooth_factor', type=float, default=0.2)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    suffix = f"_{args.model_type}_{args.mode}"
    if args.stabilize: suffix += f"_{args.stabilizer_type}"
    
    filename = f"{os.path.basename(args.input).split('.')[0]}{suffix}"
    
    video = VideoHandler(args.input, args.output_dir, filename)
    meta = video.get_info()
    logger.info(f"🎞️ Source: {meta['width']}x{meta['height']} @ {meta['fps']}fps | Mode: {args.mode.upper()}")

    # Création des dossiers pour les PNG si l'option est activée
    if args.save_frames:
        frames_dir = os.path.join(args.output_dir, filename, "frames")
        depths_dir = os.path.join(args.output_dir, filename, "depths")
        outs_dir = os.path.join(args.output_dir, filename, "outputs")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(depths_dir, exist_ok=True)
        os.makedirs(outs_dir, exist_ok=True)
        logger.info(f"📁 Sauvegarde des PNG activée dans : {os.path.join(args.output_dir, filename)}")

    # Modèle
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_type == 'depthcrafter':
        model = DepthCrafter(device=device)
    else:
        model = DepthAnythingV2(version=args.model_size, device=device)
    model.load()

    # Outils Processing
    scene_manager = SceneManager(threshold=25.0)
    
    stabilizer = None
    if args.stabilize:
        if args.stabilizer_type == 'optical_flow':
            stabilizer = OpticalFlowStabilizer(factor=args.smooth_factor)
        else:
            stabilizer = TemporalFilter(method='ema', factor=args.smooth_factor)
            
    stereo = StereoGen(divergence=args.divergence)
    metrics = MetricTracker()

    # --- NOUVEAU : Gestion de la largeur vidéo ---
    # L'anaglyphe garde la largeur originale, les autres doublent la largeur
    out_width = meta['width'] if args.mode == 'anaglyph' else meta['width'] * 2
    video.start_writer(out_width, meta['height'])

    # Boucle
    pbar = tqdm(total=meta['frames'], unit="frames")
    try:
        while True:
            ret, frame = video.read()
            if not ret: break

            # 1. Inférence
            depth = model.infer(frame)
            
            # 2. Stabilisation
            if args.stabilize:
                if scene_manager.is_cut(frame, video.frame_count):
                    stabilizer.reset()
                
                if args.stabilizer_type == 'optical_flow':
                    depth = stabilizer.process(frame, depth)
                else:
                    depth = stabilizer.process(depth)

            # 3. Métriques
            metrics.update(depth)

            # 4. Génération Sortie
            if args.mode in ['heatmap', 'heatmap_numbered']:
                d_norm = stereo.normalize_depth(depth)
                d_color = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                output_frame = np.hstack((frame, d_color))
                
                if args.mode == 'heatmap_numbered':
                    # Ajout d'un contour noir pour que le texte blanc soit visible sur fond clair
                    text = f"Frame: {video.frame_count}"
                    cv2.putText(output_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(output_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                
            elif args.mode == 'anaglyph':
                    # Mode lunettes 3D
                    output_frame = stereo.generate_sbs(frame, depth, inpaint=args.inpaint, mode='anaglyph')
                
            elif args.mode == 'vr180':
                output_frame = stereo.generate_sbs(frame, depth, inpaint=args.inpaint, mode='vr180')
                
            elif args.mode == 'vr360':
                output_frame = stereo.generate_sbs(frame, depth, inpaint=args.inpaint, mode='vr360')
                
            else:
                output_frame = stereo.generate_sbs(frame, depth, inpaint=args.inpaint, mode='standard')
            
            # Sauvegarde des PNG individuels
            if args.save_frames:
                # Frame originale
                cv2.imwrite(os.path.join(frames_dir, f"frame_{video.frame_count:05d}.png"), frame)
                # Depth map (normalisée en 0-255 niveaux de gris)
                d_norm = stereo.normalize_depth(depth)
                cv2.imwrite(os.path.join(depths_dir, f"depth_{video.frame_count:05d}.png"), (d_norm * 255).astype(np.uint8))
                # Output frame (Heatmap, SBS, Anaglyphe, etc.)
                cv2.imwrite(os.path.join(outs_dir, f"out_{video.frame_count:05d}.png"), output_frame)

            video.write(output_frame)
            pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Arrêt...")
    finally:
        pbar.close(); video.close(); video.mux_audio()
        
        summary = metrics.get_summary()
        logger.info("-" * 40)
        logger.info(f"📊 Stabilité (RMSE) : {summary['temporal_instability']:.5f}")
        logger.info(f"📊 Qualité de la Stabilisation : {summary['avg_ssim']:.4f}")
        logger.info(f"📈 Scintillement (PSD) : {summary['high_freq_psd']:.5f}")
        logger.info("-" * 40)
        
        plot_path = metrics.plot_metrics(args.output_dir, filename)
        if plot_path:
            logger.info(f"📉 Graphique sauvegardé : {plot_path}")

if __name__ == '__main__':
    main()