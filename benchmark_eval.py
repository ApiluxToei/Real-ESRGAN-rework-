"""
Benchmark Evaluation Script for Real-ESRGAN
Calculates PSNR, SSIM, LPIPS, NIQE, and MUSIQ metrics matching Section 7.2 Benchmark Table.
"""

import os
import argparse
import glob
import torch
import pyiqa
from PIL import Image
import torchvision.transforms.functional as TF

def evaluate_models(gt_dir, sr_dirs, model_names, device_str='cuda'):
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device for metrics: {device}")

    # Initialize pyiqa metrics
    print("Loading IQA metrics...")
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    print("Metrics ready.")

    # Gather GT images
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG')
    gt_paths = []
    if gt_dir and os.path.exists(gt_dir):
        for ext in extensions:
            gt_paths.extend(glob.glob(os.path.join(gt_dir, ext)))
    gt_paths = sorted(gt_paths)

    results = {}

    for model_name, sr_dir in zip(model_names, sr_dirs):
        print(f"\nEvaluating [{model_name}] in {sr_dir}...")
        sr_files = []
        for ext in extensions:
            sr_files.extend(glob.glob(os.path.join(sr_dir, ext)))
        sr_files = sorted(sr_files)

        if not sr_files:
            print(f"Warning: No images found in {sr_dir}")
            continue

        psnr_list, ssim_list, lpips_list, niqe_list, musiq_list = [], [], [], [], []

        for sr_path in sr_files:
            basename = os.path.splitext(os.path.basename(sr_path))[0]
            # remove common suffix like _out
            base_clean = basename.replace('_out', '')

            # Calculate NIQE & MUSIQ (No-reference)
            niqe_val = niqe_metric(sr_path).item()
            musiq_val = musiq_metric(sr_path).item()
            niqe_list.append(niqe_val)
            musiq_list.append(musiq_val)

            # Full-reference if GT exists
            matched_gt = None
            if gt_paths:
                for gp in gt_paths:
                    g_base = os.path.splitext(os.path.basename(gp))[0]
                    if g_base == base_clean or g_base in base_clean or base_clean in g_base:
                        matched_gt = gp
                        break

            if matched_gt:
                import cv2
                import numpy as np
                sr_img = cv2.imread(sr_path)
                gt_img = cv2.imread(matched_gt)
                h_sr, w_sr = sr_img.shape[:2]
                h_gt, w_gt = gt_img.shape[:2]
                min_h, min_w = min(h_sr, h_gt), min(w_sr, w_gt)
                sr_crop = sr_img[:min_h, :min_w]
                gt_crop = gt_img[:min_h, :min_w]
                
                # convert BGR to RGB tensor [1, 3, H, W] in [0, 1]
                sr_t = torch.from_numpy(sr_crop[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                gt_t = torch.from_numpy(gt_crop[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                sr_t, gt_t = sr_t.to(device), gt_t.to(device)

                psnr_val = psnr_metric(sr_t, gt_t).item()
                ssim_val = ssim_metric(sr_t, gt_t).item()
                lpips_val = lpips_metric(sr_t, gt_t).item()
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                lpips_list.append(lpips_val)

        avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
        avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0.0
        avg_lpips = sum(lpips_list) / len(lpips_list) if lpips_list else 0.0
        avg_niqe = sum(niqe_list) / len(niqe_list) if niqe_list else 0.0
        avg_musiq = sum(musiq_list) / len(musiq_list) if musiq_list else 0.0

        results[model_name] = {
            'PSNR': avg_psnr,
            'SSIM': avg_ssim,
            'LPIPS': avg_lpips,
            'NIQE': avg_niqe,
            'MUSIQ': avg_musiq,
            'Count': len(sr_files)
        }

    # Print summary table
    print("\n" + "="*80)
    print("Benchmark Results on 20 Standard Test Images")
    print("="*80)
    header = f"{'Model':<35} | {'PSNR ^':<8} | {'SSIM ^':<8} | {'LPIPS v':<8} | {'NIQE v':<8} | {'MUSIQ ^':<8}"
    print(header)
    print("-" * len(header))
    for model_name, res in results.items():
        print(f"{model_name:<35} | {res['PSNR']:<8.2f} | {res['SSIM']:<8.4f} | {res['LPIPS']:<8.4f} | {res['NIQE']:<8.3f} | {res['MUSIQ']:<8.2f}")
    print("="*80)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Super-Resolution benchmark metrics")
    parser.add_argument('--gt', type=str, default=None, help="Ground-truth directory (optional)")
    parser.add_argument('--sr_dirs', nargs='+', required=True, help="Directories of SR outputs")
    parser.add_argument('--names', nargs='+', required=True, help="Names of models corresponding to sr_dirs")
    parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
    args = parser.parse_args()

    evaluate_models(args.gt, args.sr_dirs, args.names, args.device)
