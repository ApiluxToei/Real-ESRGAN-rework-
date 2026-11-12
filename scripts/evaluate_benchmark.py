"""
Benchmark Evaluation Script for Real-ESRGAN
Calculates PSNR, SSIM, LPIPS, NIQE, and MUSIQ metrics on the 20 benchmark test images.
Compares performance directly against the Official Real-ESRGAN baseline.
"""

import os
import csv
import glob
import argparse
import torch
import cv2
import numpy as np
import pyiqa
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

OFFICIAL_BASELINE = {
    'PSNR': 21.75,
    'SSIM': 0.6477,
    'LPIPS': 0.2349,
    'NIQE': 4.746,
    'MUSIQ': 64.58
}

def run_inference_if_needed(model_path, lr_dir, out_dir, tile=256, device_id=0):
    os.makedirs(out_dir, exist_ok=True)
    lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.*')))
    
    # Check if all output files already exist
    existing = glob.glob(os.path.join(out_dir, '*.*'))
    if len(existing) == len(lr_files):
        print(f"Inference outputs already present in {out_dir} ({len(existing)} images).")
        return out_dir
        
    print(f"Running inference for model: {model_path} -> {out_dir}")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=device_id
    )
    
    for idx, path in enumerate(lr_files):
        fname = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        output, _ = upsampler.enhance(img, outscale=4)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, output)
        
    print(f"Inference complete: {len(lr_files)} images written to {out_dir}")
    return out_dir

def evaluate_directory(gt_dir, sr_dir, model_name, device='cuda'):
    print(f"\n[Evaluating {model_name}] in {sr_dir}...")
    
    # Load metrics
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)

    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.*')))
    sr_files = sorted(glob.glob(os.path.join(sr_dir, '*.*')))
    
    gt_dict = {os.path.splitext(os.path.basename(f))[0].replace('_out', ''): f for f in gt_files}
    
    psnr_list, ssim_list, lpips_list, niqe_list, musiq_list = [], [], [], [], []
    per_image_results = []
    
    for sr_path in sr_files:
        base = os.path.splitext(os.path.basename(sr_path))[0].replace('_out', '')
        
        # No-reference
        niqe_val = float(niqe_metric(sr_path).item())
        musiq_val = float(musiq_metric(sr_path).item())
        niqe_list.append(niqe_val)
        musiq_list.append(musiq_val)
        
        # Full-reference
        psnr_val, ssim_val, lpips_val = 0.0, 0.0, 0.0
        if base in gt_dict:
            gt_path = gt_dict[base]
            sr_img = cv2.imread(sr_path)
            gt_img = cv2.imread(gt_path)
            
            min_h = min(sr_img.shape[0], gt_img.shape[0])
            min_w = min(sr_img.shape[1], gt_img.shape[1])
            sr_crop = sr_img[:min_h, :min_w]
            gt_crop = gt_img[:min_h, :min_w]
            
            sr_t = torch.from_numpy(sr_crop[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            gt_t = torch.from_numpy(gt_crop[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            sr_t, gt_t = sr_t.to(device), gt_t.to(device)
            
            psnr_val = float(psnr_metric(sr_t, gt_t).item())
            ssim_val = float(ssim_metric(sr_t, gt_t).item())
            lpips_val = float(lpips_metric(sr_t, gt_t).item())
            
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            
        per_image_results.append({
            'name': base,
            'PSNR': psnr_val,
            'SSIM': ssim_val,
            'LPIPS': lpips_val,
            'NIQE': niqe_val,
            'MUSIQ': musiq_val
        })
        
    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0
    avg_niqe = float(np.mean(niqe_list)) if niqe_list else 0.0
    avg_musiq = float(np.mean(musiq_list)) if musiq_list else 0.0

    summary = {
        'Model': model_name,
        'PSNR': avg_psnr,
        'SSIM': avg_ssim,
        'LPIPS': avg_lpips,
        'NIQE': avg_niqe,
        'MUSIQ': avg_musiq
    }
    
    return summary, per_image_results

def print_comparison_table(results_list):
    print("\n" + "="*95)
    print("Super-Resolution Benchmark Results on 20 Standard Test Images")
    print("="*95)
    header = f"{'Model':<35} | {'PSNR ^':<8} | {'SSIM ^':<8} | {'LPIPS v':<8} | {'NIQE v':<8} | {'MUSIQ ^':<8} | Status"
    print(header)
    print("-" * len(header))
    
    for res in results_list:
        psnr_pass = res['PSNR'] > OFFICIAL_BASELINE['PSNR']
        ssim_pass = res['SSIM'] > OFFICIAL_BASELINE['SSIM']
        lpips_pass = res['LPIPS'] < OFFICIAL_BASELINE['LPIPS']
        niqe_pass = res['NIQE'] < OFFICIAL_BASELINE['NIQE']
        musiq_pass = res['MUSIQ'] > OFFICIAL_BASELINE['MUSIQ']
        
        passes = sum([psnr_pass, ssim_pass, lpips_pass, niqe_pass, musiq_pass])
        status = f"{passes}/5 Win"
        if passes == 5:
            status += " (CLEAN SWEEP 🏆)"
        elif res['Model'] == 'Real-ESRGAN (Official)':
            status = "Baseline"
            
        print(f"{res['Model']:<35} | {res['PSNR']:<8.2f} | {res['SSIM']:<8.4f} | {res['LPIPS']:<8.4f} | {res['NIQE']:<8.3f} | {res['MUSIQ']:<8.2f} | {status}")
    print("="*95)

def save_to_csv(summary_list, csv_path='reports/experiment_results.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ['Model', 'PSNR', 'SSIM', 'LPIPS', 'NIQE', 'MUSIQ', 'Beats_Official_All']
    file_exists = os.path.isfile(csv_path)
    
    rows = []
    for s in summary_list:
        beats_all = (
            s['PSNR'] > OFFICIAL_BASELINE['PSNR'] and
            s['SSIM'] > OFFICIAL_BASELINE['SSIM'] and
            s['LPIPS'] < OFFICIAL_BASELINE['LPIPS'] and
            s['NIQE'] < OFFICIAL_BASELINE['NIQE'] and
            s['MUSIQ'] > OFFICIAL_BASELINE['MUSIQ']
        )
        rows.append({
            'Model': s['Model'],
            'PSNR': f"{s['PSNR']:.2f}",
            'SSIM': f"{s['SSIM']:.4f}",
            'LPIPS': f"{s['LPIPS']:.4f}",
            'NIQE': f"{s['NIQE']:.3f}",
            'MUSIQ': f"{s['MUSIQ']:.2f}",
            'Beats_Official_All': "YES 🏆" if beats_all else "NO"
        })
        
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/benchmark_20/gt')
    parser.add_argument('--lr', type=str, default='datasets/benchmark_20/lr')
    parser.add_argument('--model_paths', nargs='+', default=None, help='List of model checkpoints to evaluate')
    parser.add_argument('--sr_dirs', nargs='+', default=None, help='List of directories with SR output images')
    parser.add_argument('--names', nargs='+', required=True, help='Display names for models')
    parser.add_argument('--out_root', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_csv', type=str, default='reports/experiment_results.csv')
    args = parser.parse_args()

    results_list = []
    
    if args.model_paths:
        for mpath, name in zip(args.model_paths, args.names):
            clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            out_dir = os.path.join(args.out_root, f"benchmark_{clean_name}")
            run_inference_if_needed(mpath, args.lr, out_dir, device_id=0 if args.device == 'cuda' else None)
            res, _ = evaluate_directory(args.gt, out_dir, name, device=args.device)
            results_list.append(res)
    elif args.sr_dirs:
        for sdir, name in zip(args.sr_dirs, args.names):
            res, _ = evaluate_directory(args.gt, sdir, name, device=args.device)
            results_list.append(res)

    print_comparison_table(results_list)
    if args.save_csv:
        save_to_csv(results_list, args.save_csv)

if __name__ == '__main__':
    main()

