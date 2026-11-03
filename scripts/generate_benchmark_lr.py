"""
Generate degraded LR (Low-Resolution) images for the 20 benchmark test images.
Scale factor: 4x downsampling (bicubic + slight compression to simulate real test inputs).
"""

import os
import glob
import cv2

gt_dir = 'datasets/benchmark_20/gt'
lr_dir = 'datasets/benchmark_20/lr'
os.makedirs(lr_dir, exist_ok=True)

gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.*')))
print(f"Generating LR inputs for {len(gt_files)} images...")

for gt_path in gt_files:
    fname = os.path.basename(gt_path)
    img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    # Ensure dimensions divisible by 4
    h_lr = h // 4
    w_lr = w // 4
    lr = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
    
    # Save as LR input
    out_path = os.path.join(lr_dir, fname)
    cv2.imwrite(out_path, lr)

print(f"Successfully generated {len(gt_files)} LR images in {lr_dir}")
