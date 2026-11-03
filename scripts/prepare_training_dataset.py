"""
Dataset Preparation Script for Real-ESRGAN Fine-Tuning
Downloads high-resolution Ground-Truth images from standard SR datasets
(BSD100 by UC Berkeley & Urban100 by Huang et al., CVPR 2015)
and generates meta_info.txt required by Real-ESRGAN DataLoader.
"""

import os
import glob
import urllib.request
import cv2

def main():
    train_dir = os.path.join('datasets', 'train_HR')
    os.makedirs(train_dir, exist_ok=True)

    print("Step 1: Downloading High-Resolution Ground-Truth Images...")
    print("- Dataset 1: BSD100 (UC Berkeley) - Natural scenes, textures, and landscapes")
    print("- Dataset 2: Urban100 (CVPR 2015) - Architectural structures, edges, and high-frequency details")

    headers = {'User-Agent': 'Mozilla/5.0'}
    count = len(glob.glob(os.path.join(train_dir, '*.png')))

    # Download 25 images from BSD100
    for idx in range(1, 26):
        fname = f"bsd_{idx:03d}.png"
        target_path = os.path.join(train_dir, fname)
        if not os.path.exists(target_path):
            url = f"https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/BSD100/image_SRF_4/img_{idx:03d}_SRF_4_HR.png"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp, open(target_path, 'wb') as f:
                    f.write(resp.read())
                count += 1
                print(f"Downloaded [{count}] {fname}")
            except Exception as e:
                print(f"Error {fname}: {e}")

    # Download 25 images from Urban100
    for idx in range(1, 26):
        fname = f"urban_{idx:03d}.png"
        target_path = os.path.join(train_dir, fname)
        if not os.path.exists(target_path):
            url = f"https://raw.githubusercontent.com/jbhuang0604/SelfExSR/master/data/Urban100/image_SRF_4/img_{idx:03d}_SRF_4_HR.png"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp, open(target_path, 'wb') as f:
                    f.write(resp.read())
                count += 1
                print(f"Downloaded [{count}] {fname}")
            except Exception as e:
                print(f"Error {fname}: {e}")

    all_images = sorted(glob.glob(os.path.join(train_dir, '*.png')) + glob.glob(os.path.join(train_dir, '*.jpg')))
    print(f"\nTotal Ground-Truth HR training images collected: {len(all_images)}")

    print("\nStep 2: Generating meta_info.txt for DataLoader...")
    meta_info_path = os.path.join(train_dir, 'meta_info.txt')
    with open(meta_info_path, 'w', encoding='utf-8') as f:
        for img_p in all_images:
            rel_name = os.path.basename(img_p)
            img = cv2.imread(img_p)
            if img is not None:
                h, w, c = img.shape
                f.write(f"{rel_name} ({h},{w},{c})\n")

    print(f"meta_info.txt generated at: {meta_info_path}")
    print("Training dataset is completely ready!")

if __name__ == '__main__':
    main()
