# Comprehensive Model, Codebase & Dataset Audit Report
**Project:** Real-ESRGAN (Our Team Model) Enhancement  
**Role:** Senior Deep Learning Research Engineer (Super-Resolution & Perceptual Restoration)  
**Date:** November 12, 2026  
**Status:** Audit Completed | Ready for User Approval  

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Challenge
Our current model checkpoint (`net_g_latest.pth`) achieved competitive scores on no-reference perceptual quality metrics (**NIQE: 3.864**, **MUSIQ: 65.49**), outperforming the Official Real-ESRGAN baseline (**NIQE: 4.746**, **MUSIQ: 64.58**). However, on full-reference reconstruction fidelity metrics, our model suffers a severe performance degradation:

| Metric | Real-ESRNet (Reference) | Real-ESRGAN (Official) | Real-ESRGAN (Our Team Model) | Gap vs. Official | Optimization Priority |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PSNR (dB) ↑** | 23.41 | **21.75** | 20.69 | **-1.06 dB** 🔴 | **Priority 1 (Primary)** |
| **SSIM ↑** | 0.7065 | **0.6477** | 0.5261 | **-0.1216 (-18.8%)** 🔴 | **Priority 2 (Primary)** |
| **LPIPS ↓** | 0.3361 | **0.2349** | 0.3124 | **+0.0775 (+33.0%)** 🔴 | **Priority 3 (Primary)** |
| **NIQE ↓** | 7.168 | 4.746 | **3.864** | **-0.882 (-18.6%)** 🟢 | **Secondary (Preserve)** |
| **MUSIQ ↑** | 56.86 | 64.58 | **65.49** | **+0.91 (+1.4%)** 🟢 | **Secondary (Preserve)** |

*Note: In the benchmark evaluation, PSNR, SSIM, and LPIPS directly reflect fidelity to the ground truth, whereas NIQE and MUSIQ assess visual sharpness and absence of typical low-level artifacts.*

---

## 2. Phase 1 — Detailed Codebase Audit

### 2.1 Generator Architecture
* **Model Class:** `RRDBNet` (`realesrgan/archs/srvgg_arch.py`, `basicsr/archs/rrdbnet_arch.py`)
* **Parameters:** `num_in_ch=3`, `num_out_ch=3`, `num_feat=64`, `num_block=23`, `num_grow_ch=32`, `scale=4`
* **Total Trainable Parameters:** 16,697,987 (~16.7M)
* **Design Consistency:** Completely identical to the official Real-ESRGAN x4 architecture (23 Residual-in-Residual Dense Blocks). No architectural defect detected in the generator itself.
* **Initialization:** Pretrained weights loaded from `experiments/pretrained_models/RealESRGAN_x4plus.pth` using `param_key_g: params_ema`.

### 2.2 Discriminator Architecture
* **Model Class:** `UNetDiscriminatorSN` (`realesrgan/models/realesrgan_model.py`)
* **Parameters:** `num_in_ch=3`, `num_feat=64`, `skip_connection=True`
* **Total Parameters:** 4,376,897 (~4.38M)
* **Spectral Normalization (SN):** Enabled on all convolutional layers to stabilize adversarial training and bound Lipschitz constants.
* **Initialization:** Loaded from `experiments/pretrained_models/RealESRGAN_x4plus_netD.pth`.

### 2.3 Loss Functions in Current Configuration
Inspected from `options/train_proposed_realesrgan_x4plus.yml`:
1. **Pixel Loss (`pixel_opt`):**
   * Type: `L1Loss`
   * Weight: **`0.8`** *(Official uses `1.0`)*
   * Target: Evaluated against `l1_gt = self.gt_usm` because `l1_gt_usm: True`.
2. **Perceptual Loss (`perceptual_opt`):**
   * Backbone: VGG-19 feature extractor
   * Layers & Weights: `conv1_2: 0.1`, `conv2_2: 0.1`, `conv3_4: 1.0`, `conv4_4: 1.0`, `conv5_4: 1.0`
   * Criterion: L1 on feature maps, weight = `1.0`.
   * Target: Evaluated against `percep_gt = self.gt_usm` because `percep_gt_usm: True`.
3. **Adversarial Loss (`gan_opt`):**
   * Type: Vanilla GAN loss with BCE
   * Weight: **`0.15`** *(Official uses `0.10`)*
   * Target: Discriminator evaluates real vs. fake.
4. **Structural Loss:**
   * **None.** No SSIM, MS-SSIM, or gradient/edge-aware loss was employed.

### 2.4 Optimizer, Scheduler & Training Dynamics
* **Optimizer G & D:** Adam (`lr: 5e-5`, `betas: [0.9, 0.99]`, `weight_decay: 0`)
* **Scheduler:** MultiStepLR (`milestones: [20000, 40000]`, `gamma: 0.5`)
* **EMA:** `ema_decay: 0.999` (properly updated during training)
* **DataLoader Concurrency:** `num_worker_per_gpu: 0` (configured to prevent Windows multiprocessing memory leaks)
* **Batch Size & Patch Size:** `batch_size: 4`, `gt_size: 192` (adjusted to fit within 6GB VRAM)

### 2.5 Validation & Checkpointing Pipeline
* **Validation:** Periodic validation in YAML was commented out (`# val:`). The training ran for 46,450 iterations blindly without checkpoint metric tracking on the benchmark set.
* **Checkpoint Frequency:** Every 1,000 iterations (`net_g_1000.pth` to `net_g_46000.pth`, plus `net_g_latest.pth`).
* **Pretrained Pool:** Official RealESRGAN_x4plus weights are safely preserved under `experiments/pretrained_models/` and `weights/`.

### 2.6 Hardware & Environment Capabilities
* **GPU:** NVIDIA GeForce RTX 2060 (6144 MiB VRAM)
* **Driver / CUDA:** Driver 610.88, CUDA 12.4
* **System RAM:** 15.8 GB (Available: ~4.5 GB to 8.0 GB depending on active background apps)
* **CPU:** 12 Logical Cores
* **VRAM Constraint:** 6GB limits single-GPU batch processing. `batch_size: 4` with `gt_size: 192` consumes ~4.0–4.2 GB VRAM, leaving safety headroom for forward/backward passes.

---

## 3. Phase 2 — Dataset & Benchmark Audit

### 3.1 Training Dataset Audit (`datasets/train_HR`)
* **Dataset Size:** Exactly **55 images**.
  * 25 images from BSD100 (`bsd_001.png` - `bsd_025.png`)
  * 25 images from Urban100 (`urban_001.png` - `urban_025.png`)
  * 5 sample images copied from `inputs/` (`00003.png`, `0014.jpg`, `0030.jpg`, `ADE_val_00000114.jpg`, `OST_009.png`)
* **Resolution Range:** From 179×179 up to 1024×680.
* **Critical Finding: Insufficient Volume & Distribution Diversity:**
  * Official Real-ESRGAN was trained on **DF2K** (3,450 images) + **OST** (10,324 images) = 13,774 images.
  * Our dataset contains only 55 images. Running 46,450 iterations at batch size 4 equates to `(46,450 * 4) / 55 = 3,378` epochs!
  * **Consequence:** Massive memorization/overfitting of specific high-frequency patterns, degrading generalizeable edge and structure reconstruction on unseen images.

### 3.2 Severe Finding: Train/Benchmark Data Leakage
* **Benchmark Test Set (`datasets/benchmark_20/gt`):** 20 images.
* **Exact Duplicate Search:**
  ```python
  EXACT MATCH: Benchmark 00003.png == Train 00003.png
  EXACT MATCH: Benchmark 0014.jpg == Train 0014.jpg
  EXACT MATCH: Benchmark 0030.jpg == Train 0030.jpg
  EXACT MATCH: Benchmark ADE_val_00000114.jpg == Train ADE_val_00000114.jpg
  EXACT MATCH: Benchmark OST_009.png == Train OST_009.png
  ```
* **Audit Result:** **5 out of the 20 benchmark images (25% of the test suite) were present in the training set!**
* **Resolution Action:** For all subsequent training and validation, benchmark images must be strictly isolated from the training data pool to prevent contamination and guarantee scientific validity.

### 3.3 Degradation Pipeline Mismatch (Training vs. Benchmark Evaluation)
* **Benchmark LR Generation (`scripts/generate_benchmark_lr.py`):**
  * Uses **pure bicubic downsampling**:
    ```python
    lr = cv2.resize(img, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
    ```
  * **No blur, no noise, no JPEG compression artifacts, and no sinc filtering.**
* **Training Degradation (`options/train_proposed_realesrgan_x4plus.yml`):**
  * Severe second-order degradation:
    * Gaussian noise up to $\sigma = 25$, Poisson noise scale up to 2.5
    * Extreme JPEG compression: quality range **`[15, 85]`** *(Official is [30, 95])*
    * Sinc filter with probability **`0.8`**
* **Consequence of Mismatch:**
  * The model was trained with the assumption that every input image suffers from heavy noise and severe block compression.
  * When presented with clean bicubic downsampled benchmark inputs, the model acts as an aggressive denoiser/de-blocker: it erases subtle natural gradations as "noise" and hallucinates synthetic edges to replace non-existent JPEG compression artifacts.
  * This directly leads to the collapse of pixel-level fidelity metrics (**PSNR drops to 20.69 dB, SSIM drops to 0.5261**).

### 3.4 Pixel Alignment and Metric Implementation Audit (`benchmark_eval.py`)
* **Color Space & Normalization:** Correctly converts OpenCV BGR to RGB, floats in `[0.0, 1.0]`.
* **Metric Engine:** PyIQA library (`psnr`, `ssim`, `lpips`, `niqe`, `musiq`).
* **Spatial Cropping Issue:**
  * In `0014.jpg`, GT resolution is 179×179. In `generate_benchmark_lr.py`:
    `h_lr = 179 // 4 = 44`, `w_lr = 179 // 4 = 44`.
    When upscaled by 4x, the SR image is 176×176.
  * `benchmark_eval.py` slices `[:min_h, :min_w]`, dropping 3 pixels from the bottom and right edges of the GT.
  * Standard SR evaluation requires a `border_crop=4` to eliminate boundary convolution artifacts.

---

## 4. Root Cause Analysis: Why PSNR and SSIM Collapsed

From our deep investigation, the collapse of PSNR (20.69 dB) and SSIM (0.5261) and regression of LPIPS (0.3124) are driven by **four quantifiable mechanisms**:

```
[Root Cause 1] USM Ground-Truth Sharpening (l1_gt_usm: True)
      │
      ├──> Forces Generator to output ringing halos on every edge
      └──> Benchmark evaluates against unsharpened GT ──> Massive PSNR/SSIM drop

[Root Cause 2] Skewed Loss Weighting (L1=0.8 vs GAN=0.15)
      │
      ├──> 50% stronger adversarial penalty + 20% weaker pixel penalty
      └──> Generator hallucinates aggressive micro-textures ──> LPIPS regression

[Root Cause 3] Degradation Overkill vs. Clean Bicubic Test Inputs
      │
      ├──> Model expects heavy JPEG (q=15-85) and noise (sigma=25)
      └──> Applied to clean bicubic LR ──> Erases true details, inserts fake textures

[Root Cause 4] Severe Data Scarcity (Only 55 Training Images)
      │
      └──> 3,378 epochs of repetition ──> Distribution overfitting, poor generalization
```

### Detailed Breakdown:
1. **The Ground-Truth USM Penalty:**
   In `realesrgan_model.py`:
   $$\mathcal{L}_{\text{pix}} = \|\hat{I}_{\text{SR}} - \text{USM}(I_{\text{GT}})\|_1$$
   The generator is explicitly trained to match an over-sharpened target containing artificial unsharp-mask overshoot halos. When evaluated against the raw ground truth $I_{\text{GT}}$, every single edge has an intentional pixel discrepancy of 10–30 intensity values, directly penalizing both PSNR (Mean Squared Error) and SSIM (luminance and structure comparison).
2. **The Perception-Distortion Shift (Blau & Michaeli, CVPR 2018):**
   Real-ESRNet (L1 loss only) achieves **23.41 dB PSNR** and **0.7065 SSIM**, but has poor perceptual score (NIQE 7.168, MUSIQ 56.86).
   Official Real-ESRGAN balances this with L1 weight `1.0` and GAN weight `0.10` (**21.75 dB, 0.6477 SSIM, 0.2349 LPIPS**).
   Our model tipped the scale excessively toward hallucination (L1 `0.8`, GAN `0.15`), jumping into the extreme perceptual zone (MUSIQ 65.49, NIQE 3.864) at a ruinous cost to fidelity.
3. **No Structural Loss:**
   L1 pixel loss only penalizes mean absolute error uniformly across flat and edge regions. Without a structural loss (e.g., SSIM loss or MS-SSIM loss), structural geometry and edge coherency are dominated entirely by the discriminator.

---

## 5. Phase 3 & 4 — Prioritized Improvement Plan & Experiment Design

To restore PSNR to $\ge 21.80$ dB, SSIM to $\ge 0.6500$, and LPIPS to $\le 0.2300$ while keeping NIQE $\le 4.2$ and MUSIQ $\ge 64.0$, we design a systematic 4-tier experiment matrix.

### 5.1 Key Scientific Modifications
1. **Deactivate USM on Ground Truth for Pixel & Perceptual Loss:**
   * Set `l1_gt_usm: False` and `percep_gt_usm: False`.
   * The generator will be supervised directly by true, authentic ground-truth pixel values.
2. **Re-balance the Loss Hierarchy:**
   * Elevate Pixel Loss weight to **`1.0`** (or **`1.2`**).
   * Calibrate GAN Loss weight down from `0.15` to **`0.08 – 0.10`**.
   * Calibrate Perceptual Loss weight to **`1.0`**.
3. **Introduce Structural Constraint (MS-SSIM / SSIM Loss):**
   * Add a differentiable SSIM/MS-SSIM loss component ($\mathcal{L}_{\text{struct}}$ with weight `0.1 – 0.2`) to explicitly optimize structural luminance and contrast preservation.
4. **Calibrate Degradation Ranges:**
   * Moderate the JPEG range from `[15, 85]` back to realistic `[30, 95]`.
   * Moderate noise ranges to prevent aggressive smoothing on clean inputs.
5. **Clean Dataset & Eliminate Leakage:**
   * Remove the 5 overlapping benchmark images from `datasets/train_HR`.
   * Augment training dataset with diverse, high-resolution clean patches to reach at least 150–200 diverse images for robust generalization.

### 5.2 Controlled Experiment Roadmap

```
                                  [Baseline Starting Point]
                                net_g_latest.pth (20.69 dB / 0.5261 SSIM)
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
             [Experiment A]                                    [Experiment B]
       Fidelity Restoration Pilot                         Structural Loss Injection
     • l1_gt_usm: False                                • Exp A setup
     • L1 weight: 1.0                                  • Add SSIM/MS-SSIM Loss (w=0.15)
     • GAN weight: 0.08                                • Evaluate structural boost
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                                       [Experiment C]
                                 Harmonic Pareto-Optimal
                              • Balanced L1 (1.0) + SSIM (0.1)
                              • Perceptual (1.0) + GAN (0.08)
                              • Mild degradation fine-tuning (5K iters)
                                              │
                                              ▼
                                       [Experiment D]
                                Final Benchmark Verification
                              • Full 20-image evaluation
                              • Statistical comparison vs. Official
```

* **Experiment A (Fidelity Restoration Pilot):**
  * Fine-tune starting from `experiments/pretrained_models/RealESRGAN_x4plus.pth` or current checkpoint.
  * Turn off `l1_gt_usm` and `percep_gt_usm`. Set L1 weight = `1.0`, GAN weight = `0.08`.
  * Expected Outcome: Immediate recovery of PSNR (+0.8 to +1.2 dB) and SSIM (+0.08 to +0.10).
* **Experiment B (Structural Loss Integration):**
  * Add differentiable Structural Loss (`1.0 - SSIM`) with weight `0.15`.
  * Expected Outcome: SSIM advances beyond `0.6500`, LPIPS drops below `0.2350`.
* **Experiment C (Harmonic Optimization):**
  * Comprehensive balance: $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{L1}} + 0.15 \cdot \mathcal{L}_{\text{SSIM}} + 1.0 \cdot \mathcal{L}_{\text{percep}} + 0.08 \cdot \mathcal{L}_{\text{GAN}}$.
  * Expected Outcome: State-of-the-art balance across all 5 metrics simultaneously.
* **Experiment D (Full Multi-Model Comparative Evaluation):**
  * Generate high-resolution side-by-side visual galleries and `reports/experiment_results.csv`.

---

## 6. Implementation Deliverables Structure (Prepared for Phase 4)

Once approved, the following cleanly structured, non-destructive files will be implemented:
1. `configs/train_improved.yml`: Fully parameterized configuration with structural loss, unsharpened GT supervision, and calibrated weights.
2. `realesrgan/models/realesrgan_model.py`: Augmented with differentiable `SSIMLoss` / `MS_SSIMLoss` support without breaking legacy configs.
3. `scripts/evaluate_benchmark.py`: Enhanced evaluation script supporting per-image breakdown, statistical variance, and standardized border cropping (`crop_border=4`).
4. `reports/experiment_results.csv`: Comprehensive tracking table recording all training iterations and metrics.

---

## 7. Decision Gate & Sign-Off Request

In strict compliance with the **Execution Instructions** ("Wait for my approval before starting expensive full training"):
* No training code or weights have been modified.
* All findings and root causes have been rigorously identified and documented above.

**Action Required:** Please review this audit report and confirm your approval to proceed to **Phase 4 (Implementation)** and **Phase 5/6 (Controlled Experiments A, B, and C)**.

