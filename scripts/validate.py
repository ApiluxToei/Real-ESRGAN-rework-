"""
Standalone Validation Script for Real-ESRGAN Checkpoints
Evaluates a given checkpoint (.pth) on the 20 benchmark test images.
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Validate a Real-ESRGAN checkpoint on 20 benchmark images")
    parser.add_argument('--model_path', type=str, required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument('--name', type=str, default="Candidate Model", help="Display name for the model")
    parser.add_argument('--gt', type=str, default="datasets/benchmark_20/gt")
    parser.add_argument('--lr', type=str, default="datasets/benchmark_20/lr")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"Error: Checkpoint file not found: {args.model_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        'scripts/evaluate_benchmark.py',
        '--model_paths', args.model_path,
        '--names', args.name,
        '--gt', args.gt,
        '--lr', args.lr,
        '--device', args.device
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()

