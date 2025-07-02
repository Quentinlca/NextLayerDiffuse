#!/usr/bin/env python3
"""
CUDA-Safe Optimized Training Script
Fixed version that avoids CUDA initialization errors by using safe defaults.
"""

import subprocess
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="CUDA-Safe Optimized training launcher")
    
    # Performance optimization arguments with CUDA-safe defaults
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Base batch size (reduced default for CUDA safety)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps (increased to compensate for smaller batch)")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                       help="Mixed precision mode")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                       help="Number of data loading workers (0 = no multiprocessing, CUDA-safe)")
    
    # Training parameters
    parser.add_argument("--model_version", type=str, default="DDIMNextTokenV1",
                       choices=["DDPMNextTokenV1", "DDPMNextTokenV2", "DDPMNextTokenV3", "DDIMNextTokenV1"])
    parser.add_argument("--train_size", type=int, default=16000)
    parser.add_argument("--val_size", type=int, default=1600)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--warming_steps", type=int, default=1000, 
                       help="Learning rate scheduler warming steps")
    parser.add_argument("--num_cycles", type=float, default=0.5,
                       help="Number of cycles for the cosine learning rate scheduler")
    parser.add_argument("--train_tags", type=str, nargs='*', default=None,
                       help="Tags for the training run (optional, can be used for wandb tagging)")
    parser.add_argument("--dataset_name", type=str, default="QLeca/modular_characters_hairs_RGB")
    
    args = parser.parse_args()
    
    # Set CUDA environment variables for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"🚀 Starting CUDA-safe optimized training with:")
    print(f"   Base batch size: {args.batch_size}")
    print(f"   Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Mixed precision: {args.mixed_precision}")
    print(f"   Data workers: {args.dataloader_num_workers} (0 = CUDA-safe)")
    print(f"   Model: {args.model_version}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Warming steps: {args.warming_steps}")
    print(f"   Num cycles: {args.num_cycles}")
    if args.train_tags:
        print(f"   Train tags: {', '.join(args.train_tags)}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Dataset: {args.dataset_name}")
    
    if args.dataloader_num_workers == 0:
        print("✅ Using CUDA-safe configuration (no multiprocessing workers)")
    else:
        print("⚠️  Warning: Using multiprocessing workers may cause CUDA errors")
    
    # Build command
    cmd = [
        sys.executable, "training.py",
        "--batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--mixed_precision", args.mixed_precision,
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--model_version", args.model_version,
        "--train_size", str(args.train_size),
        "--val_size", str(args.val_size),
        "--num_epochs", str(args.num_epochs),
        "--lr", str(args.lr),
        "--warming_steps", str(args.warming_steps),
        "--num_cycles", str(args.num_cycles),
        "--dataset_name", args.dataset_name,
    ]
    
    # Add train_tags if provided
    if args.train_tags:
        cmd.extend(["--train_tags"] + args.train_tags)
    
    print(f"\n🏃‍♂️ Running command: {' '.join(cmd)}")
    
    # Run the training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("\n🔧 CUDA Error Troubleshooting:")
        print("1. Try with even smaller batch size: --batch_size 8")
        print("2. Ensure no multiprocessing: --dataloader_num_workers 0")
        print("3. Check GPU memory: nvidia-smi")
        print("4. Restart Python to clear CUDA context")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
