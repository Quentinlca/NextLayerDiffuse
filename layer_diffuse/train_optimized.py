#!/usr/bin/env python3
"""
Optimized training script with performance improvements for faster training.

This script includes several optimizations:
1. Mixed precision training (fp16/bf16)
2. Gradient accumulation for larger effective batch sizes
3. Multi-worker data loading
4. Memory optimizations

Usage examples:
    # Basic fast training with mixed precision and larger effective batch size
    python train_optimized.py --batch_size 32 --gradient_accumulation_steps 4 --mixed_precision fp16

    # Maximum performance setup
    python train_optimized.py --batch_size 64 --gradient_accumulation_steps 2 --mixed_precision bf16 --dataloader_num_workers 8

    # For limited GPU memory
    python train_optimized.py --batch_size 16 --gradient_accumulation_steps 8 --mixed_precision fp16

    # With custom learning rate and scheduler settings
    python train_optimized.py --lr 0.0001 --warming_steps 500 --num_cycles 1.0

    # With wandb tags for experiment tracking
    python train_optimized.py --train_tags experiment_1 fast_training mixed_precision
"""

import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Optimized training launcher")
    
    # Performance optimization arguments
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Base batch size (increase if you have more GPU memory)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps (effective_batch_size = batch_size * gradient_accumulation_steps)")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                       help="Mixed precision mode (fp16 for older GPUs, bf16 for newer)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of data loading workers")
    
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
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"🚀 Starting optimized training with:")
    print(f"   Base batch size: {args.batch_size}")
    print(f"   Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Mixed precision: {args.mixed_precision}")
    print(f"   Data workers: {args.dataloader_num_workers}")
    print(f"   Model: {args.model_version}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Warming steps: {args.warming_steps}")
    print(f"   Num cycles: {args.num_cycles}")
    if args.train_tags:
        print(f"   Train tags: {', '.join(args.train_tags)}")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Dataset: {args.dataset_name}")
    
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
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
