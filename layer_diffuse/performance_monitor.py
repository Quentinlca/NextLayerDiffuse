#!/usr/bin/env python3
"""
Performance monitoring and GPU utilization script.
Use this to monitor your training performance and identify bottlenecks.
"""

import torch
import psutil
import time
import GPUtil
from datetime import datetime

def monitor_system():
    """Monitor system resources during training."""
    print("🔍 System Performance Monitor")
    print("=" * 50)
    
    # GPU Information
    if torch.cuda.is_available():
        print(f"🎮 GPU Information:")
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {gpu.name}")
            print(f"   Memory: {gpu.total_memory / 1024**3:.1f} GB")
        
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"   GPU {gpu.id} Utilization: {gpu.load*100:.1f}%")
            print(f"   GPU {gpu.id} Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
    else:
        print("❌ CUDA not available")
    
    # CPU Information
    print(f"\n🖥️  CPU Information:")
    print(f"   CPU Usage: {psutil.cpu_percent()}%")
    print(f"   CPU Cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical")
    
    # Memory Information
    memory = psutil.virtual_memory()
    print(f"\n💾 Memory Information:")
    print(f"   Total: {memory.total / 1024**3:.1f} GB")
    print(f"   Available: {memory.available / 1024**3:.1f} GB")
    print(f"   Used: {memory.used / 1024**3:.1f} GB ({memory.percent}%)")

def get_optimal_settings():
    """Suggest optimal training settings based on hardware."""
    print("\n💡 Recommended Settings:")
    print("=" * 50)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        gpu_memory_gb = gpu.total_memory / 1024**3
        
        print(f"Based on your GPU ({gpu.name}, {gpu_memory_gb:.1f} GB):")
        
        if gpu_memory_gb >= 24:  # RTX 4090, A100, etc.
            print("   Recommended: --batch_size 64 --gradient_accumulation_steps 2 --mixed_precision bf16")
            print("   High memory: --batch_size 128 --gradient_accumulation_steps 1 --mixed_precision bf16")
        elif gpu_memory_gb >= 16:  # RTX 4080, RTX 3090, etc.
            print("   Recommended: --batch_size 32 --gradient_accumulation_steps 4 --mixed_precision fp16")
            print("   Conservative: --batch_size 24 --gradient_accumulation_steps 4 --mixed_precision fp16")
        elif gpu_memory_gb >= 12:  # RTX 4070 Ti, RTX 3080, etc.
            print("   Recommended: --batch_size 24 --gradient_accumulation_steps 4 --mixed_precision fp16")
            print("   Conservative: --batch_size 16 --gradient_accumulation_steps 6 --mixed_precision fp16")
        elif gpu_memory_gb >= 8:   # RTX 4060 Ti, RTX 3070, etc.
            print("   Recommended: --batch_size 16 --gradient_accumulation_steps 6 --mixed_precision fp16")
            print("   Conservative: --batch_size 12 --gradient_accumulation_steps 8 --mixed_precision fp16")
        else:
            print("   Recommended: --batch_size 8 --gradient_accumulation_steps 8 --mixed_precision fp16")
            print("   Conservative: --batch_size 4 --gradient_accumulation_steps 16 --mixed_precision fp16")
        
        # Data loading recommendations
        cpu_cores = psutil.cpu_count(logical=False)
        print(f"\n   Data loading: --dataloader_num_workers {min(8, cpu_cores)}")
        
    else:
        print("   CPU-only training not recommended for diffusion models")

def benchmark_data_loading():
    """Benchmark data loading performance."""
    print("\n⏱️  Data Loading Benchmark:")
    print("=" * 50)
    
    try:
        from data_loaders.ModularCharatersDataLoader import get_modular_char_dataloader
        
        # Test different worker counts
        for num_workers in [0, 2, 4, 8]:
            print(f"Testing {num_workers} workers...")
            start_time = time.time()
            
            dataloader = get_modular_char_dataloader(
                dataset_name="QLeca/modular_characters_hairs_RGB",
                split="train",
                image_size=128,
                batch_size=16,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if num_workers > 0 else False,
                persistent_workers=True if num_workers > 0 else False,
            )
            
            # Time loading first few batches
            for i, batch in enumerate(dataloader):
                if i >= 5:  # Test first 5 batches
                    break
            
            elapsed = time.time() - start_time
            print(f"   {num_workers} workers: {elapsed:.2f}s for 5 batches")
            
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")

if __name__ == "__main__":
    monitor_system()
    get_optimal_settings()
    benchmark_data_loading()
