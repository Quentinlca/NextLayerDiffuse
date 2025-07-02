#!/usr/bin/env python3
"""
Fixed Performance monitoring and GPU utilization script.
Use this to monitor your training performance and identify bottlenecks.
"""

import torch
import psutil
import time
import subprocess
from datetime import datetime

def get_gpu_info_nvidia_smi():
    """Alternative GPU info using nvidia-smi command."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        gpu_info.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'memory_total': int(parts[2]),
                            'memory_used': int(parts[3]),
                            'gpu_util': float(parts[4]),
                            'memory_util': float(parts[5])
                        })
            return gpu_info
        else:
            return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

def get_gpu_info_pynvml():
    """Alternative GPU info using nvidia-ml-py3."""
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        gpu_info = []
        device_count = nvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_info.append({
                'id': i,
                'name': name,
                'memory_total': mem_info.total // 1024**2,  # Convert to MB
                'memory_used': mem_info.used // 1024**2,
                'gpu_util': float(util.gpu),
                'memory_util': float(util.memory)
            })
        
        return gpu_info
    except ImportError:
        return None
    except Exception:
        return None

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
        
        # Try different methods to get GPU utilization
        gpu_info = get_gpu_info_pynvml()
        
        # Try nvidia-smi if pynvml failed
        if gpu_info is None:
            gpu_info = get_gpu_info_nvidia_smi()
        
        # Display GPU info
        if gpu_info:
            for gpu in gpu_info:
                print(f"   GPU {gpu['id']} Utilization: {gpu['gpu_util']:.1f}%")
                print(f"   GPU {gpu['id']} Memory: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_util']:.1f}%)")
        else:
            print("   ⚠️  GPU utilization monitoring unavailable")
            print("   💡 Install nvidia-ml-py3: pip install nvidia-ml-py3")
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
        cpu_cores = psutil.cpu_count(logical=False) or 4  # Default to 4 if None
        print(f"\n   Data loading: --dataloader_num_workers {min(8, cpu_cores)}")
        
        # Generate example commands
        print(f"\n🚀 Example Commands:")
        if gpu_memory_gb >= 16:
            print("   Fast training:")
            print("   python train_optimized.py --batch_size 32 --gradient_accumulation_steps 4 --mixed_precision fp16")
        else:
            print("   Memory-efficient training:")
            print("   python train_optimized.py --batch_size 16 --gradient_accumulation_steps 8 --mixed_precision fp16")
        
    else:
        print("   CPU-only training not recommended for diffusion models")

def benchmark_data_loading():
    """Benchmark data loading performance."""
    print("\n⏱️  Data Loading Benchmark:")
    print("=" * 50)
    
    try:
        from data_loaders.ModularCharatersDataLoader import get_modular_char_dataloader
        
        # Test different worker counts
        best_time = float('inf')
        best_workers = 0
        
        for num_workers in [0, 2, 4, 8]:
            try:
                print(f"Testing {num_workers} workers...", end=" ")
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
                    if i >= 3:  # Test first 3 batches
                        break
                
                elapsed = time.time() - start_time
                print(f"{elapsed:.2f}s")
                
                if elapsed < best_time:
                    best_time = elapsed
                    best_workers = num_workers
                    
            except Exception as e:
                print(f"Failed: {e}")
        
        print(f"\n🏆 Best performance: {best_workers} workers ({best_time:.2f}s)")
        print(f"   Recommended: --dataloader_num_workers {best_workers}")
            
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
        print("   💡 Make sure you're running from the layer_diffuse directory")

def install_missing_packages():
    """Install missing packages for better GPU monitoring."""
    print("\n📦 Package Installation Helper:")
    print("=" * 50)
    
    print("For better GPU monitoring, install:")
    print("   pip install nvidia-ml-py3")
    print()
    print("For additional dependencies:")
    print("   pip install psutil torch")
    print()
    print("Note: GPUtil has installation issues, so we use nvidia-ml-py3 instead.")

if __name__ == "__main__":
    monitor_system()
    get_optimal_settings()
    benchmark_data_loading()
    install_missing_packages()
