# Training Performance Optimization Guide

This guide contains tools and strategies to significantly speed up your diffusion model training.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install performance monitoring dependencies
python install_monitoring_deps.py

# OR manually install
pip install nvidia-ml-py3 psutil
```

### 2. Check Your Hardware
```bash
# Get personalized recommendations for your GPU
python performance_monitor_fixed.py
```

### 3. Start Optimized Training
```bash
# Use the optimized training script with recommended settings
python train_optimized.py --batch_size 32 --gradient_accumulation_steps 4 --mixed_precision fp16
```

## 📊 Performance Monitoring

### `performance_monitor_fixed.py`
- Analyzes your hardware (GPU, CPU, RAM)
- Provides personalized training recommendations
- Benchmarks data loading performance
- Works without GPUtil (uses nvidia-ml-py3 or nvidia-smi)

### `install_monitoring_deps.py`
- Automatically installs required monitoring dependencies
- Handles package installation errors gracefully

## 🏃‍♂️ Optimized Training

### `train_optimized.py`
- Easy-to-use training launcher with performance optimizations
- Automatically calculates effective batch sizes
- Supports all optimization parameters

### `training.py` (Enhanced)
- Added support for:
  - Mixed precision training (fp16/bf16)
  - Gradient accumulation
  - Multi-worker data loading
  - Memory optimizations

## 🎯 Optimization Strategies

### 1. Mixed Precision Training (2x Speed)
```bash
# Use fp16 for most GPUs
python train_optimized.py --mixed_precision fp16

# Use bf16 for newer GPUs (RTX 30xx+, better stability)
python train_optimized.py --mixed_precision bf16
```

### 2. Larger Effective Batch Size (1.5-3x Speed)
```bash
# Effective batch size = batch_size × gradient_accumulation_steps
python train_optimized.py --batch_size 32 --gradient_accumulation_steps 4  # Effective: 128
```

### 3. Optimized Data Loading (20-50% Speed)
```bash
# Use multiple workers for parallel data loading
python train_optimized.py --dataloader_num_workers 8
```

## 🖥️ Hardware-Specific Recommendations

### High-End GPUs (24+ GB VRAM)
```bash
# RTX 4090, A100, etc.
python train_optimized.py --batch_size 64 --gradient_accumulation_steps 2 --mixed_precision bf16
```

### Mid-Range GPUs (12-24 GB VRAM)
```bash
# RTX 4080, RTX 3090, etc.
python train_optimized.py --batch_size 32 --gradient_accumulation_steps 4 --mixed_precision fp16
```

### Budget GPUs (8-12 GB VRAM)
```bash
# RTX 4060 Ti, RTX 3070, etc.
python train_optimized.py --batch_size 16 --gradient_accumulation_steps 8 --mixed_precision fp16
```

## 📈 Expected Speed Improvements

| Optimization | Speed Improvement | Memory Impact |
|--------------|------------------|---------------|
| Mixed Precision (fp16/bf16) | 2x faster | 50% less memory |
| Larger Batch Size | 1.5-2x faster | More memory used |
| Multi-worker Data Loading | 20-50% faster | Negligible |
| **Combined** | **3-5x faster** | **Configurable** |

## 🔧 Troubleshooting

### GPU Memory Issues
```bash
# Reduce batch size, increase gradient accumulation
python train_optimized.py --batch_size 8 --gradient_accumulation_steps 16
```

### Slow Data Loading
```bash
# Reduce number of workers if causing issues
python train_optimized.py --dataloader_num_workers 2
```

### Package Installation Issues
```bash
# Use alternative installation
pip install nvidia-ml-py3 --no-cache-dir
# OR
conda install nvidia-ml-py3
```

## 📝 Files Overview

| File | Purpose |
|------|---------|
| `train_optimized.py` | Main optimized training launcher |
| `performance_monitor_fixed.py` | Hardware analysis and recommendations |
| `install_monitoring_deps.py` | Dependency installer |
| `speed_optimization_guide.py` | Comprehensive optimization strategies |
| `training.py` | Enhanced training script with optimizations |

## 🎓 Advanced Optimizations

### Enable Gradient Checkpointing (Memory Efficient)
Add to your model initialization:
```python
self.unet.enable_gradient_checkpointing()
```

### Use XFormers for Memory-Efficient Attention
```bash
pip install xformers
```
Then in model init:
```python
self.unet.enable_xformers_memory_efficient_attention()
```

### Monitor Training in Real-Time
```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Or use the monitoring script
python performance_monitor_fixed.py
```

## 🏆 Best Practices

1. **Start with safe settings** - Use `performance_monitor_fixed.py` to get recommendations
2. **Monitor GPU utilization** - Should be >90% during training
3. **Use mixed precision** - Almost always beneficial
4. **Optimize data loading** - Benchmark different worker counts
5. **Scale gradually** - Increase batch size until you hit memory limits

## 🆘 Need Help?

1. Run `python performance_monitor_fixed.py` for personalized recommendations
2. Check GPU memory usage with `nvidia-smi`
3. Monitor training progress with wandb
4. Start with conservative settings and scale up

The optimizations maintain the same model quality while significantly reducing training time!
