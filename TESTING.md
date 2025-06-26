# CUDAgent Testing Guide

This guide explains how to test the CUDAgent PyTorch to CUDA translator system on both CUDA-enabled and CPU-only systems.

## Prerequisites

### For CUDA Testing
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.0 or higher)
- PyTorch with CUDA support
- All dependencies from `requirements.txt`

### For CPU Testing
- Any system with Python 3.8+
- PyTorch (CPU version is fine)
- All dependencies from `requirements.txt`

## Environment Setup

1. **Activate your conda environment:**
   ```bash
   conda activate cudagent
   ```

2. **Verify PyTorch installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Quick Start - Testing Commands

### 🖥️ CPU Testing (Works on any system)
```bash
# Test the enhanced system on CPU
python test_enhanced_cudagent_cpu.py

# Test basic functionality on CPU
python test_cudagent.py
```

### 🎮 CUDA Testing (Requires NVIDIA GPU)
```bash
# Test the enhanced system on CUDA
python test_enhanced_cudagent.py
```

## Detailed Testing Options

### Option 1: CPU-Only Testing (Recommended for most users)

**Command:**
```bash
python test_enhanced_cudagent_cpu.py
```

**What this tests:**
- ✅ Operation capture system
- ✅ Enhanced parser analysis
- ✅ Kernel generation (without actual CUDA execution)
- ✅ Enhanced optimizer analysis
- ✅ All core functionality without requiring CUDA

**Expected Output:**
```
🚀 Enhanced CUDAgent PyTorch to CUDA Translator Tests (CPU Version)
============================================================
✅ PyTorch version: 2.2.2
✅ CUDA available: False
⚠️  Running in CPU-only mode (CUDA not available)

============================================================
TEST 1: Operation Capture System Demo
============================================================
Capturing operations...
✅ Captured matmul: Atorch.Size([100, 200]) @ Btorch.Size([200, 150]) -> torch.Size([100, 150])
✅ Captured add: torch.Size([100, 150]) + torch.Size([100, 150]) -> torch.Size([100, 150])
✅ Captured relu: torch.Size([100, 150]) -> torch.Size([100, 150])

Operation History (3 operations):
  1. matmul: [torch.Size([100, 200]), torch.Size([200, 150])] -> torch.Size([100, 150])
     Device: cpu, Dtype: torch.float32
     Parameters: {'A_shape': torch.Size([100, 200]), 'B_shape': torch.Size([200, 150]), 'M': 100, 'K': 200, 'N': 150, 'transpose_A': False, 'transpose_B': False}
```

### Option 2: CUDA Testing (For NVIDIA GPU users)

**Command:**
```bash
python test_enhanced_cudagent.py
```

**What this tests:**
- ✅ All CPU features PLUS actual CUDA execution
- ✅ Real performance benchmarking
- ✅ Speedup measurements
- ✅ Actual CUDA kernel compilation and execution

**Expected Output:**
```
🚀 Enhanced CUDAgent PyTorch to CUDA Translator Tests
============================================================
✅ CUDA available: NVIDIA GeForce RTX 3080
✅ PyTorch version: 2.2.2

============================================================
TEST 1: Enhanced Matrix Multiplication Optimization
============================================================
Input shapes: A(512, 1024), B(1024, 256)
Expected output shape: (512, 256)

Capturing and optimizing matmul operation...
✅ Optimization successful!
   Speedup: 2.45x
   Original time: 0.000156s
   Optimized time: 0.000064s
   Optimization time: 0.12s

Operation Analysis:
   Type: matmul
   Complexity: high
   Parallelization: 2D_grid
   Optimization opportunities: ['high_parallelization', 'mixed_precision']

Captured Operation Details:
   Function: matmul
   Input shapes: [torch.Size([512, 1024]), torch.Size([1024, 256])]
   Output shape: torch.Size([512, 256])
   Device: cuda:0

Generated Kernel Info:
   Kernel length: 1503 characters
   Contains shared memory: True
   Contains optimization comments: True
```

### Option 3: Basic Functionality Testing

**Command:**
```bash
python test_cudagent.py
```

**What this tests:**
- ✅ Basic optimizer functionality
- ✅ Simple tensor analysis
- ✅ Core system components

## Testing Commands Summary

| Test Type | Command | Requirements | What it tests |
|-----------|---------|--------------|---------------|
| **CPU Enhanced** | `python test_enhanced_cudagent_cpu.py` | Any system | Full enhanced system (no CUDA execution) |
| **CUDA Enhanced** | `python test_enhanced_cudagent.py` | NVIDIA GPU + CUDA | Full enhanced system with CUDA execution |
| **Basic** | `python test_cudagent.py` | Any system | Basic functionality only |

## Test Components Explained

### 1. Operation Capture System
- **What it does:** Captures real PyTorch operations with full context
- **Tests:** `test_operation_capture_demo()`
- **Key features:**
  - Records function name, input shapes, output shapes
  - Extracts operation-specific parameters
  - Maintains operation history

### 2. Enhanced Parser
- **What it does:** Analyzes captured operations for optimization opportunities
- **Tests:** `test_enhanced_parser_analysis()`
- **Key features:**
  - Determines operation type (matmul, conv2d, add, etc.)
  - Calculates computational complexity
  - Identifies optimization strategies

### 3. Enhanced Kernel Generator
- **What it does:** Generates optimized CUDA kernels based on operation analysis
- **Tests:** `test_enhanced_kernel_generation()`
- **Key features:**
  - Dynamic block/grid size optimization
  - Shared memory usage
  - Operation-specific optimizations

### 4. Enhanced Optimizer
- **What it does:** Orchestrates the entire optimization pipeline
- **Tests:** `test_enhanced_optimizer_analysis()`
- **Key features:**
  - End-to-end optimization workflow
  - Performance benchmarking
  - Speedup calculation

## Understanding Test Output

### Operation Type Detection
The system correctly identifies operation types:
```
Operation type: matmul
```
This comes from the `__name__` attribute of the PyTorch function being captured.

### Performance Metrics
- **Speedup:** How much faster the optimized version is
- **Original time:** Time for unoptimized PyTorch operation
- **Optimized time:** Time for optimized CUDA kernel
- **Optimization time:** Time spent generating the optimization

### Kernel Analysis
- **Kernel length:** Size of generated CUDA code
- **Shared memory:** Whether the kernel uses shared memory
- **Optimization comments:** Whether the kernel includes optimization hints

## Troubleshooting

### Common Issues

1. **CUDA not available:**
   ```
   ❌ CUDA is not available. Tests will be limited.
   ```
   **Solution:** Use `python test_enhanced_cudagent_cpu.py` instead

2. **Import errors:**
   ```
   ModuleNotFoundError: No module named 'cudagent'
   ```
   **Solution:** Make sure you're in the correct directory and conda environment is activated

3. **PyTorch version issues:**
   ```
   RuntimeError: Expected all tensors to be on the same device
   ```
   **Solution:** Check PyTorch installation and CUDA compatibility

### Debug Mode

To see detailed logging, set the log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Expectations

### CUDA Testing
- **Speedup:** 2-5x for matrix operations
- **Kernel generation:** <1 second
- **Memory usage:** Varies by tensor size

### CPU Testing
- **Speedup:** Not applicable (no actual CUDA execution)
- **Analysis time:** <1 second
- **Memory usage:** Minimal

## Advanced Testing

### Custom Operations
You can test custom operations by modifying the test files:

```python
# Test custom operation
result = optimizer.capture_and_optimize_operation(
    torch.special.exp, input_tensor
)
```

### Different Tensor Sizes
Test with various tensor dimensions:
```python
sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
for M, N in sizes:
    A = torch.randn(M, N, device='cuda')
    B = torch.randn(N, M, device='cuda')
    # Test optimization...
```

### Memory Profiling
For detailed memory analysis, add memory profiling:
```python
import torch.profiler
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    # Your test code here
    pass
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Continuous Integration

For automated testing, you can run:
```bash
# Run all tests
python -m pytest test_*.py -v

# Run with coverage
python -m pytest test_*.py --cov=cudagent --cov-report=html
```

## Next Steps

After successful testing:
1. Review the generated CUDA kernels in the output
2. Analyze the optimization opportunities identified
3. Compare performance with different tensor sizes
4. Explore the operation capture history for insights

For more information, see the main [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) files. 