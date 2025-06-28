"""
Example: Benchmarking a Generated CUDA Kernel
"""
from cudagent.profiling.benchmarker import PerformanceBenchmarker

# Path to a generated kernel (replace with your actual kernel path)
kernel_path = "generated_kernels/matmul_kernel.cu"

# Initialize the benchmarker
benchmarker = PerformanceBenchmarker(iterations=100, warmup_iterations=10)

# Benchmark the kernel
result = benchmarker.benchmark_kernel(
    kernel_path=kernel_path,
    input_shape="1024,1024",
    output_shape="1024,1024"
)

print("Benchmark results:", result) 