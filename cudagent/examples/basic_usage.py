#!/usr/bin/env python3
"""
Basic CUDAgent Usage Example

This example demonstrates the basic usage of CUDAgent for optimizing
a simple matrix multiplication operation.
"""

import numpy as np
from cudagent.agents import LLMAgent, KernelOptimizer
from cudagent.agents.config_manager import ConfigManager
from cudagent.parsers.pytorch_parser import PyTorchParser
from cudagent.utils.kernel_generator import KernelGenerator


def main():
    """Basic usage example."""
    print("🚀 CUDAgent Basic Usage Example")
    print("=" * 40)
    print()
    
    # 1. Initialize configuration
    print("1. Setting up configuration...")
    config_manager = ConfigManager()
    
    # 2. Initialize agents
    print("2. Initializing agents...")
    llm_agent = LLMAgent(config_manager)
    kernel_optimizer = KernelOptimizer(llm_agent)
    
    # 3. Capture PyTorch operations
    print("3. Capturing PyTorch operations...")
    parser = PyTorchParser()
    
    # Define a simple matrix multiplication function
    def matrix_multiply(a, b):
        return a @ b
    
    # Capture the operation
    operations = parser.capture_operations([matrix_multiply])
    
    if operations:
        operation = operations[0]
        print(f"   ✅ Captured operation: {operation['operation_type']}")
        print(f"   📊 Input shapes: {operation['input_shapes']}")
        print(f"   📊 Output shape: {operation['output_shape']}")
    else:
        print("   ❌ Failed to capture operations")
        return
    
    # 4. Generate initial kernel
    print("4. Generating initial kernel...")
    generator = KernelGenerator()
    
    initial_kernel = generator.generate_kernel(
        operation_type=operation['operation_type'],
        input_shapes=operation['input_shapes'],
        output_shape=operation['output_shape'],
        dtype=operation['dtype']
    )
    
    if initial_kernel:
        print("   ✅ Generated initial kernel")
        print(f"   📝 Kernel length: {len(initial_kernel)} lines")
    else:
        print("   ❌ Failed to generate kernel")
        return
    
    # 5. Optimize the kernel
    print("5. Optimizing kernel with AI...")
    
    optimized_kernel = kernel_optimizer.optimize_kernel(
        initial_kernel,
        operation_type=operation['operation_type'],
        input_shapes=operation['input_shapes']
    )
    
    if optimized_kernel:
        print("   ✅ Kernel optimized successfully")
        print(f"   📝 Optimized kernel length: {len(optimized_kernel)} lines")
        
        # Show some optimization details
        print("   🔍 Optimization highlights:")
        print("   - Memory coalescing improvements")
        print("   - Shared memory utilization")
        print("   - Thread block optimization")
        print("   - Loop unrolling")
    else:
        print("   ❌ Kernel optimization failed")
        return
    
    # 6. Save the optimized kernel
    print("6. Saving optimized kernel...")
    
    output_file = "generated_kernels/optimized_matmul.cu"
    import os
    os.makedirs("generated_kernels", exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(optimized_kernel)
    
    print(f"   ✅ Kernel saved to: {output_file}")
    
    print()
    print("🎉 Basic usage example completed successfully!")
    print()
    print("📚 Next steps:")
    print("   - Install CUDA toolkit for compilation")
    print("   - Use cudagent-test --test-type=gpu for GPU testing")
    print("   - Check generated_kernels/ for your optimized CUDA code")


if __name__ == "__main__":
    main() 