#!/usr/bin/env python3
"""
Test script for LLM Agent logging functionality.
"""

import os
from cudagent.agents import LLMOptimizationAgent, OptimizationRequest

def test_logging():
    """Test the logging functionality."""
    print("🧪 Testing LLM Agent Logging")
    print("=" * 40)
    
    # Initialize the agent
    agent = LLMOptimizationAgent()
    
    # Create a test request
    request = OptimizationRequest(
        operation_type="matmul",
        operation_info={
            "tensor_info": {
                "input_tensors": [
                    {"shape": (512, 512), "dtype": "float32"},
                    {"shape": (512, 512), "dtype": "float32"}
                ],
                "output_tensor": {"shape": (512, 512), "dtype": "float32"}
            }
        },
        optimization_goals=["maximize performance", "minimize memory usage"],
        constraints={"max_shared_memory": 16384}
    )
    
    print("📝 Request created:")
    print(f"   Operation: {request.operation_type}")
    print(f"   Input shapes: {request.operation_info['tensor_info']['input_tensors']}")
    print()
    
    # Generate optimized kernel
    print("🚀 Generating optimized kernel...")
    response = agent.optimize_kernel(request)
    
    print("✅ Generation completed!")
    print(f"   Provider: {response.provider_used}")
    print(f"   Generation time: {response.generation_time:.2f}s")
    print(f"   Confidence: {response.confidence_score}")
    print(f"   Kernel length: {len(response.optimized_kernel)} characters")
    print()
    
    # Check if files were created
    print("📁 Checking generated files:")
    
    # Check kernel files
    kernel_dir = "generated_kernels"
    if os.path.exists(kernel_dir):
        kernel_files = [f for f in os.listdir(kernel_dir) if f.endswith('.cu')]
        print(f"   Kernel files: {len(kernel_files)} found")
        for file in kernel_files:
            print(f"     - {file}")
    else:
        print("   ❌ No kernel directory found")
    
    # Check log files
    log_dir = "logs/optimizations"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        print(f"   Log files: {len(log_files)} found")
        for file in log_files:
            print(f"     - {file}")
    else:
        print("   ❌ No log directory found")
    
    print()
    print("🎉 Logging test completed!")
    print()
    print("📋 Summary:")
    print("   - Generated kernels are saved to: generated_kernels/")
    print("   - Detailed logs are saved to: logs/optimizations/")
    print("   - All files are automatically ignored by git")
    print("   - Files include timestamps and metadata")

if __name__ == "__main__":
    test_logging() 