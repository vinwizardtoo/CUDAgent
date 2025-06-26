#!/usr/bin/env python3
"""
Test script for the enhanced CUDAgent PyTorch to CUDA translator.
Demonstrates operation capture, enhanced parsing, and dynamic kernel generation.
"""

import torch
import torch.nn.functional as F
import logging
import time
from cudagent.core.enhanced_optimizer import EnhancedCUDAAgentOptimizer
from cudagent.parsers.operation_capture import OperationCapture

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_matmul_optimization():
    """Test enhanced matrix multiplication optimization with real operation capture."""
    print("\n" + "="*60)
    print("TEST 1: Enhanced Matrix Multiplication Optimization")
    print("="*60)
    
    # Initialize enhanced optimizer
    optimizer = EnhancedCUDAAgentOptimizer()
    
    # Create test tensors
    A = torch.randn(512, 1024, device='cuda')
    B = torch.randn(1024, 256, device='cuda')
    
    print(f"Input shapes: A{A.shape}, B{B.shape}")
    print(f"Expected output shape: ({A.shape[0]}, {B.shape[1]})")
    
    # Capture and optimize the matmul operation
    print("\nCapturing and optimizing matmul operation...")
    result = optimizer.capture_and_optimize_operation(torch.matmul, A, B)
    
    if result["success"]:
        print(f"✅ Optimization successful!")
        print(f"   Speedup: {result['speedup']:.2f}x")
        print(f"   Original time: {result['original_metrics']['execution_time']:.6f}s")
        print(f"   Optimized time: {result['optimized_metrics']['execution_time']:.6f}s")
        print(f"   Optimization time: {result['optimization_time']:.2f}s")
        
        # Show operation analysis
        operation_analysis = result['operation_analysis']
        print(f"\nOperation Analysis:")
        print(f"   Type: {operation_analysis['operation_type']}")
        print(f"   Complexity: {operation_analysis['complexity']}")
        print(f"   Parallelization: {operation_analysis['parallelization_strategy']['strategy']}")
        print(f"   Optimization opportunities: {operation_analysis['optimization_opportunities']}")
        
        # Show captured operation details
        captured_op = result['captured_operation']
        print(f"\nCaptured Operation Details:")
        print(f"   Function: {captured_op['function_name']}")
        print(f"   Input shapes: {captured_op['input_shapes']}")
        print(f"   Output shape: {captured_op['result_shape']}")
        print(f"   Device: {captured_op['device']}")
        
        # Show generated kernel info
        kernel = result['optimized_kernel']
        print(f"\nGenerated Kernel Info:")
        print(f"   Kernel length: {len(kernel)} characters")
        print(f"   Contains shared memory: {'__shared__' in kernel}")
        print(f"   Contains optimization comments: {'// Optimized' in kernel}")
        
    else:
        print(f"❌ Optimization failed: {result['error']}")

def test_enhanced_conv2d_optimization():
    """Test enhanced 2D convolution optimization with real operation capture."""
    print("\n" + "="*60)
    print("TEST 2: Enhanced 2D Convolution Optimization")
    print("="*60)
    
    # Initialize enhanced optimizer
    optimizer = EnhancedCUDAAgentOptimizer()
    
    # Create test tensors for convolution
    input_tensor = torch.randn(1, 3, 224, 224, device='cuda')
    conv_kernel = torch.randn(64, 3, 7, 7, device='cuda')
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Kernel shape: {conv_kernel.shape}")
    print(f"Expected output shape: (1, 64, 218, 218)")  # (224-7+1)
    
    # Capture and optimize the conv2d operation
    print("\nCapturing and optimizing conv2d operation...")
    result = optimizer.capture_and_optimize_operation(
        F.conv2d, input_tensor, conv_kernel, 
        padding=0, stride=1, bias=None
    )
    
    if result["success"]:
        print(f"✅ Optimization successful!")
        print(f"   Speedup: {result['speedup']:.2f}x")
        print(f"   Original time: {result['original_metrics']['execution_time']:.6f}s")
        print(f"   Optimized time: {result['optimized_metrics']['execution_time']:.6f}s")
        
        # Show operation analysis
        operation_analysis = result['operation_analysis']
        print(f"\nOperation Analysis:")
        print(f"   Type: {operation_analysis['operation_type']}")
        print(f"   Complexity: {operation_analysis['complexity']}")
        print(f"   Parallelization: {operation_analysis['parallelization_strategy']['strategy']}")
        
        # Show convolution parameters
        conv_params = operation_analysis['operation_info'].get('convolution_params', {})
        print(f"\nConvolution Parameters:")
        print(f"   Kernel size: {conv_params.get('kernel_size', 'N/A')}")
        print(f"   Stride: {conv_params.get('stride', 'N/A')}")
        print(f"   Padding: {conv_params.get('padding', 'N/A')}")
        
    else:
        print(f"❌ Optimization failed: {result['error']}")

def test_enhanced_activation_optimization():
    """Test enhanced activation function optimization with real operation capture."""
    print("\n" + "="*60)
    print("TEST 3: Enhanced Activation Function Optimization")
    print("="*60)
    
    # Initialize enhanced optimizer
    optimizer = EnhancedCUDAAgentOptimizer()
    
    # Create test tensor
    input_tensor = torch.randn(1000, 1000, device='cuda')
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Test ReLU activation
    print("\nCapturing and optimizing ReLU operation...")
    result = optimizer.capture_and_optimize_operation(F.relu, input_tensor)
    
    if result["success"]:
        print(f"✅ ReLU optimization successful!")
        print(f"   Speedup: {result['speedup']:.2f}x")
        print(f"   Original time: {result['original_metrics']['execution_time']:.6f}s")
        print(f"   Optimized time: {result['optimized_metrics']['execution_time']:.6f}s")
        
        # Show operation analysis
        operation_analysis = result['operation_analysis']
        print(f"\nOperation Analysis:")
        print(f"   Type: {operation_analysis['operation_type']}")
        print(f"   Complexity: {operation_analysis['complexity']}")
        print(f"   Parallelization: {operation_analysis['parallelization_strategy']['strategy']}")
        
    else:
        print(f"❌ ReLU optimization failed: {result['error']}")

def test_operation_capture_demo():
    """Demonstrate the operation capture system."""
    print("\n" + "="*60)
    print("TEST 4: Operation Capture System Demo")
    print("="*60)
    
    # Initialize operation capture
    capture = OperationCapture()
    
    # Create test tensors
    A = torch.randn(100, 200, device='cuda')
    B = torch.randn(200, 150, device='cuda')
    C = torch.randn(100, 150, device='cuda')
    
    print("Capturing operations...")
    
    # Capture matmul operation
    result1 = capture.capture_operation(torch.matmul, A, B)
    print(f"✅ Captured matmul: A{A.shape} @ B{B.shape} -> {result1.shape}")
    
    # Capture add operation
    result2 = capture.capture_operation(torch.add, result1, C)
    print(f"✅ Captured add: {result1.shape} + {C.shape} -> {result2.shape}")
    
    # Capture ReLU operation
    result3 = capture.capture_operation(F.relu, result2)
    print(f"✅ Captured relu: {result2.shape} -> {result3.shape}")
    
    # Show operation history
    history = capture.get_operation_history()
    print(f"\nOperation History ({len(history)} operations):")
    
    for i, op in enumerate(history):
        print(f"  {i+1}. {op['function_name']}: {op['input_shapes']} -> {op['result_shape']}")
        print(f"     Device: {op['device']}, Dtype: {op['input_dtypes'][0]}")
        
        # Show operation-specific parameters
        params = op.get('operation_params', {})
        if params:
            print(f"     Parameters: {params}")

def test_enhanced_optimizer_comparison():
    """Compare enhanced optimizer with basic functionality."""
    print("\n" + "="*60)
    print("TEST 5: Enhanced vs Basic Optimizer Comparison")
    print("="*60)
    
    # Test with different matrix sizes
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    for M, N in sizes:
        print(f"\nTesting {M}x{N} matrix multiplication...")
        
        # Create tensors
        A = torch.randn(M, N, device='cuda')
        B = torch.randn(N, M, device='cuda')
        
        # Initialize enhanced optimizer
        optimizer = EnhancedCUDAAgentOptimizer()
        
        # Optimize
        result = optimizer.capture_and_optimize_operation(torch.matmul, A, B)
        
        if result["success"]:
            print(f"  ✅ {M}x{N}: {result['speedup']:.2f}x speedup")
            print(f"     Original: {result['original_metrics']['execution_time']:.6f}s")
            print(f"     Optimized: {result['optimized_metrics']['execution_time']:.6f}s")
        else:
            print(f"  ❌ {M}x{N}: Failed - {result['error']}")

def main():
    """Run all enhanced CUDAgent tests."""
    print("🚀 Enhanced CUDAgent PyTorch to CUDA Translator Tests")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Tests will be limited.")
        return
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ PyTorch version: {torch.__version__}")
    
    try:
        # Run tests
        test_operation_capture_demo()
        test_enhanced_matmul_optimization()
        test_enhanced_conv2d_optimization()
        test_enhanced_activation_optimization()
        test_enhanced_optimizer_comparison()
        
        print("\n" + "="*60)
        print("🎉 All Enhanced CUDAgent tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 