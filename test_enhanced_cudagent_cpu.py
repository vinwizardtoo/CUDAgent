#!/usr/bin/env python3
"""
CPU-only test script for the enhanced CUDAgent PyTorch to CUDA translator.
Demonstrates operation capture, enhanced parsing, and dynamic kernel generation without CUDA.
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

def test_operation_capture_demo():
    """Demonstrate the operation capture system."""
    print("\n" + "="*60)
    print("TEST 1: Operation Capture System Demo")
    print("="*60)
    
    # Initialize operation capture
    capture = OperationCapture()
    
    # Create test tensors (CPU)
    A = torch.randn(100, 200)
    B = torch.randn(200, 150)
    C = torch.randn(100, 150)
    
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

def test_enhanced_parser_analysis():
    """Test the enhanced parser with captured operations."""
    print("\n" + "="*60)
    print("TEST 2: Enhanced Parser Analysis")
    print("="*60)
    
    from cudagent.parsers.pytorch_parser import EnhancedPyTorchOperationParser
    
    # Initialize parser
    parser = EnhancedPyTorchOperationParser()
    
    # Initialize operation capture
    capture = OperationCapture()
    
    # Create test tensors
    A = torch.randn(512, 1024)
    B = torch.randn(1024, 256)
    
    # Capture matmul operation
    result = capture.capture_operation(torch.matmul, A, B)
    operation_info = capture.get_last_operation()
    
    # Parse the captured operation
    print("Parsing captured matmul operation...")
    operation_analysis = parser.parse_captured_operation(operation_info)
    
    print(f"✅ Parsing successful!")
    print(f"   Operation type: {operation_analysis['operation_type']}")
    print(f"   Complexity: {operation_analysis['complexity']}")
    print(f"   Is supported: {operation_analysis['is_supported']}")
    
    # Show detailed analysis
    op_info = operation_analysis['operation_info']
    print(f"\nDetailed Analysis:")
    print(f"   Input shapes: {op_info['input_shapes']}")
    print(f"   Output shape: {op_info['output_shape']}")
    print(f"   Compute intensity: {op_info['compute_intensity']:,}")
    print(f"   Memory access: {op_info['memory_access']:,}")
    print(f"   Arithmetic intensity: {op_info['arithmetic_intensity']:.2f}")
    print(f"   Parallelization: {op_info['parallelization']}")
    print(f"   Block size optimization: {op_info['block_size_optimization']}")
    print(f"   Optimization strategies: {op_info['optimization_strategies']}")
    
    # Show tensor info
    tensor_info = operation_analysis['tensor_info']
    print(f"\nTensor Information:")
    print(f"   Input tensors: {len(tensor_info['input_tensors'])}")
    for i, tensor in enumerate(tensor_info['input_tensors']):
        print(f"     Tensor {i+1}: {tensor['shape']}, {tensor['dtype']}, {tensor['device']}")
    print(f"   Output tensor: {tensor_info['output_tensor']['shape']}, {tensor_info['output_tensor']['dtype']}")

def test_enhanced_kernel_generation():
    """Test the enhanced kernel generator with captured operations."""
    print("\n" + "="*60)
    print("TEST 3: Enhanced Kernel Generation")
    print("="*60)
    
    from cudagent.utils.enhanced_kernel_generator import EnhancedCUDAKernelGenerator
    from cudagent.parsers.pytorch_parser import EnhancedPyTorchOperationParser
    
    # Initialize components
    parser = EnhancedPyTorchOperationParser()
    kernel_generator = EnhancedCUDAKernelGenerator()
    capture = OperationCapture()
    
    # Test different operations
    operations = [
        ("matmul", torch.matmul, [torch.randn(256, 512), torch.randn(512, 128)]),
        ("add", torch.add, [torch.randn(100, 100), torch.randn(100, 100)]),
        ("relu", F.relu, [torch.randn(200, 200)]),
    ]
    
    for op_name, op_func, args in operations:
        print(f"\nTesting {op_name} operation...")
        
        # Capture operation
        result = capture.capture_operation(op_func, *args)
        operation_info = capture.get_last_operation()
        
        # Parse operation
        operation_analysis = parser.parse_captured_operation(operation_info)
        
        # Generate kernel
        kernel = kernel_generator.generate_kernel(operation_analysis)
        
        print(f"✅ Generated kernel for {op_name}")
        print(f"   Kernel length: {len(kernel)} characters")
        print(f"   Contains shared memory: {'__shared__' in kernel}")
        print(f"   Contains optimization comments: {'// Optimized' in kernel}")
        print(f"   Contains block size config: {'Block size:' in kernel}")
        
        # Show a snippet of the kernel
        lines = kernel.split('\n')
        print(f"   First few lines:")
        for i, line in enumerate(lines[:5]):
            if line.strip():
                print(f"     {line.strip()}")

def test_conv2d_analysis():
    """Test enhanced analysis with 2D convolution."""
    print("\n" + "="*60)
    print("TEST 4: 2D Convolution Analysis")
    print("="*60)
    
    from cudagent.parsers.pytorch_parser import EnhancedPyTorchOperationParser
    
    # Initialize parser
    parser = EnhancedPyTorchOperationParser()
    
    # Initialize operation capture
    capture = OperationCapture()
    
    # Create test tensors for convolution
    input_tensor = torch.randn(1, 3, 224, 224)
    conv_kernel = torch.randn(64, 3, 7, 7)
    
    # Capture conv2d operation
    result = capture.capture_operation(
        F.conv2d, input_tensor, conv_kernel, 
        padding=3, stride=2, bias=None
    )
    operation_info = capture.get_last_operation()
    
    # Parse the captured operation
    print("Parsing captured conv2d operation...")
    operation_analysis = parser.parse_captured_operation(operation_info)
    
    print(f"✅ Parsing successful!")
    print(f"   Operation type: {operation_analysis.get('operation_type', 'unknown')}")
    print(f"   Complexity: {operation_analysis.get('complexity', 'unknown')}")
    
    # Show detailed analysis
    op_info = operation_analysis.get('operation_info', {})
    if op_info:
        print(f"\nDetailed Analysis:")
        print(f"   Input shapes: {op_info.get('input_shapes', 'N/A')}")
        print(f"   Output shape: {op_info.get('output_shape', 'N/A')}")
        print(f"   Compute intensity: {op_info.get('compute_intensity', 'N/A'):,}" if isinstance(op_info.get('compute_intensity'), int) else f"   Compute intensity: {op_info.get('compute_intensity', 'N/A')}")
        print(f"   Memory access: {op_info.get('memory_access', 'N/A'):,}" if isinstance(op_info.get('memory_access'), int) else f"   Memory access: {op_info.get('memory_access', 'N/A')}")
        print(f"   Arithmetic intensity: {op_info.get('arithmetic_intensity', 'N/A'):.2f}" if isinstance(op_info.get('arithmetic_intensity'), (int, float)) else f"   Arithmetic intensity: {op_info.get('arithmetic_intensity', 'N/A')}")
        print(f"   Parallelization: {op_info.get('parallelization', 'N/A')}")
        
        # Show convolution parameters
        conv_params = op_info.get('convolution_params', {})
        print(f"\nConvolution Parameters:")
        print(f"   Kernel size: {conv_params.get('kernel_size', 'N/A')}")
        print(f"   Stride: {conv_params.get('stride', 'N/A')}")
        print(f"   Padding: {conv_params.get('padding', 'N/A')}")
        print(f"   Dilation: {conv_params.get('dilation', 'N/A')}")
        print(f"   Groups: {conv_params.get('groups', 'N/A')}")
    else:
        print(f"   ❌ Detailed analysis not available due to parsing error")

def test_enhanced_optimizer_analysis():
    """Test the enhanced optimizer analysis capabilities."""
    print("\n" + "="*60)
    print("TEST 5: Enhanced Optimizer Analysis")
    print("="*60)
    
    # Initialize enhanced optimizer
    optimizer = EnhancedCUDAAgentOptimizer()
    
    # Create test tensors
    A = torch.randn(256, 512)
    B = torch.randn(512, 128)
    
    print("Testing enhanced optimizer analysis...")
    
    # Capture and analyze operation (without actual optimization since no CUDA)
    try:
        # This will work up to the point where CUDA is needed
        result_tensor = optimizer.operation_capture.capture_operation(torch.matmul, A, B)
        operation_info = optimizer.operation_capture.get_last_operation()
        
        # Parse the operation
        operation_analysis = optimizer.parser.parse_captured_operation(operation_info)
        
        # Generate kernel
        kernel = optimizer.kernel_generator.generate_kernel(operation_analysis)
        
        print(f"✅ Analysis successful!")
        print(f"   Operation captured: {operation_info['function_name']}")
        print(f"   Operation analyzed: {operation_analysis['operation_type']}")
        print(f"   Kernel generated: {len(kernel)} characters")
        print(f"   Optimization opportunities: {operation_analysis['optimization_opportunities']}")
        print(f"   Memory analysis: {operation_analysis['memory_analysis']}")
        print(f"   Parallelization strategy: {operation_analysis['parallelization_strategy']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

def main():
    """Run all enhanced CUDAgent tests (CPU version)."""
    print("🚀 Enhanced CUDAgent PyTorch to CUDA Translator Tests (CPU Version)")
    print("="*60)
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running in CPU-only mode (CUDA not available)")
    
    try:
        # Run tests
        test_operation_capture_demo()
        test_enhanced_parser_analysis()
        test_enhanced_kernel_generation()
        test_conv2d_analysis()
        test_enhanced_optimizer_analysis()
        
        print("\n" + "="*60)
        print("🎉 All Enhanced CUDAgent tests completed!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Real PyTorch operation capture with full context")
        print("✅ Enhanced operation parsing with real parameters")
        print("✅ Dynamic CUDA kernel generation based on actual operation data")
        print("✅ Block size optimization based on real tensor dimensions")
        print("✅ Memory layout analysis and optimization strategies")
        print("✅ Operation-specific parameter extraction (conv2d, matmul, etc.)")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 