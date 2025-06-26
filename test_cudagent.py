#!/usr/bin/env python3
"""
Simple test script for CUDAgent to verify the implementation works.
"""

import torch
import logging
from cudagent import CUDAgentOptimizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_basic_optimization():
    """Test basic optimization functionality."""
    print("🧪 Testing CUDAgent Basic Functionality")
    print("=" * 50)
    
    # Create optimizer
    optimizer = CUDAgentOptimizer()
    
    # Test with a simple matrix multiplication
    print("Creating test tensor...")
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    z = torch.matmul(x, y)
    
    print(f"Tensor shape: {z.shape}")
    print(f"Tensor device: {z.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test optimization
    print("\n🔧 Starting optimization...")
    try:
        result = optimizer.optimize_operation(z)
        
        if result["success"]:
            print("✅ Optimization completed successfully!")
            print(f"Speedup: {result['speedup']:.2f}x")
            print(f"Optimization time: {result['optimization_time']:.2f}s")
            print(f"Iterations: {result['iterations']}")
            
            # Show some details
            print(f"\nOperation type: {result['operation_info']['operation_type']}")
            print(f"Complexity: {result['operation_info']['complexity']}")
            print(f"Supported: {result['operation_info']['is_supported']}")
            
        else:
            print("❌ Optimization failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
    
    print("\n" + "=" * 50)

def test_supported_operations():
    """Test getting supported operations."""
    print("📋 Testing Supported Operations")
    print("=" * 50)
    
    optimizer = CUDAgentOptimizer()
    supported_ops = optimizer.get_supported_operations()
    
    print(f"Supported operations ({len(supported_ops)}):")
    for op in supported_ops:
        print(f"  - {op}")
    
    print("\n" + "=" * 50)

def test_optimization_stats():
    """Test optimization statistics."""
    print("📊 Testing Optimization Statistics")
    print("=" * 50)
    
    optimizer = CUDAgentOptimizer()
    stats = optimizer.get_optimization_stats()
    
    print("Optimization statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    print("🚀 CUDAgent Test Suite")
    print("=" * 50)
    
    # Run tests
    test_supported_operations()
    test_optimization_stats()
    test_basic_optimization()
    
    print("🎉 Test suite completed!") 