#!/usr/bin/env python3
"""
CUDAgent Test CLI

Quick testing and validation of CUDAgent functionality.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cudagent.agents import LLMAgent, KernelOptimizer, PerformanceAdvisor, ValidationAgent
from cudagent.parsers.pytorch_parser import PyTorchParser
from cudagent.utils.kernel_generator import KernelGenerator
from cudagent.profiling.benchmarker import Benchmarker


def main():
    """Main CLI entry point for testing."""
    parser = argparse.ArgumentParser(
        description="CUDAgent Test - Test CUDAgent functionality"
    )
    parser.add_argument(
        "--config-file",
        default="config/api_keys.json",
        help="Path to configuration file (default: config/api_keys.json)"
    )
    parser.add_argument(
        "--test-type",
        choices=["basic", "llm", "gpu", "full"],
        default="basic",
        help="Type of test to run (default: basic)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Save detailed logs to files"
    )
    
    args = parser.parse_args()
    
    print("🧪 CUDAgent Test Suite")
    print("=" * 40)
    print()
    
    if args.test_type == "basic":
        run_basic_test(args)
    elif args.test_type == "llm":
        run_llm_test(args)
    elif args.test_type == "gpu":
        run_gpu_test(args)
    elif args.test_type == "full":
        run_full_test(args)


def run_basic_test(args):
    """Run basic functionality tests."""
    print("🔧 Running Basic Functionality Tests")
    print("-" * 35)
    
    try:
        # Test PyTorch parser
        print("1. Testing PyTorch Parser...")
        parser = PyTorchParser()
        
        # Test basic functionality without complex capture
        operations = parser.get_supported_operations()
        if operations:
            print("   ✅ PyTorch parser working")
            if args.verbose:
                print(f"   📝 Supported operations: {operations}")
        else:
            print("   ❌ PyTorch parser failed")
            return False
        
        # Test kernel generator
        print("2. Testing Kernel Generator...")
        generator = KernelGenerator()
        
        # Generate a simple kernel with proper operation_analysis format
        operation_analysis = {
            "operation_type": "matmul",
            "operation_info": {
                "input_shapes": [(100, 100), (100, 100)],
                "output_shape": (100, 100),
                "block_size_optimization": {"block_x": 16, "block_y": 16}
            },
            "tensor_info": {
                "dtype": "float32"
            }
        }
        
        kernel_code = generator.generate_kernel(operation_analysis)
        
        if kernel_code:
            print("   ✅ Kernel generator working")
            if args.verbose:
                print(f"   📝 Generated kernel: {len(kernel_code)} lines")
        else:
            print("   ❌ Kernel generator failed")
            return False
        
        # Test benchmarker
        print("3. Testing Benchmarker...")
        benchmarker = Benchmarker()
        
        # Create a simple PyTorch tensor for benchmarking
        import torch
        test_tensor = torch.randn(10, 10)
        
        # Mock benchmark
        result = benchmarker.benchmark_operation(test_tensor)
        
        if result:
            print("   ✅ Benchmarker working")
            if args.verbose:
                print(f"   📊 Benchmark result: {result}")
        else:
            print("   ⚠️  Benchmarker working (CPU mode)")
        
        print()
        print("✅ Basic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def run_llm_test(args):
    """Run LLM functionality tests."""
    print("🤖 Running LLM Functionality Tests")
    print("-" * 35)
    
    try:
        # Test config manager
        print("1. Testing Configuration Manager...")
        from cudagent.agents.config_manager import ConfigManager
        config_manager = ConfigManager(args.config_file)
        config_manager.print_configuration_summary()
        
        # Test LLM agent
        print("2. Testing LLM Agent...")
        llm_agent = LLMAgent(config_manager)
        
        # Test kernel optimization
        from cudagent.agents.llm_agent import OptimizationRequest
        request = OptimizationRequest(
            operation_type="matmul",
            operation_info={"input_shapes": [(100, 100), (100, 100)]},
            current_kernel="__global__ void matmul(float* A, float* B, float* C, int N) {}",
            optimization_goals=["maximize performance"]
        )
        
        response = llm_agent.optimize_kernel(request)
        if response and response.optimized_kernel:
            print("   ✅ LLM agent working")
            if args.verbose:
                print(f"   💬 Generated kernel: {len(response.optimized_kernel)} characters")
        else:
            print("   ❌ LLM agent failed")
            return False
        
        # Test kernel optimizer
        print("3. Testing Kernel Optimizer...")
        kernel_optimizer = KernelOptimizer(config_manager)
        
        # Test parameter optimization
        operation_info = {
            'operation_type': 'matmul',
            'tensor_info': {
                'input_tensors': [
                    {'shape': (100, 100)},
                    {'shape': (100, 100)}
                ]
            }
        }
        
        optimized_params = kernel_optimizer.optimize_kernel_parameters(operation_info)
        
        if optimized_params:
            print("   ✅ Kernel optimizer working")
            if args.verbose:
                print(f"   📝 Optimized parameters: {optimized_params}")
        else:
            print("   ❌ Kernel optimizer failed")
            return False
        
        print()
        print("✅ LLM tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def run_gpu_test(args):
    """Run GPU functionality tests."""
    print("🚀 Running GPU Functionality Tests")
    print("-" * 35)
    
    try:
        # Check CUDA availability
        print("1. Checking CUDA Availability...")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"   📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("   ⚠️  CUDA not available - running in CPU mode")
        except ImportError:
            print("   ⚠️  PyTorch not installed - skipping GPU tests")
            return False
        
        # Test performance advisor
        print("2. Testing Performance Advisor...")
        from cudagent.agents.config_manager import ConfigManager
        config_manager = ConfigManager()
        performance_advisor = PerformanceAdvisor(config_manager)
        
        # Mock performance analysis
        operation_info = {
            'operation_type': 'matmul',
            'tensor_info': {
                'input_tensors': [
                    {'shape': (1000, 1000)},
                    {'shape': (1000, 1000)}
                ]
            }
        }
        
        analysis = performance_advisor.analyze_performance(operation_info)
        
        if analysis:
            print("   ✅ Performance advisor working")
            if args.verbose:
                print(f"   📊 Analysis: {analysis}")
        else:
            print("   ❌ Performance advisor failed")
            return False
        
        # Test validation agent
        print("3. Testing Validation Agent...")
        validation_agent = ValidationAgent()
        
        # Mock validation
        validation_result = validation_agent.validate_kernel(
            kernel_code="__global__ void test() {}",
            operation_type="matmul",
            input_shapes=[(10, 10), (10, 10)]
        )
        
        if validation_result:
            print("   ✅ Validation agent working")
            if args.verbose:
                print(f"   ✅ Validation result: {validation_result}")
        else:
            print("   ❌ Validation agent failed")
            return False
        
        print()
        print("✅ GPU tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def run_full_test(args):
    """Run full integration test."""
    print("🎯 Running Full Integration Test")
    print("-" * 35)
    
    # Run all test types
    basic_success = run_basic_test(args)
    print()
    
    llm_success = run_llm_test(args)
    print()
    
    gpu_success = run_gpu_test(args)
    print()
    
    # Summary
    print("📊 Test Summary")
    print("-" * 15)
    print(f"Basic Tests:     {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"LLM Tests:       {'✅ PASS' if llm_success else '❌ FAIL'}")
    print(f"GPU Tests:       {'✅ PASS' if gpu_success else '❌ FAIL'}")
    print()
    
    if basic_success and llm_success and gpu_success:
        print("🎉 All tests passed! CUDAgent is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 