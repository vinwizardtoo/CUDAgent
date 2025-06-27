#!/usr/bin/env python3
"""
Test script for AI Agent Framework

This script demonstrates:
- API key management and provider detection
- LLM routing and fallback mechanisms
- Kernel optimization with different providers
- Configuration management
"""

import os
import sys
import logging
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cudagent.agents import (
    ConfigManager, 
    LLMOptimizationAgent, 
    OptimizationRequest,
    KernelOptimizationAgent,
    PerformanceAdvisorAgent,
    KernelValidationAgent
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_manager():
    """Test the configuration manager and API key detection."""
    print("\n" + "="*60)
    print("TEST 1: Configuration Manager")
    print("="*60)
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Print configuration summary
    config_manager.print_configuration_summary()
    
    # Show setup guide
    print("\nSetup Guide:\n")
    
    # Test provider availability
    available_providers = config_manager.get_available_providers()
    best_provider = config_manager.get_best_provider()
    
    print(f"\nAvailable Providers: {available_providers}")
    print(f"Best Provider: {best_provider}")
    
    return config_manager

def test_llm_agent_with_mock():
    """Test LLM agent with mock responses."""
    print("\n" + "="*60)
    print("TEST 2: LLM Agent with Mock Provider")
    print("="*60)
    
    # Initialize LLM agent
    llm_agent = LLMOptimizationAgent()
    
    # Print configuration summary
    llm_agent.print_configuration_summary()
    
    # Create test operation info
    operation_info = {
        "operation_type": "matmul",
        "tensor_info": {
            "input_tensors": [
                {"shape": (512, 1024), "dtype": "float32", "device": "cuda"},
                {"shape": (1024, 256), "dtype": "float32", "device": "cuda"}
            ],
            "output_tensor": {"shape": (512, 256), "dtype": "float32", "device": "cuda"}
        },
        "operation_info": {
            "compute_intensity": 268435456,
            "memory_access": 917504,
            "arithmetic_intensity": 292.57
        }
    }
    
    # Create optimization request
    request = OptimizationRequest(
        operation_type="matmul",
        operation_info=operation_info,
        optimization_goals=["maximize performance", "minimize memory usage"],
        constraints={"max_shared_memory": 49152}
    )
    
    # Test optimization
    print("\nTesting kernel optimization...")
    response = llm_agent.optimize_kernel(request)
    
    # Display results
    print(f"\n✅ Optimization completed!")
    print(f"   Provider used: {response.provider_used}")
    print(f"   Generation time: {response.generation_time:.2f}s")
    print(f"   Confidence score: {response.confidence_score:.2f}")
    print(f"   Kernel length: {len(response.optimized_kernel)} characters")
    
    if response.optimization_explanations:
        print(f"\nOptimization explanations:")
        for i, explanation in enumerate(response.optimization_explanations, 1):
            print(f"   {i}. {explanation}")
    
    if response.performance_predictions:
        print(f"\nPerformance predictions:")
        for key, value in response.performance_predictions.items():
            print(f"   {key}: {value}")
    
    if response.warnings:
        print(f"\nWarnings:")
        for warning in response.warnings:
            print(f"   ⚠️  {warning}")
    
    if response.errors:
        print(f"\nErrors:")
        for error in response.errors:
            print(f"   ❌ {error}")
    
    return llm_agent

def test_kernel_optimization_agent():
    """Test the kernel optimization agent."""
    print("\n" + "="*60)
    print("TEST 3: Kernel Optimization Agent")
    print("="*60)
    
    # Initialize kernel optimization agent
    config_manager = ConfigManager()
    kernel_agent = KernelOptimizationAgent(config_manager)
    
    # Test operation info
    operation_info = {
        "operation_type": "matmul",
        "tensor_info": {
            "input_tensors": [
                {"shape": (512, 1024), "dtype": "float32", "device": "cuda"},
                {"shape": (1024, 256), "dtype": "float32", "device": "cuda"}
            ],
            "output_tensor": {"shape": (512, 256), "dtype": "float32", "device": "cuda"}
        }
    }
    
    # Test parameter optimization
    print("Testing kernel parameter optimization...")
    optimized_params = kernel_agent.optimize_kernel_parameters(operation_info)
    
    print(f"✅ Parameter optimization completed!")
    print(f"   Block size: {optimized_params.block_size}")
    print(f"   Grid size: {optimized_params.grid_size}")
    print(f"   Shared memory size: {optimized_params.shared_memory_size} bytes")
    print(f"   Occupancy: {optimized_params.occupancy:.2f}")
    
    # Test optimization strategies
    print("\nTesting optimization strategy suggestions...")
    strategies = kernel_agent.suggest_optimization_strategies(operation_info)
    
    print(f"✅ Found {len(strategies)} optimization strategies:")
    for i, strategy in enumerate(strategies, 1):
        print(f"   {i}. {strategy.name} (Priority: {strategy.priority}, Impact: {strategy.expected_improvement:.1f}x)")
        print(f"      {strategy.description}")
    
    return kernel_agent

def test_performance_advisor():
    """Test the performance advisor agent."""
    print("\n" + "="*60)
    print("TEST 4: Performance Advisor Agent")
    print("="*60)
    
    # Initialize performance advisor
    config_manager = ConfigManager()
    advisor = PerformanceAdvisorAgent(config_manager)
    
    # Test operation info
    operation_info = {
        "operation_type": "matmul",
        "tensor_info": {
            "input_tensors": [
                {"shape": (512, 1024), "dtype": "float32", "device": "cuda"},
                {"shape": (1024, 256), "dtype": "float32", "device": "cuda"}
            ],
            "output_tensor": {"shape": (512, 256), "dtype": "float32", "device": "cuda"}
        }
    }
    
    # Test performance analysis
    print("Testing performance analysis...")
    analysis = advisor.analyze_performance(operation_info)
    
    print(f"✅ Performance analysis completed!")
    print(f"   Operation type: {analysis.operation_type}")
    print(f"   Optimization potential: {analysis.optimization_potential:.2f}x")
    print(f"   Bottlenecks found: {len(analysis.bottlenecks)}")
    print(f"   Recommendations: {len(analysis.recommendations)}")
    
    if analysis.bottlenecks:
        print(f"\nBottlenecks:")
        for bottleneck in analysis.bottlenecks:
            print(f"   🔴 {bottleneck}")
    
    if analysis.recommendations:
        print(f"\nTop recommendations:")
        for i, rec in enumerate(analysis.recommendations[:3], 1):
            print(f"   {i}. {rec.title} (Impact: {rec.impact}, Effort: {rec.effort})")
    
    return advisor

def test_validation_agent():
    """Test the validation agent."""
    print("\n" + "="*60)
    print("TEST 5: Validation Agent")
    print("="*60)
    
    # Initialize validation agent
    validator = KernelValidationAgent()
    
    # Test kernel code
    test_kernel = """
__global__ void test_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
"""
    
    operation_info = {
        "operation_type": "add",
        "tensor_info": {
            "input_tensors": [
                {"shape": (1000,), "dtype": "float32", "device": "cuda"},
                {"shape": (1000,), "dtype": "float32", "device": "cuda"}
            ],
            "output_tensor": {"shape": (1000,), "dtype": "float32", "device": "cuda"}
        }
    }
    
    # Test validation
    print("Testing kernel validation...")
    report = validator.validate_kernel(test_kernel, operation_info)
    
    print(f"✅ Validation completed!")
    print(f"   Overall result: {report.overall_result.value}")
    print(f"   Is compilable: {report.is_compilable}")
    print(f"   Is optimized: {report.is_optimized}")
    print(f"   Issues found: {len(report.issues)}")
    print(f"   Validation time: {report.validation_time:.2f}s")
    
    if report.issues:
        print(f"\nIssues found:")
        for issue in report.issues:
            status_icon = "❌" if issue.level.value in ["error", "critical"] else "⚠️"
            print(f"   {status_icon} {issue.category}: {issue.message}")
            if issue.suggestion:
                print(f"      Suggestion: {issue.suggestion}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"   💡 {rec}")
    
    return validator

def test_integration():
    """Test integration of all agents."""
    print("\n" + "="*60)
    print("TEST 6: Agent Integration")
    print("="*60)
    
    # Initialize all agents
    config_manager = ConfigManager()
    llm_agent = LLMOptimizationAgent(config_manager)
    kernel_agent = KernelOptimizationAgent(config_manager)
    advisor = PerformanceAdvisorAgent(config_manager)
    validator = KernelValidationAgent()
    
    # Test operation
    operation_info = {
        "operation_type": "matmul",
        "tensor_info": {
            "input_tensors": [
                {"shape": (256, 512), "dtype": "float32", "device": "cuda"},
                {"shape": (512, 128), "dtype": "float32", "device": "cuda"}
            ],
            "output_tensor": {"shape": (256, 128), "dtype": "float32", "device": "cuda"}
        }
    }
    
    print("Testing integrated agent workflow...")
    
    # 1. Performance analysis
    print("1. Analyzing performance...")
    analysis = advisor.analyze_performance(operation_info)
    
    # 2. Kernel parameter optimization
    print("2. Optimizing kernel parameters...")
    params = kernel_agent.optimize_kernel_parameters(operation_info)
    
    # 3. LLM optimization
    print("3. Generating optimized kernel...")
    request = OptimizationRequest(
        operation_type="matmul",
        operation_info=operation_info,
        optimization_goals=["maximize performance"]
    )
    response = llm_agent.optimize_kernel(request)
    
    # 4. Validation
    print("4. Validating generated kernel...")
    if response.optimized_kernel:
        validation_report = validator.validate_kernel(
            response.optimized_kernel, operation_info
        )
    else:
        validation_report = None
    
    # Display integrated results
    print(f"\n✅ Integrated workflow completed!")
    print(f"   Performance analysis: {analysis.optimization_potential:.2f}x potential")
    print(f"   Kernel parameters: {params.block_size} blocks, {params.occupancy:.2f} occupancy")
    print(f"   LLM optimization: {response.provider_used} provider, {response.confidence_score:.2f} confidence")
    if validation_report:
        print(f"   Validation: {validation_report.overall_result.value} result, {len(validation_report.issues)} issues")
    
    return {
        'config_manager': config_manager,
        'llm_agent': llm_agent,
        'kernel_agent': kernel_agent,
        'advisor': advisor,
        'validator': validator
    }

def main():
    """Run all AI agent tests."""
    print("🚀 AI Agent Framework Tests")
    print("="*60)
    
    try:
        # Test configuration management
        config_manager = test_configuration_manager()
        
        # Test LLM agent
        llm_agent = test_llm_agent_with_mock()
        
        # Test kernel optimization agent
        kernel_agent = test_kernel_optimization_agent()
        
        # Test performance advisor
        advisor = test_performance_advisor()
        
        # Test validation agent
        validator = test_validation_agent()
        
        # Test integration
        agents = test_integration()
        
        print("\n" + "="*60)
        print("🎉 All AI Agent Framework tests completed!")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("✅ API key management and provider detection")
        print("✅ Automatic LLM routing and fallback")
        print("✅ Kernel optimization with parameter tuning")
        print("✅ Performance analysis and recommendations")
        print("✅ Kernel validation and error detection")
        print("✅ Integrated agent workflow")
        
        print("\nNext Steps:")
        print("1. Set up API keys for real LLM providers")
        print("2. Test with actual CUDA operations")
        print("3. Integrate with the enhanced optimizer")
        print("4. Implement feedback loop for performance improvement")
        
        # Show generated files summary
        print("\n📁 Generated Files Summary:")
        import os
        from pathlib import Path
        
        # Check kernel files
        kernel_dir = Path("generated_kernels")
        if kernel_dir.exists():
            kernel_files = list(kernel_dir.glob("*.cu"))
            print(f"   Generated kernels: {len(kernel_files)} files in generated_kernels/")
            for file in kernel_files[-3:]:  # Show last 3 files
                print(f"     - {file.name}")
        else:
            print("   Generated kernels: No files found")
        
        # Check log files
        log_dir = Path("logs/optimizations")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.json"))
            print(f"   Optimization logs: {len(log_files)} files in logs/optimizations/")
            for file in log_files[-3:]:  # Show last 3 files
                print(f"     - {file.name}")
        else:
            print("   Optimization logs: No files found")
        
        print("\n💡 Note: All generated files are automatically saved and ignored by git")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 