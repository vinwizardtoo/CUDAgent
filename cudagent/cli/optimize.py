#!/usr/bin/env python3
"""
CUDAgent CLI - Kernel Optimization Command
Provides a command-line interface for optimizing PyTorch operations to CUDA kernels.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

def main():
    """Main entry point for the cudagent-optimize command."""
    parser = argparse.ArgumentParser(
        description="Optimize PyTorch operations to CUDA kernels using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize a simple matrix multiplication
  cudagent-optimize --operation matmul --input-shape "1024,1024" --output-shape "1024,1024"
  
  # Optimize with specific optimization level
  cudagent-optimize --operation conv2d --input-shape "32,3,224,224" --optimization-level aggressive
  
  # Optimize from a Python file
  cudagent-optimize --file my_model.py --target-operation "forward"
  
  # Optimize with custom CUDA architecture
  cudagent-optimize --operation matmul --cuda-arch "sm_80" --optimization-level balanced
        """
    )
    
    # Operation specification
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--operation", "-o",
        type=str,
        help="PyTorch operation to optimize (e.g., matmul, conv2d, relu)"
    )
    operation_group.add_argument(
        "--file", "-f",
        type=str,
        help="Python file containing PyTorch operations to optimize"
    )
    
    # Input/Output specifications
    parser.add_argument(
        "--input-shape", "-i",
        type=str,
        help="Input tensor shape (e.g., '1024,1024' or 'batch,channels,height,width')"
    )
    parser.add_argument(
        "--output-shape", "-out",
        type=str,
        help="Output tensor shape (e.g., '1024,1024')"
    )
    parser.add_argument(
        "--target-operation",
        type=str,
        help="Specific operation name to target when using --file"
    )
    
    # Optimization settings
    parser.add_argument(
        "--optimization-level",
        choices=["basic", "balanced", "aggressive"],
        default="balanced",
        help="Optimization aggressiveness level"
    )
    parser.add_argument(
        "--cuda-arch",
        type=str,
        default="sm_70",
        help="Target CUDA architecture (e.g., sm_70, sm_80, sm_86)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM-based optimization (requires API keys)"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "anthropic", "local"],
        default="openai",
        help="LLM provider to use for optimization"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-d",
        type=str,
        default="./generated_kernels",
        help="Directory to save generated kernels"
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        help="Custom name for the generated kernel"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks on generated kernels"
    )
    
    args = parser.parse_args()
    
    try:
        # Import here to avoid circular imports
        from cudagent import CUDAgent
        
        # Initialize CUDAgent
        agent = CUDAgent(
            optimization_level=args.optimization_level,
            cuda_arch=args.cuda_arch,
            use_llm=args.use_llm,
            llm_provider=args.llm_provider,
            verbose=args.verbose
        )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.file:
            # Optimize from file
            if not os.path.exists(args.file):
                print(f"Error: File '{args.file}' not found.")
                sys.exit(1)
            
            print(f"Analyzing file: {args.file}")
            result = agent.optimize_from_file(
                file_path=args.file,
                target_operation=args.target_operation,
                output_dir=str(output_dir)
            )
        else:
            # Optimize specific operation
            if not args.input_shape:
                print("Error: --input-shape is required when using --operation")
                sys.exit(1)
            
            print(f"Optimizing operation: {args.operation}")
            result = agent.optimize_operation(
                operation_type=args.operation,
                input_shape=args.input_shape,
                output_shape=args.output_shape,
                kernel_name=args.kernel_name,
                output_dir=str(output_dir)
            )
        
        # Display results
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        
        if result.get('kernel_path'):
            print(f"Generated kernel: {result['kernel_path']}")
        
        if result.get('optimization_summary'):
            print(f"\nOptimization Summary:")
            for key, value in result['optimization_summary'].items():
                print(f"  {key}: {value}")
        
        if result.get('performance_metrics'):
            print(f"\nPerformance Metrics:")
            for metric, value in result['performance_metrics'].items():
                print(f"  {metric}: {value}")
        
        if args.benchmark and result.get('benchmark_results'):
            print(f"\nBenchmark Results:")
            for benchmark, result_data in result['benchmark_results'].items():
                print(f"  {benchmark}: {result_data}")
        
        print(f"\nKernel saved to: {output_dir}")
        print("="*50)
        
    except ImportError as e:
        print(f"Error: Could not import CUDAgent. Make sure it's properly installed.")
        print(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during optimization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 