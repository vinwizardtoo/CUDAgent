#!/usr/bin/env python3
"""
CUDAgent CLI - Benchmark Command
Provides a command-line interface for benchmarking CUDA kernels and comparing performance.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List

def main():
    """Main entry point for the cudagent-benchmark command."""
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA kernels and compare performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark a single kernel
  cudagent-benchmark --kernel my_kernel.cu --input-shape "1024,1024"
  
  # Compare multiple kernels
  cudagent-benchmark --kernels kernel1.cu kernel2.cu --input-shape "1024,1024"
  
  # Benchmark with multiple input sizes
  cudagent-benchmark --kernel my_kernel.cu --input-shapes "512,512" "1024,1024" "2048,2048"
  
  # Benchmark against PyTorch baseline
  cudagent-benchmark --kernel my_kernel.cu --compare-pytorch --operation matmul
        """
    )
    
    # Kernel specification
    kernel_group = parser.add_mutually_exclusive_group(required=True)
    kernel_group.add_argument(
        "--kernel", "-k",
        type=str,
        help="Single CUDA kernel file to benchmark"
    )
    kernel_group.add_argument(
        "--kernels",
        nargs="+",
        help="Multiple CUDA kernel files to benchmark and compare"
    )
    
    # Input specifications
    parser.add_argument(
        "--input-shape", "-i",
        type=str,
        help="Input tensor shape (e.g., '1024,1024')"
    )
    parser.add_argument(
        "--input-shapes",
        nargs="+",
        help="Multiple input tensor shapes to test"
    )
    parser.add_argument(
        "--output-shape",
        type=str,
        help="Output tensor shape (e.g., '1024,1024')"
    )
    
    # Benchmark settings
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations for benchmarking"
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32", "fp64"],
        default="fp32",
        help="Precision for benchmarking"
    )
    parser.add_argument(
        "--cuda-arch",
        type=str,
        default="sm_70",
        help="Target CUDA architecture"
    )
    
    # Comparison options
    parser.add_argument(
        "--compare-pytorch",
        action="store_true",
        help="Compare against PyTorch baseline"
    )
    parser.add_argument(
        "--operation",
        type=str,
        help="PyTorch operation for baseline comparison"
    )
    parser.add_argument(
        "--baseline-kernel",
        type=str,
        help="Baseline kernel for comparison"
    )
    
    # Output settings
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save benchmark results to file"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format for results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate performance plots"
    )
    
    args = parser.parse_args()
    
    try:
        # Import here to avoid circular imports
        from cudagent import CUDAgent
        from cudagent.profiling.benchmarker import PerformanceBenchmarker
        
        # Initialize CUDAgent
        agent = CUDAgent(verbose=args.verbose)
        
        # Initialize benchmarker
        benchmarker = PerformanceBenchmarker(
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            precision=args.precision,
            cuda_arch=args.cuda_arch
        )
        
        # Determine input shapes
        input_shapes = []
        if args.input_shapes:
            input_shapes = args.input_shapes
        elif args.input_shape:
            input_shapes = [args.input_shape]
        else:
            print("Error: Either --input-shape or --input-shapes is required")
            sys.exit(1)
        
        # Determine kernels to benchmark
        kernels = []
        if args.kernels:
            kernels = args.kernels
        elif args.kernel:
            kernels = [args.kernel]
        
        # Validate kernel files
        for kernel in kernels:
            if not os.path.exists(kernel):
                print(f"Error: Kernel file '{kernel}' not found.")
                sys.exit(1)
        
        print("Starting benchmark...")
        print(f"Kernels: {kernels}")
        print(f"Input shapes: {input_shapes}")
        print(f"Iterations: {args.iterations}")
        print(f"Precision: {args.precision}")
        print("-" * 50)
        
        # Run benchmarks
        results = {}
        
        for kernel in kernels:
            kernel_name = Path(kernel).stem
            results[kernel_name] = {}
            
            for shape in input_shapes:
                print(f"Benchmarking {kernel_name} with shape {shape}...")
                
                try:
                    benchmark_result = benchmarker.benchmark_kernel(
                        kernel_path=kernel,
                        input_shape=shape,
                        output_shape=args.output_shape
                    )
                    results[kernel_name][shape] = benchmark_result
                    
                    if args.verbose:
                        print(f"  Average time: {benchmark_result.get('average_time', 'N/A')} ms")
                        print(f"  Throughput: {benchmark_result.get('throughput', 'N/A')} ops/sec")
                        
                except Exception as e:
                    print(f"  Error benchmarking {kernel_name} with shape {shape}: {e}")
                    results[kernel_name][shape] = {"error": str(e)}
        
        # Compare with PyTorch baseline if requested
        if args.compare_pytorch and args.operation:
            print("\nComparing with PyTorch baseline...")
            try:
                pytorch_results = benchmarker.benchmark_pytorch_operation(
                    operation=args.operation,
                    input_shapes=input_shapes,
                    output_shape=args.output_shape
                )
                results["PyTorch_Baseline"] = pytorch_results
            except Exception as e:
                print(f"Error benchmarking PyTorch baseline: {e}")
        
        # Display results
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        if args.format == "text":
            display_text_results(results, input_shapes)
        elif args.format == "json":
            import json
            print(json.dumps(results, indent=2))
        elif args.format == "csv":
            display_csv_results(results, input_shapes)
        
        # Save results to file if requested
        if args.output_file:
            save_results(results, args.output_file, args.format)
            print(f"\nResults saved to: {args.output_file}")
        
        # Generate plots if requested
        if args.plot:
            try:
                generate_performance_plots(results, input_shapes, args.output_file)
                print("Performance plots generated.")
            except Exception as e:
                print(f"Error generating plots: {e}")
        
        print("="*60)
        
    except ImportError as e:
        print(f"Error: Could not import CUDAgent. Make sure it's properly installed.")
        print(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def display_text_results(results, input_shapes):
    """Display benchmark results in text format."""
    for kernel_name, kernel_results in results.items():
        print(f"\n{kernel_name}:")
        print("-" * 40)
        
        for shape in input_shapes:
            if shape in kernel_results:
                result = kernel_results[shape]
                if "error" in result:
                    print(f"  {shape}: ERROR - {result['error']}")
                else:
                    avg_time = result.get('average_time', 'N/A')
                    throughput = result.get('throughput', 'N/A')
                    print(f"  {shape}: {avg_time} ms, {throughput} ops/sec")

def display_csv_results(results, input_shapes):
    """Display benchmark results in CSV format."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    header = ["Kernel"] + input_shapes
    writer.writerow(header)
    
    # Data
    for kernel_name, kernel_results in results.items():
        row = [kernel_name]
        for shape in input_shapes:
            if shape in kernel_results:
                result = kernel_results[shape]
                if "error" in result:
                    row.append(f"ERROR: {result['error']}")
                else:
                    avg_time = result.get('average_time', 'N/A')
                    row.append(str(avg_time))
            else:
                row.append("N/A")
        writer.writerow(row)
    
    print(output.getvalue())

def save_results(results, output_file, format_type):
    """Save benchmark results to file."""
    if format_type == "json":
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif format_type == "csv":
        import csv
        with open(output_file, 'w', newline='') as f:
            # Implementation would be similar to display_csv_results
            pass

def generate_performance_plots(results, input_shapes, output_file):
    """Generate performance plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # This is a placeholder - actual implementation would create meaningful plots
        print("Plot generation not yet implemented")
        
    except ImportError:
        print("matplotlib not available for plotting")

if __name__ == "__main__":
    main() 