"""
Performance benchmarker for measuring execution time and performance metrics.
"""

import time
import logging
import statistics
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import psutil

logger = logging.getLogger(__name__)


class PerformanceBenchmarker:
    """
    Benchmarker for measuring performance of PyTorch operations and CUDA kernels.
    """
    
    def __init__(self, warmup_iterations: int = 10, benchmark_iterations: int = 100):
        """
        Initialize the performance benchmarker.
        
        Args:
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device = torch.device("cuda")
            logger.info("CUDA is available for benchmarking")
        else:
            self.device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU for benchmarking")
    
    def benchmark_operation(self, operation: torch.Tensor) -> Dict[str, Any]:
        """
        Benchmark a PyTorch operation.
        
        Args:
            operation: PyTorch tensor operation to benchmark
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Ensure operation is on the correct device
            if self.cuda_available and operation.device.type != "cuda":
                operation = operation.to(self.device)
            
            # Warmup
            self._warmup_operation(operation)
            
            # Benchmark
            execution_times = []
            memory_usage = []
            
            for _ in range(self.benchmark_iterations):
                # Record memory before
                if self.cuda_available:
                    torch.cuda.synchronize()
                    memory_before = torch.cuda.memory_allocated()
                else:
                    memory_before = psutil.Process().memory_info().rss
                
                # Time the operation
                start_time = time.perf_counter()
                result = self._execute_operation(operation)
                if self.cuda_available:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                # Record memory after
                if self.cuda_available:
                    memory_after = torch.cuda.memory_allocated()
                else:
                    memory_after = psutil.Process().memory_info().rss
                
                execution_times.append(end_time - start_time)
                memory_usage.append(memory_after - memory_before)
            
            # Calculate statistics
            metrics = self._calculate_metrics(execution_times, memory_usage, operation)
            
            logger.info(f"Benchmarked operation: {metrics['execution_time']:.6f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {str(e)}")
            return self._create_error_metrics(str(e))
    
    def benchmark_kernel(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmark a CUDA kernel (simulated for now).
        
        Args:
            kernel_code: CUDA kernel code
            operation_info: Information about the operation
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # For now, we'll simulate kernel execution
            # In a real implementation, you'd compile and run the actual CUDA kernel
            
            # Simulate kernel execution time based on operation complexity
            base_time = self._estimate_kernel_time(operation_info)
            
            # Add some randomness to simulate real execution
            execution_times = []
            for _ in range(self.benchmark_iterations):
                # Simulate kernel execution time
                simulated_time = base_time * np.random.uniform(0.8, 1.2)
                execution_times.append(simulated_time)
            
            # Simulate memory usage
            memory_usage = [operation_info["tensor_info"]["numel"] * 4] * self.benchmark_iterations
            
            # Calculate metrics
            metrics = self._calculate_metrics(execution_times, memory_usage, None)
            metrics["kernel_code"] = kernel_code
            
            logger.info(f"Benchmarked kernel: {metrics['execution_time']:.6f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Kernel benchmarking failed: {str(e)}")
            return self._create_error_metrics(str(e))
    
    def _warmup_operation(self, operation: torch.Tensor):
        """Perform warmup iterations to stabilize performance."""
        for _ in range(self.warmup_iterations):
            self._execute_operation(operation)
            if self.cuda_available:
                torch.cuda.synchronize()
    
    def _execute_operation(self, operation: torch.Tensor) -> torch.Tensor:
        """
        Execute a PyTorch operation.
        This is a simplified implementation - in practice, you'd need to
        handle different types of operations properly.
        """
        # For now, we'll just return the operation as-is
        # In a real implementation, you'd execute the actual operation
        return operation
    
    def _estimate_kernel_time(self, operation_info: Dict[str, Any]) -> float:
        """
        Estimate kernel execution time based on operation complexity.
        This is a simplified model - real estimation would be much more complex.
        """
        tensor_info = operation_info["tensor_info"]
        operation_type = operation_info["operation_type"]
        
        # Base time per element (in seconds)
        base_time_per_element = 1e-9  # 1 nanosecond per element
        
        # Complexity multipliers
        complexity_multipliers = {
            "matmul": 10.0,
            "conv2d": 5.0,
            "layer_norm": 3.0,
            "batch_norm": 2.0,
            "softmax": 4.0,
            "generic": 1.0,
        }
        
        multiplier = complexity_multipliers.get(operation_type, 1.0)
        numel = tensor_info["numel"]
        
        estimated_time = base_time_per_element * numel * multiplier
        
        # Add some overhead
        overhead = 1e-6  # 1 microsecond overhead
        return estimated_time + overhead
    
    def _calculate_metrics(self, execution_times: List[float], 
                          memory_usage: List[float], 
                          operation: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Calculate performance metrics from execution times and memory usage."""
        if not execution_times:
            return self._create_error_metrics("No execution times recorded")
        
        # Execution time statistics
        execution_time_mean = statistics.mean(execution_times)
        execution_time_std = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        execution_time_min = min(execution_times)
        execution_time_max = max(execution_times)
        
        # Memory usage statistics
        memory_mean = statistics.mean(memory_usage) if memory_usage else 0.0
        memory_max = max(memory_usage) if memory_usage else 0.0
        
        # Calculate throughput (operations per second)
        throughput = 1.0 / execution_time_mean if execution_time_mean > 0 else 0.0
        
        # Calculate efficiency metrics
        if operation is not None:
            tensor_info = {
                "shape": list(operation.shape),
                "numel": operation.numel(),
                "dtype": str(operation.dtype),
            }
        else:
            tensor_info = {}
        
        return {
            "execution_time": execution_time_mean,
            "execution_time_std": execution_time_std,
            "execution_time_min": execution_time_min,
            "execution_time_max": execution_time_max,
            "memory_usage": memory_mean,
            "memory_peak": memory_max,
            "throughput": throughput,
            "iterations": len(execution_times),
            "tensor_info": tensor_info,
            "device": str(self.device),
            "success": True,
        }
    
    def _create_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Create error metrics when benchmarking fails."""
        return {
            "execution_time": float('inf'),
            "execution_time_std": 0.0,
            "execution_time_min": float('inf'),
            "execution_time_max": 0.0,
            "memory_usage": 0.0,
            "memory_peak": 0.0,
            "throughput": 0.0,
            "iterations": 0,
            "error": error_message,
            "success": False,
        }
    
    def compare_performance(self, original_metrics: Dict[str, Any], 
                          optimized_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance between original and optimized implementations.
        
        Args:
            original_metrics: Performance metrics of original implementation
            optimized_metrics: Performance metrics of optimized implementation
            
        Returns:
            Dictionary containing comparison results
        """
        if not original_metrics.get("success", False) or not optimized_metrics.get("success", False):
            return {
                "speedup": 0.0,
                "memory_improvement": 0.0,
                "throughput_improvement": 0.0,
                "comparison_valid": False,
                "error": "One or both metrics are invalid",
            }
        
        # Calculate speedup
        original_time = original_metrics["execution_time"]
        optimized_time = optimized_metrics["execution_time"]
        speedup = original_time / optimized_time if optimized_time > 0 else 0.0
        
        # Calculate memory improvement
        original_memory = original_metrics["memory_usage"]
        optimized_memory = optimized_metrics["memory_usage"]
        memory_improvement = (original_memory - optimized_memory) / original_memory if original_memory > 0 else 0.0
        
        # Calculate throughput improvement
        original_throughput = original_metrics["throughput"]
        optimized_throughput = optimized_metrics["throughput"]
        throughput_improvement = (optimized_throughput - original_throughput) / original_throughput if original_throughput > 0 else 0.0
        
        return {
            "speedup": speedup,
            "memory_improvement": memory_improvement,
            "throughput_improvement": throughput_improvement,
            "comparison_valid": True,
            "original_metrics": original_metrics,
            "optimized_metrics": optimized_metrics,
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            "device": str(self.device),
            "cuda_available": self.cuda_available,
        }
        
        if self.cuda_available:
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(),
                "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory,
            })
        
        info.update({
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
        })
        
        return info 