"""
Enhanced CUDAgent optimizer for converting captured PyTorch operations to optimized CUDA kernels.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
import numpy as np

from ..parsers.operation_capture import OperationCapture
from ..parsers.pytorch_parser import EnhancedPyTorchOperationParser
from ..profiling.benchmarker import PerformanceBenchmarker
from ..utils.enhanced_kernel_generator import EnhancedCUDAKernelGenerator
from ..utils.validation import KernelValidator

logger = logging.getLogger(__name__)


class EnhancedCUDAAgentOptimizer:
    """
    Enhanced optimizer class that orchestrates the conversion of captured PyTorch operations
    to optimized CUDA kernels using the new operation capture system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced CUDAAgent optimizer.
        
        Args:
            config: Configuration dictionary for optimization settings
        """
        self.config = config or self._get_default_config()
        self.operation_capture = OperationCapture()
        self.parser = EnhancedPyTorchOperationParser()
        self.benchmarker = PerformanceBenchmarker()
        self.kernel_generator = EnhancedCUDAKernelGenerator()
        self.validator = KernelValidator()
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            logger.warning("CUDA is not available. Some optimizations may be limited.")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the enhanced optimizer."""
        return {
            "max_iterations": 10,
            "population_size": 5,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "target_speedup": 2.0,
            "timeout_seconds": 300,
            "enable_fusion": True,
            "memory_optimization": True,
            "enable_operation_capture": True,
        }
    
    def capture_and_optimize_operation(self, operation_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Capture a PyTorch operation and optimize it into an optimized CUDA kernel.
        
        Args:
            operation_func: The PyTorch operation function (e.g., torch.matmul)
            *args: Positional arguments to the operation
            **kwargs: Keyword arguments to the operation
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        try:
            # Step 1: Capture the operation with full context
            logger.info("Capturing PyTorch operation...")
            result_tensor = self.operation_capture.capture_operation(operation_func, *args, **kwargs)
            
            # Step 2: Get the captured operation information
            operation_info = self.operation_capture.get_last_operation()
            if not operation_info:
                raise ValueError("Failed to capture operation information")
            
            # Step 3: Parse the captured operation
            logger.info("Parsing captured operation...")
            operation_analysis = self.parser.parse_captured_operation(operation_info)
            
            # Step 4: Generate initial CUDA kernel
            logger.info("Generating initial CUDA kernel...")
            initial_kernel = self.kernel_generator.generate_kernel(operation_analysis)
            
            # Step 5: Validate kernel correctness
            logger.info("Validating kernel correctness...")
            validation_result = self.validator.validate_kernel(
                initial_kernel, operation_analysis
            )
            
            if not validation_result["is_valid"]:
                logger.error(f"Kernel validation failed: {validation_result['errors']}")
                return self._create_error_result("Kernel validation failed", validation_result)
            
            # Step 6: Benchmark original operation
            logger.info("Benchmarking original operation...")
            original_metrics = self.benchmarker.benchmark_operation(result_tensor)
            
            # Step 7: Optimize kernel using evolutionary algorithm
            logger.info("Starting kernel optimization...")
            optimized_kernel = self._optimize_kernel_evolutionary(
                initial_kernel, operation_analysis, original_metrics
            )
            
            # Step 8: Benchmark optimized kernel
            logger.info("Benchmarking optimized kernel...")
            optimized_metrics = self.benchmarker.benchmark_kernel(
                optimized_kernel, operation_analysis
            )
            
            # Step 9: Calculate speedup
            speedup = original_metrics["execution_time"] / optimized_metrics["execution_time"]
            
            # Step 10: Prepare results
            results = {
                "success": True,
                "original_metrics": original_metrics,
                "optimized_metrics": optimized_metrics,
                "speedup": speedup,
                "optimized_kernel": optimized_kernel,
                "operation_analysis": operation_analysis,
                "captured_operation": operation_info,
                "optimization_time": time.time() - start_time,
                "iterations": self.config["max_iterations"],
            }
            
            logger.info(f"Enhanced optimization completed. Speedup: {speedup:.2f}x")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced optimization failed: {str(e)}")
            return self._create_error_result("Enhanced optimization failed", {"error": str(e)})
    
    def _optimize_kernel_evolutionary(self, initial_kernel: str, 
                                    operation_analysis: Dict[str, Any],
                                    original_metrics: Dict[str, Any]) -> str:
        """
        Optimize kernel using evolutionary algorithm.
        
        Args:
            initial_kernel: Initial CUDA kernel code
            operation_analysis: Detailed analysis from EnhancedPyTorchOperationParser
            original_metrics: Performance metrics of original operation
            
        Returns:
            Optimized CUDA kernel code
        """
        population = [initial_kernel]
        
        for iteration in range(self.config["max_iterations"]):
            logger.info(f"Evolutionary iteration {iteration + 1}/{self.config['max_iterations']}")
            
            # Generate new population through crossover and mutation
            new_population = []
            
            for _ in range(self.config["population_size"]):
                if len(population) > 1 and np.random.random() < self.config["crossover_rate"]:
                    # Crossover
                    parent1, parent2 = np.random.choice(population, 2, replace=False)
                    child = self.kernel_generator.crossover_kernels(parent1, parent2)
                else:
                    # Mutation
                    parent = np.random.choice(population)
                    child = self.kernel_generator.mutate_kernel(parent)
                
                new_population.append(child)
            
            # Evaluate fitness of new population
            fitness_scores = []
            for kernel in new_population:
                try:
                    metrics = self.benchmarker.benchmark_kernel(kernel, operation_analysis)
                    speedup = original_metrics["execution_time"] / metrics["execution_time"]
                    fitness_scores.append(speedup)
                except Exception:
                    fitness_scores.append(0.0)
            
            # Select best kernels for next generation
            population = [kernel for _, kernel in sorted(
                zip(fitness_scores, new_population), reverse=True
            )[:self.config["population_size"]//2]]
            
            # Keep best kernel from previous generation
            if len(population) > 0:
                best_kernel = population[0]
                best_speedup = max(fitness_scores)
                logger.info(f"Best speedup in iteration {iteration + 1}: {best_speedup:.2f}x")
        
        return population[0] if population else initial_kernel
    
    def _create_error_result(self, message: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Create an error result dictionary."""
        return {
            "success": False,
            "error": message,
            "details": details,
            "optimization_time": 0.0,
        }
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported PyTorch operations."""
        return self.parser.get_supported_operations()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_optimizations": 0,  # TODO: Implement tracking
            "average_speedup": 0.0,    # TODO: Implement tracking
            "fusion_applications": 0,   # TODO: Implement tracking
        }
    
    def clear_operation_history(self):
        """Clear the operation capture history."""
        self.operation_capture.clear_history()
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get the complete operation capture history."""
        return self.operation_capture.get_operation_history() 