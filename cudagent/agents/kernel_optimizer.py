"""
Kernel Optimization Agent

This module provides specialized kernel optimization capabilities including
parameter tuning, memory access optimization, and kernel fusion strategies.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class KernelParameters:
    """CUDA kernel execution parameters."""
    block_size: Tuple[int, int, int] = (16, 16, 1)
    grid_size: Tuple[int, int, int] = (1, 1, 1)
    shared_memory_size: int = 0
    max_blocks_per_sm: int = 32
    occupancy: float = 0.0

@dataclass
class OptimizationStrategy:
    """Kernel optimization strategy."""
    name: str
    description: str
    priority: int  # 1-5, higher is more important
    expected_improvement: float
    implementation_complexity: str  # low, medium, high
    requirements: List[str]

class KernelOptimizationAgent:
    """
    Specialized agent for CUDA kernel optimization.
    
    This agent focuses on:
    - Parameter tuning (block/grid sizes)
    - Memory access optimization
    - Kernel fusion opportunities
    - Register usage optimization
    - Shared memory utilization
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.optimization_strategies = self._initialize_strategies()
        self.parameter_history = []
        
    def _initialize_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize optimization strategies."""
        return {
            "memory_coalescing": OptimizationStrategy(
                name="Memory Coalescing",
                description="Optimize memory access patterns for coalesced reads/writes",
                priority=5,
                expected_improvement=2.0,
                implementation_complexity="medium",
                requirements=["Analyze memory access patterns", "Restructure loops"]
            ),
            "shared_memory": OptimizationStrategy(
                name="Shared Memory Usage",
                description="Utilize shared memory for frequently accessed data",
                priority=4,
                expected_improvement=1.8,
                implementation_complexity="high",
                requirements=["Identify data reuse patterns", "Manage shared memory"]
            ),
            "register_optimization": OptimizationStrategy(
                name="Register Optimization",
                description="Optimize register usage and reduce register pressure",
                priority=3,
                expected_improvement=1.3,
                implementation_complexity="medium",
                requirements=["Analyze register usage", "Restructure computations"]
            ),
            "loop_unrolling": OptimizationStrategy(
                name="Loop Unrolling",
                description="Unroll loops to reduce loop overhead and improve instruction-level parallelism",
                priority=3,
                expected_improvement=1.2,
                implementation_complexity="low",
                requirements=["Identify unrollable loops", "Balance unroll factor"]
            ),
            "kernel_fusion": OptimizationStrategy(
                name="Kernel Fusion",
                description="Fuse multiple kernels to reduce kernel launch overhead",
                priority=4,
                expected_improvement=1.5,
                implementation_complexity="high",
                requirements=["Identify fusible operations", "Manage intermediate results"]
            ),
            "parameter_tuning": OptimizationStrategy(
                name="Parameter Tuning",
                description="Optimize block and grid sizes for maximum occupancy",
                priority=4,
                expected_improvement=1.4,
                implementation_complexity="low",
                requirements=["Calculate optimal block sizes", "Maximize occupancy"]
            )
        }
    
    def optimize_kernel_parameters(self, operation_info: Dict[str, Any], 
                                 current_params: Optional[KernelParameters] = None) -> KernelParameters:
        """
        Optimize kernel execution parameters.
        
        Args:
            operation_info: Information about the operation
            current_params: Current kernel parameters (optional)
            
        Returns:
            Optimized kernel parameters
        """
        try:
            logger.info("Starting kernel parameter optimization")
            
            # Analyze operation characteristics
            operation_type = operation_info.get('operation_type', 'unknown')
            tensor_info = operation_info.get('tensor_info', {})
            
            # Get optimal block size based on operation type
            optimal_block_size = self._calculate_optimal_block_size(operation_type, tensor_info)
            
            # Calculate grid size
            grid_size = self._calculate_grid_size(operation_type, tensor_info, optimal_block_size)
            
            # Estimate shared memory requirements
            shared_memory_size = self._estimate_shared_memory_size(operation_type, tensor_info)
            
            # Calculate occupancy
            occupancy = self._estimate_occupancy(optimal_block_size, shared_memory_size)
            
            optimized_params = KernelParameters(
                block_size=optimal_block_size,
                grid_size=grid_size,
                shared_memory_size=shared_memory_size,
                occupancy=occupancy
            )
            
            # Store in history
            self.parameter_history.append({
                'operation_info': operation_info,
                'original_params': current_params,
                'optimized_params': optimized_params,
                'timestamp': time.time()
            })
            
            logger.info(f"Parameter optimization completed. Occupancy: {occupancy:.2f}")
            return optimized_params
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {str(e)}")
            return current_params or KernelParameters()
    
    def _calculate_optimal_block_size(self, operation_type: str, tensor_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Calculate optimal block size for the operation."""
        
        if operation_type == "matmul":
            return self._optimize_matmul_block_size(tensor_info)
        elif operation_type == "conv2d":
            return self._optimize_conv2d_block_size(tensor_info)
        elif operation_type in ["add", "mul", "relu", "sigmoid"]:
            return self._optimize_elementwise_block_size(tensor_info)
        else:
            return self._optimize_generic_block_size(tensor_info)
    
    def _optimize_matmul_block_size(self, tensor_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Optimize block size for matrix multiplication."""
        input_tensors = tensor_info.get('input_tensors', [])
        
        if len(input_tensors) >= 2:
            A_shape = input_tensors[0].get('shape', (1, 1))
            B_shape = input_tensors[1].get('shape', (1, 1))
            
            M, K = A_shape[0], A_shape[1]
            N = B_shape[1] if len(B_shape) > 1 else B_shape[0]
            
            # Optimal block size for matmul (considering shared memory constraints)
            if M >= 512 and N >= 512:
                return (16, 16, 1)  # Large matrices
            elif M >= 256 and N >= 256:
                return (16, 16, 1)  # Medium matrices
            else:
                return (8, 8, 1)   # Small matrices
        
        return (16, 16, 1)  # Default
    
    def _optimize_conv2d_block_size(self, tensor_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Optimize block size for 2D convolution."""
        input_tensors = tensor_info.get('input_tensors', [])
        
        if input_tensors:
            input_shape = input_tensors[0].get('shape', (1, 1, 1, 1))
            if len(input_shape) >= 4:
                height, width = input_shape[2], input_shape[3]
                
                # Optimal block size for conv2d
                if height >= 224 and width >= 224:
                    return (16, 16, 1)  # Large images
                elif height >= 64 and width >= 64:
                    return (16, 16, 1)  # Medium images
                else:
                    return (8, 8, 1)   # Small images
        
        return (16, 16, 1)  # Default
    
    def _optimize_elementwise_block_size(self, tensor_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Optimize block size for elementwise operations."""
        output_tensor = tensor_info.get('output_tensor', {})
        numel = output_tensor.get('numel', 1)
        
        # For elementwise operations, use 1D blocks
        if numel >= 1000000:
            return (256, 1, 1)  # Large tensors
        elif numel >= 100000:
            return (128, 1, 1)  # Medium tensors
        else:
            return (64, 1, 1)   # Small tensors
    
    def _optimize_generic_block_size(self, tensor_info: Dict[str, Any]) -> Tuple[int, int, int]:
        """Optimize block size for generic operations."""
        return (16, 16, 1)  # Conservative default
    
    def _calculate_grid_size(self, operation_type: str, tensor_info: Dict[str, Any], 
                           block_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate grid size based on operation and block size."""
        
        if operation_type == "matmul":
            return self._calculate_matmul_grid_size(tensor_info, block_size)
        elif operation_type == "conv2d":
            return self._calculate_conv2d_grid_size(tensor_info, block_size)
        elif operation_type in ["add", "mul", "relu", "sigmoid"]:
            return self._calculate_elementwise_grid_size(tensor_info, block_size)
        else:
            return self._calculate_generic_grid_size(tensor_info, block_size)
    
    def _calculate_matmul_grid_size(self, tensor_info: Dict[str, Any], 
                                  block_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate grid size for matrix multiplication."""
        input_tensors = tensor_info.get('input_tensors', [])
        
        if len(input_tensors) >= 2:
            A_shape = input_tensors[0].get('shape', (1, 1))
            B_shape = input_tensors[1].get('shape', (1, 1))
            
            M, K = A_shape[0], A_shape[1]
            N = B_shape[1] if len(B_shape) > 1 else B_shape[0]
            
            grid_x = (N + block_size[0] - 1) // block_size[0]
            grid_y = (M + block_size[1] - 1) // block_size[1]
            
            return (grid_x, grid_y, 1)
        
        return (1, 1, 1)
    
    def _calculate_conv2d_grid_size(self, tensor_info: Dict[str, Any], 
                                  block_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate grid size for 2D convolution."""
        input_tensors = tensor_info.get('input_tensors', [])
        
        if input_tensors:
            input_shape = input_tensors[0].get('shape', (1, 1, 1, 1))
            if len(input_shape) >= 4:
                height, width = input_shape[2], input_shape[3]
                
                # Estimate output dimensions (simplified)
                out_height = height - 2  # Assuming 3x3 kernel, no padding
                out_width = width - 2
                
                grid_x = (out_width + block_size[0] - 1) // block_size[0]
                grid_y = (out_height + block_size[1] - 1) // block_size[1]
                
                return (grid_x, grid_y, 1)
        
        return (1, 1, 1)
    
    def _calculate_elementwise_grid_size(self, tensor_info: Dict[str, Any], 
                                       block_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate grid size for elementwise operations."""
        output_tensor = tensor_info.get('output_tensor', {})
        numel = output_tensor.get('numel', 1)
        
        grid_x = (numel + block_size[0] - 1) // block_size[0]
        
        return (grid_x, 1, 1)
    
    def _calculate_generic_grid_size(self, tensor_info: Dict[str, Any], 
                                   block_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Calculate grid size for generic operations."""
        return (1, 1, 1)
    
    def _estimate_shared_memory_size(self, operation_type: str, tensor_info: Dict[str, Any]) -> int:
        """Estimate shared memory requirements."""
        
        if operation_type == "matmul":
            # Shared memory for tiling
            return 4096  # 2 * 32 * 32 * 4 bytes (float)
        elif operation_type == "conv2d":
            # Shared memory for input tiles
            return 2048  # 16 * 16 * 4 bytes
        else:
            return 0  # No shared memory needed
    
    def _estimate_occupancy(self, block_size: Tuple[int, int, int], shared_memory_size: int) -> float:
        """Estimate kernel occupancy."""
        # Simplified occupancy calculation
        threads_per_block = block_size[0] * block_size[1] * block_size[2]
        
        # Assume 1024 max threads per block, 32 max blocks per SM
        max_threads_per_sm = 1024
        max_blocks_per_sm = 32
        
        # Calculate theoretical occupancy
        blocks_per_sm = min(max_blocks_per_sm, max_threads_per_sm // threads_per_block)
        occupancy = (blocks_per_sm * threads_per_block) / max_threads_per_sm
        
        return min(occupancy, 1.0)
    
    def suggest_optimization_strategies(self, operation_info: Dict[str, Any], 
                                      current_kernel: Optional[str] = None) -> List[OptimizationStrategy]:
        """
        Suggest optimization strategies based on operation analysis.
        
        Args:
            operation_info: Information about the operation
            current_kernel: Current kernel code (optional)
            
        Returns:
            List of suggested optimization strategies
        """
        strategies = []
        operation_type = operation_info.get('operation_type', 'unknown')
        
        # Always suggest parameter tuning
        strategies.append(self.optimization_strategies["parameter_tuning"])
        
        # Suggest memory coalescing for most operations
        if operation_type in ["matmul", "conv2d", "add", "mul"]:
            strategies.append(self.optimization_strategies["memory_coalescing"])
        
        # Suggest shared memory for compute-intensive operations
        if operation_type in ["matmul", "conv2d"]:
            strategies.append(self.optimization_strategies["shared_memory"])
        
        # Suggest loop unrolling for elementwise operations
        if operation_type in ["add", "mul", "relu", "sigmoid"]:
            strategies.append(self.optimization_strategies["loop_unrolling"])
        
        # Suggest register optimization for complex operations
        if operation_type in ["matmul", "conv2d"]:
            strategies.append(self.optimization_strategies["register_optimization"])
        
        # Sort by priority
        strategies.sort(key=lambda s: s.priority, reverse=True)
        
        return strategies
    
    def analyze_kernel_performance(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze kernel code for performance characteristics.
        
        Args:
            kernel_code: CUDA kernel code
            operation_info: Information about the operation
            
        Returns:
            Performance analysis results
        """
        analysis = {
            "memory_access_patterns": self._analyze_memory_access(kernel_code),
            "compute_intensity": self._analyze_compute_intensity(kernel_code),
            "parallelization_efficiency": self._analyze_parallelization(kernel_code),
            "potential_bottlenecks": self._identify_bottlenecks(kernel_code),
            "optimization_opportunities": self._identify_opportunities(kernel_code)
        }
        
        return analysis
    
    def _analyze_memory_access(self, kernel_code: str) -> Dict[str, Any]:
        """Analyze memory access patterns in kernel code."""
        analysis = {
            "coalesced_access": False,
            "shared_memory_usage": False,
            "global_memory_accesses": 0,
            "shared_memory_accesses": 0
        }
        
        # Check for shared memory usage
        if "__shared__" in kernel_code:
            analysis["shared_memory_usage"] = True
            analysis["shared_memory_accesses"] = kernel_code.count("shared_")
        
        # Count global memory accesses
        global_patterns = [r"\[.*\]", r"->", r"\."]
        for pattern in global_patterns:
            analysis["global_memory_accesses"] += len(re.findall(pattern, kernel_code))
        
        # Simple coalescing check (simplified)
        if "threadIdx.x" in kernel_code and "blockIdx.x" in kernel_code:
            analysis["coalesced_access"] = True
        
        return analysis
    
    def _analyze_compute_intensity(self, kernel_code: str) -> Dict[str, Any]:
        """Analyze compute intensity of the kernel."""
        analysis = {
            "arithmetic_operations": 0,
            "memory_operations": 0,
            "compute_intensity_ratio": 0.0
        }
        
        # Count arithmetic operations
        arithmetic_ops = ["+", "-", "*", "/", "fma", "exp", "log", "sin", "cos"]
        for op in arithmetic_ops:
            analysis["arithmetic_operations"] += kernel_code.count(op)
        
        # Count memory operations
        memory_ops = ["[", "]", "->", "."]
        for op in memory_ops:
            analysis["memory_operations"] += kernel_code.count(op)
        
        # Calculate ratio
        if analysis["memory_operations"] > 0:
            analysis["compute_intensity_ratio"] = analysis["arithmetic_operations"] / analysis["memory_operations"]
        
        return analysis
    
    def _analyze_parallelization(self, kernel_code: str) -> Dict[str, Any]:
        """Analyze parallelization efficiency."""
        analysis = {
            "thread_utilization": 0.0,
            "warp_divergence_risk": False,
            "synchronization_points": 0
        }
        
        # Count synchronization points
        analysis["synchronization_points"] = kernel_code.count("__syncthreads")
        
        # Check for potential warp divergence
        if "if" in kernel_code or "else" in kernel_code:
            analysis["warp_divergence_risk"] = True
        
        # Estimate thread utilization (simplified)
        if "threadIdx" in kernel_code and "blockIdx" in kernel_code:
            analysis["thread_utilization"] = 0.8  # Assume good utilization
        
        return analysis
    
    def _identify_bottlenecks(self, kernel_code: str) -> List[str]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        # Check for common bottlenecks
        if kernel_code.count("__syncthreads") > 2:
            bottlenecks.append("Excessive synchronization")
        
        if "if" in kernel_code and kernel_code.count("if") > 3:
            bottlenecks.append("High conditional branching")
        
        if kernel_code.count("[") > kernel_code.count("*") * 2:
            bottlenecks.append("Memory-bound operation")
        
        if "__shared__" not in kernel_code and "matmul" in kernel_code.lower():
            bottlenecks.append("Missing shared memory optimization")
        
        return bottlenecks
    
    def _identify_opportunities(self, kernel_code: str) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Check for optimization opportunities
        if "__shared__" not in kernel_code:
            opportunities.append("Shared memory utilization")
        
        if "for" in kernel_code and "unroll" not in kernel_code:
            opportunities.append("Loop unrolling")
        
        if "threadIdx.x" in kernel_code and "threadIdx.y" not in kernel_code:
            opportunities.append("2D thread utilization")
        
        if kernel_code.count("__syncthreads") == 0 and "shared" in kernel_code:
            opportunities.append("Missing synchronization")
        
        return opportunities
    
    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """Get parameter optimization history."""
        return self.parameter_history.copy()
    
    def clear_history(self):
        """Clear parameter optimization history."""
        self.parameter_history.clear() 