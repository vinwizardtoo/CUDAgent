"""
CUDAgent: AI-Powered CUDA Kernel Optimization

Transform PyTorch code into highly optimized CUDA kernels using AI.

Quick Start:
    >>> from cudagent import CUDAgent
    >>> agent = CUDAgent()
    >>> optimized_kernel = agent.optimize("matmul", [(100, 100), (100, 100)])
"""

__version__ = "0.1.0"
__author__ = "CUDAgent Contributors"

# Core components
from .core.optimizer import CUDAgentOptimizer
from .core.enhanced_optimizer import EnhancedCUDAAgentOptimizer

# Parsers
from .parsers.pytorch_parser import PyTorchParser
from .parsers.operation_capture import OperationCapture

# Profiling
from .profiling.benchmarker import Benchmarker

# Utils
from .utils.kernel_generator import KernelGenerator
from .utils.enhanced_kernel_generator import EnhancedKernelGenerator
from .utils.validation import KernelValidator

# Agents
from .agents.config_manager import ConfigManager
from .agents.llm_agent import LLMAgent
from .agents.kernel_optimizer import KernelOptimizer
from .agents.performance_advisor import PerformanceAdvisor
from .agents.validation_agent import ValidationAgent

# Main API class
class CUDAgent:
    """
    Main CUDAgent class for easy usage.
    
    This class provides a simple interface to all CUDAgent functionality.
    """
    
    def __init__(self, config_file="config/api_keys.json"):
        """Initialize CUDAgent with configuration."""
        self.config_manager = ConfigManager(config_file)
        self.llm_agent = LLMAgent(self.config_manager)
        self.kernel_optimizer = KernelOptimizer(self.llm_agent)
        self.performance_advisor = PerformanceAdvisor()
        self.validation_agent = ValidationAgent()
        self.parser = PyTorchParser()
        self.generator = KernelGenerator()
        self.benchmarker = Benchmarker()
    
    def optimize(self, operation_type, input_shapes, output_shape=None, dtype="float32"):
        """
        Optimize a CUDA kernel for the given operation.
        
        Args:
            operation_type (str): Type of operation (e.g., "matmul", "conv2d")
            input_shapes (list): List of input tensor shapes
            output_shape (tuple, optional): Output tensor shape
            dtype (str): Data type for the operation
            
        Returns:
            str: Optimized CUDA kernel code
        """
        # Generate initial kernel
        if output_shape is None:
            # Infer output shape for common operations
            if operation_type == "matmul":
                output_shape = (input_shapes[0][0], input_shapes[1][1])
            else:
                output_shape = input_shapes[0]  # Default to first input shape
        
        initial_kernel = self.generator.generate_kernel(
            operation_type=operation_type,
            input_shapes=input_shapes,
            output_shape=output_shape,
            dtype=dtype
        )
        
        if not initial_kernel:
            raise ValueError(f"Failed to generate kernel for {operation_type}")
        
        # Optimize the kernel
        optimized_kernel = self.kernel_optimizer.optimize_kernel(
            initial_kernel,
            operation_type=operation_type,
            input_shapes=input_shapes
        )
        
        return optimized_kernel
    
    def capture_and_optimize(self, functions):
        """
        Capture PyTorch operations from functions and optimize them.
        
        Args:
            functions (list): List of PyTorch functions to capture
            
        Returns:
            list: List of optimized kernels
        """
        operations = self.parser.capture_operations(functions)
        optimized_kernels = []
        
        for operation in operations:
            kernel = self.optimize(
                operation['operation_type'],
                operation['input_shapes'],
                operation['output_shape'],
                operation['dtype']
            )
            optimized_kernels.append(kernel)
        
        return optimized_kernels
    
    def benchmark(self, operation_type, input_shapes, iterations=100):
        """
        Benchmark an operation.
        
        Args:
            operation_type (str): Type of operation
            input_shapes (list): Input tensor shapes
            iterations (int): Number of iterations for benchmarking
            
        Returns:
            dict: Benchmark results
        """
        return self.benchmarker.benchmark_operation(
            operation_type=operation_type,
            input_shapes=input_shapes,
            iterations=iterations
        )
    
    def validate(self, kernel_code, operation_type, input_shapes):
        """
        Validate a CUDA kernel.
        
        Args:
            kernel_code (str): CUDA kernel code
            operation_type (str): Type of operation
            input_shapes (list): Input tensor shapes
            
        Returns:
            dict: Validation results
        """
        return self.validation_agent.validate_kernel(
            kernel_code=kernel_code,
            operation_type=operation_type,
            input_shapes=input_shapes
        )

# Public API
__all__ = [
    # Main class
    "CUDAgent",
    
    # Core components
    "CUDAgentOptimizer",
    "EnhancedCUDAAgentOptimizer",
    
    # Parsers
    "PyTorchParser",
    "OperationCapture",
    
    # Profiling
    "Benchmarker",
    
    # Utils
    "KernelGenerator",
    "EnhancedKernelGenerator", 
    "KernelValidator",
    
    # Agents
    "ConfigManager",
    "LLMAgent",
    "KernelOptimizer",
    "PerformanceAdvisor",
    "ValidationAgent",
] 