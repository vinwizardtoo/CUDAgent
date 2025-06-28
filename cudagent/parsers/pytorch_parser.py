"""
Enhanced PyTorch operation parser for analyzing captured operations and extracting optimization information.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
import inspect

logger = logging.getLogger(__name__)


class EnhancedPyTorchOperationParser:
    """
    Enhanced parser for analyzing captured PyTorch operations and extracting computational information
    needed for CUDA kernel generation.
    """
    
    def __init__(self):
        """Initialize the enhanced PyTorch operation parser."""
        self.supported_operations = {
            'matmul': self._parse_matmul_operation,
            'add': self._parse_add_operation,
            'mul': self._parse_mul_operation,
            'relu': self._parse_activation_operation,
            'sigmoid': self._parse_activation_operation,
            'tanh': self._parse_activation_operation,
            'conv2d': self._parse_conv2d_operation,
            'max_pool2d': self._parse_pooling_operation,
            'avg_pool2d': self._parse_pooling_operation,
            'batch_norm': self._parse_batch_norm_operation,
            'layer_norm': self._parse_layer_norm_operation,
            'softmax': self._parse_softmax_operation,
            'dropout': self._parse_dropout_operation,
        }
    
    def parse_captured_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a captured PyTorch operation and extract computational information.
        
        Args:
            operation_info: Captured operation information from OperationCapture
            
        Returns:
            Dictionary containing detailed operation analysis
        """
        try:
            function_name = operation_info.get('function_name', 'unknown')
            
            # Parse operation-specific information
            if function_name in self.supported_operations:
                operation_analysis = self.supported_operations[function_name](operation_info)
            else:
                operation_analysis = self._parse_generic_operation(operation_info)
            
            # Add common analysis
            result = {
                "operation_type": function_name,
                "operation_info": operation_analysis,
                "tensor_info": self._extract_tensor_info(operation_info),
                "is_supported": function_name in self.supported_operations,
                "complexity": self._estimate_complexity(operation_info, function_name),
                "optimization_opportunities": self._identify_optimization_opportunities(operation_info),
                "memory_analysis": self._analyze_memory_patterns(operation_info),
                "parallelization_strategy": self._determine_parallelization_strategy(operation_info),
            }
            
            logger.info(f"Parsed captured operation: {function_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse captured operation: {str(e)}")
            return {
                "operation_type": "unknown",
                "error": str(e),
                "is_supported": False,
            }
    
    def _extract_tensor_info(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive tensor information from captured operation."""
        tensor_info = {
            "input_tensors": [],
            "output_tensor": None,
            "device": operation_info.get('device'),
            "dtype": None,
        }
        
        # Extract input tensor information
        for tensor_arg in operation_info.get('tensor_args', []):
            tensor_info["input_tensors"].append({
                "shape": tensor_arg['shape'],
                "dtype": tensor_arg['dtype'],
                "device": tensor_arg['device'],
                "requires_grad": tensor_arg['requires_grad'],
                "is_contiguous": tensor_arg['is_contiguous'],
                "memory_format": tensor_arg['memory_format'],
                "numel": tensor_arg['tensor'].numel(),
                "dim": tensor_arg['tensor'].dim(),
            })
        
        # Extract output tensor information
        if 'result' in operation_info:
            result = operation_info['result']
            tensor_info["output_tensor"] = {
                "shape": operation_info['result_shape'],
                "dtype": operation_info['result_dtype'],
                "device": operation_info['result_device'],
                "numel": result.numel(),
                "dim": result.dim(),
            }
            tensor_info["dtype"] = operation_info['result_dtype']
        
        return tensor_info
    
    def _parse_matmul_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse matrix multiplication operation with real parameters."""
        params = operation_info.get('operation_params', {})
        tensor_info = operation_info.get('tensor_info', {})
        
        M = params.get('M', 1)
        K = params.get('K', 1)
        N = params.get('N', 1)
        
        # Calculate real computational complexity
        compute_intensity = M * N * K * 2  # 2 FLOPS per multiply-add
        memory_access = M * K + K * N + M * N  # Input A + Input B + Output C
        
        return {
            "input_shapes": [params.get('A_shape'), params.get('B_shape')],
            "output_shape": (M, N),
            "compute_intensity": compute_intensity,
            "memory_access": memory_access,
            "arithmetic_intensity": compute_intensity / memory_access,
            "parallelization": "2D",
            "block_size_optimization": self._optimize_matmul_block_size(M, N, K),
            "memory_layout": self._analyze_matmul_memory_layout(params),
            "optimization_strategies": [
                "tile_based",
                "shared_memory",
                "coalesced_access",
                "loop_unrolling",
                "register_tiling"
            ],
            "transpose_flags": {
                "transpose_A": params.get('transpose_A', False),
                "transpose_B": params.get('transpose_B', False),
            }
        }
    
    def _parse_conv2d_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse 2D convolution operation with real parameters."""
        params = operation_info.get('operation_params', {})
        
        input_shape = params.get('input_shape', (1, 1, 1, 1))
        weight_shape = params.get('weight_shape', (1, 1, 1, 1))
        kernel_size = params.get('kernel_size', (1, 1))
        stride = params.get('stride', (1, 1))
        padding = params.get('padding', (0, 0))
        
        batch_size, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        
        # Calculate output dimensions
        out_height = (height + 2 * padding[0] - kernel_h) // stride[0] + 1
        out_width = (width + 2 * padding[1] - kernel_w) // stride[1] + 1
        
        # Calculate computational complexity
        compute_intensity = batch_size * out_channels * out_height * out_width * in_channels * kernel_h * kernel_w * 2
        memory_access = batch_size * in_channels * height * width + out_channels * in_channels * kernel_h * kernel_w + batch_size * out_channels * out_height * out_width
        
        return {
            "input_shapes": [input_shape, weight_shape],
            "output_shape": (batch_size, out_channels, out_height, out_width),
            "compute_intensity": compute_intensity,
            "memory_access": memory_access,
            "arithmetic_intensity": compute_intensity / memory_access,
            "parallelization": "3D",
            "block_size_optimization": self._optimize_conv2d_block_size(out_height, out_width, out_channels),
            "memory_layout": self._analyze_conv2d_memory_layout(params),
            "optimization_strategies": [
                "im2col",
                "shared_memory",
                "tile_based",
                "coalesced_access",
                "register_tiling",
                "loop_unrolling"
            ],
            "convolution_params": {
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": params.get('dilation', (1, 1)),
                "groups": params.get('groups', 1),
            }
        }
    
    def _parse_add_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse addition operation with real parameters."""
        params = operation_info.get('operation_params', {})
        
        A_shape = params.get('A_shape', (1, 1))
        B_shape = params.get('B_shape', (1, 1))
        alpha = params.get('alpha', 1.0)
        
        # Calculate complexity
        total_elements = max(A_shape[0] * A_shape[1], B_shape[0] * B_shape[1])
        compute_intensity = total_elements
        memory_access = total_elements * 3  # Read A, Read B, Write C
        
        return {
            "input_shapes": [A_shape, B_shape],
            "output_shape": A_shape,  # Assuming broadcasting
            "compute_intensity": compute_intensity,
            "memory_access": memory_access,
            "arithmetic_intensity": compute_intensity / memory_access,
            "parallelization": "1D",
            "block_size_optimization": self._optimize_elementwise_block_size(total_elements),
            "memory_layout": self._analyze_elementwise_memory_layout(params),
            "optimization_strategies": [
                "vectorized_loads",
                "coalesced_access",
                "loop_unrolling"
            ],
            "broadcasting": params.get('broadcasting', False),
            "alpha": alpha,
        }
    
    def _parse_mul_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse multiplication operation."""
        return self._parse_add_operation(operation_info)
    
    def _parse_activation_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse activation function operation."""
        params = operation_info.get('operation_params', {})
        input_shape = params.get('input_shape', (1, 1))
        
        total_elements = input_shape[0] * input_shape[1]
        compute_intensity = total_elements * 2  # Rough estimate for activation
        memory_access = total_elements * 2  # Read input, Write output
        
        return {
            "input_shapes": [input_shape],
            "output_shape": input_shape,
            "compute_intensity": compute_intensity,
            "memory_access": memory_access,
            "arithmetic_intensity": compute_intensity / memory_access,
            "parallelization": "1D",
            "block_size_optimization": self._optimize_elementwise_block_size(total_elements),
            "memory_layout": self._analyze_elementwise_memory_layout(params),
            "optimization_strategies": [
                "vectorized_compute",
                "coalesced_access",
                "loop_unrolling"
            ],
            "activation_type": operation_info.get('function_name', 'unknown'),
        }
    
    def _parse_pooling_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pooling operation."""
        params = operation_info.get('operation_params', {})
        input_shape = params.get('input_shape', (1, 1, 1, 1))
        kernel_size = params.get('kernel_size', (2, 2))
        
        batch_size, channels, height, width = input_shape
        kernel_h, kernel_w = kernel_size
        
        # Calculate output dimensions (simplified)
        out_height = height // kernel_h
        out_width = width // kernel_w
        
        total_elements = batch_size * channels * out_height * out_width
        compute_intensity = total_elements * kernel_h * kernel_w
        memory_access = total_elements * (kernel_h * kernel_w + 1)  # Read window + Write output
        
        return {
            "input_shapes": [input_shape],
            "output_shape": (batch_size, channels, out_height, out_width),
            "compute_intensity": compute_intensity,
            "memory_access": memory_access,
            "arithmetic_intensity": compute_intensity / memory_access,
            "parallelization": "2D",
            "block_size_optimization": self._optimize_pooling_block_size(out_height, out_width),
            "memory_layout": self._analyze_pooling_memory_layout(params),
            "optimization_strategies": [
                "shared_memory",
                "coalesced_access",
                "loop_unrolling"
            ],
            "pooling_params": {
                "kernel_size": kernel_size,
                "stride": params.get('stride'),
                "padding": params.get('padding', 0),
            }
        }
    
    def _parse_batch_norm_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse batch normalization operation."""
        # Implementation similar to activation functions
        return self._parse_activation_operation(operation_info)
    
    def _parse_layer_norm_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse layer normalization operation."""
        # Implementation similar to activation functions
        return self._parse_activation_operation(operation_info)
    
    def _parse_softmax_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse softmax operation."""
        # Implementation similar to activation functions
        return self._parse_activation_operation(operation_info)
    
    def _parse_dropout_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse dropout operation."""
        # Implementation similar to activation functions
        return self._parse_activation_operation(operation_info)
    
    def _parse_generic_operation(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse generic operation."""
        return {
            "input_shapes": operation_info.get('input_shapes', []),
            "output_shape": operation_info.get('result_shape', (1, 1)),
            "compute_intensity": 1000,  # Default estimate
            "memory_access": 1000,      # Default estimate
            "parallelization": "1D",
            "optimization_strategies": ["generic"],
        }
    
    def _optimize_matmul_block_size(self, M: int, N: int, K: int) -> Dict[str, Any]:
        """Optimize block size for matrix multiplication."""
        # Simple heuristic-based optimization
        if M <= 32 and N <= 32:
            return {"block_x": 16, "block_y": 16}
        elif M <= 64 and N <= 64:
            return {"block_x": 32, "block_y": 32}
        else:
            return {"block_x": 16, "block_y": 16}
    
    def _optimize_conv2d_block_size(self, out_height: int, out_width: int, out_channels: int) -> Dict[str, Any]:
        """Optimize block size for 2D convolution."""
        return {"block_x": 16, "block_y": 16, "block_z": 4}
    
    def _optimize_elementwise_block_size(self, total_elements: int) -> Dict[str, Any]:
        """Optimize block size for elementwise operations."""
        if total_elements <= 1024:
            return {"block_x": 256}
        else:
            return {"block_x": 512}
    
    def _optimize_pooling_block_size(self, out_height: int, out_width: int) -> Dict[str, Any]:
        """Optimize block size for pooling operations."""
        return {"block_x": 16, "block_y": 16}
    
    def _analyze_matmul_memory_layout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory layout for matrix multiplication."""
        return {
            "coalesced_access": True,
            "shared_memory_usage": "high",
            "memory_hierarchy": ["global", "shared", "registers"],
        }
    
    def _analyze_conv2d_memory_layout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory layout for 2D convolution."""
        return {
            "coalesced_access": True,
            "shared_memory_usage": "high",
            "memory_hierarchy": ["global", "shared", "registers"],
        }
    
    def _analyze_elementwise_memory_layout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory layout for elementwise operations."""
        return {
            "coalesced_access": True,
            "shared_memory_usage": "low",
            "memory_hierarchy": ["global", "registers"],
        }
    
    def _analyze_pooling_memory_layout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory layout for pooling operations."""
        return {
            "coalesced_access": True,
            "shared_memory_usage": "medium",
            "memory_hierarchy": ["global", "shared", "registers"],
        }
    
    def _estimate_complexity(self, operation_info: Dict[str, Any], operation_type: str) -> str:
        """Estimate computational complexity."""
        params = operation_info.get('operation_params', {})
        
        if operation_type == 'matmul':
            M, K, N = params.get('M', 1), params.get('K', 1), params.get('N', 1)
            flops = M * N * K * 2
        elif operation_type == 'conv2d':
            input_shape = params.get('input_shape', (1, 1, 1, 1))
            weight_shape = params.get('weight_shape', (1, 1, 1, 1))
            kernel_size = params.get('kernel_size', (1, 1))
            flops = input_shape[0] * weight_shape[0] * input_shape[2] * input_shape[3] * input_shape[1] * kernel_size[0] * kernel_size[1] * 2
        else:
            flops = 1000  # Default estimate
        
        if flops < 1000:
            return "low"
        elif flops < 1000000:
            return "medium"
        else:
            return "high"
    
    def _identify_optimization_opportunities(self, operation_info: Dict[str, Any]) -> List[str]:
        """Identify optimization opportunities for the operation."""
        opportunities = []
        params = operation_info.get('operation_params', {})
        
        # Memory optimization opportunities
        if any(not t['is_contiguous'] for t in operation_info.get('tensor_args', [])):
            opportunities.append("memory_layout_optimization")
        
        # Parallelization opportunities
        if operation_info.get('function_name') in ['matmul', 'conv2d']:
            opportunities.append("high_parallelization")
        
        # Precision optimization
        if any(t['dtype'] == torch.float32 for t in operation_info.get('tensor_args', [])):
            opportunities.append("mixed_precision")
        
        return opportunities
    
    def _analyze_memory_patterns(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory access patterns."""
        return {
            "access_pattern": "coalesced",
            "memory_bandwidth": "high",
            "cache_efficiency": "good",
        }
    
    def _determine_parallelization_strategy(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine optimal parallelization strategy."""
        function_name = operation_info.get('function_name', 'unknown')
        
        if function_name == 'matmul':
            return {"strategy": "2D_grid", "dimensions": 2}
        elif function_name == 'conv2d':
            return {"strategy": "3D_grid", "dimensions": 3}
        else:
            return {"strategy": "1D_grid", "dimensions": 1}
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported PyTorch operations."""
        return list(self.supported_operations.keys())
    
    def is_operation_supported(self, operation_info: Dict[str, Any]) -> bool:
        """Check if operation is supported."""
        function_name = operation_info.get('function_name', 'unknown')
        return function_name in self.supported_operations

# Add backward compatibility alias at the end of the file
class PyTorchOperationParser(EnhancedPyTorchOperationParser):
    """
    Backward compatibility alias for the original PyTorchOperationParser.
    This maintains compatibility with existing code while providing access to enhanced functionality.
    """
    
    def parse_operation(self, operation: torch.Tensor) -> Dict[str, Any]:
        """
        Parse a PyTorch operation (backward compatibility method).
        This method is deprecated - use parse_captured_operation with captured operation info instead.
        """
        import warnings
        warnings.warn(
            "PyTorchOperationParser.parse_operation is deprecated. "
            "Use EnhancedPyTorchOperationParser.parse_captured_operation with captured operation info instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Create a minimal operation info structure for backward compatibility
        operation_info = {
            'function_name': 'unknown',
            'tensor_args': [{
                'tensor': operation,
                'shape': operation.shape,
                'dtype': operation.dtype,
                'device': operation.device,
                'requires_grad': operation.requires_grad,
                'is_contiguous': operation.is_contiguous(),
                'memory_format': 'unknown'
            }],
            'result': operation,
            'result_shape': operation.shape,
            'result_dtype': operation.dtype,
            'result_device': operation.device,
            'device': operation.device,
            'operation_params': {}
        }
        
        return self.parse_captured_operation(operation_info)

# Add PyTorchParser alias for the new API
class PyTorchParser(EnhancedPyTorchOperationParser):
    """
    PyTorchParser - Main parser class for the new CUDAgent API.
    This provides a clean interface for capturing and parsing PyTorch operations.
    """
    
    def capture_operations(self, functions):
        """
        Capture operations from a list of functions.
        
        Args:
            functions: List of PyTorch functions to capture
            
        Returns:
            List of captured operation information
        """
        from ..parsers.operation_capture import OperationCapture
        
        capture = OperationCapture()
        operations = []
        
        for func in functions:
            try:
                # Create mock tensors for testing
                import numpy as np
                if func.__name__ == 'matmul':
                    a = torch.randn(100, 100)
                    b = torch.randn(100, 100)
                    result = func(a, b)
                elif func.__name__ == 'add':
                    a = torch.randn(100, 100)
                    b = torch.randn(100, 100)
                    result = func(a, b)
                else:
                    # Generic case
                    a = torch.randn(10, 10)
                    result = func(a)
                
                # Capture the operation
                capture.capture_operation(func, a, b) if func.__name__ in ['matmul', 'add'] else capture.capture_operation(func, a)
                operation_info = capture.get_last_operation()
                
                if operation_info:
                    # Parse the operation
                    parsed = self.parse_captured_operation(operation_info)
                    operations.append({
                        'operation_type': parsed['operation_type'],
                        'input_shapes': parsed['operation_info']['input_shapes'],
                        'output_shape': parsed['operation_info']['output_shape'],
                        'dtype': parsed['tensor_info']['dtype'] or 'float32'
                    })
                
            except Exception as e:
                logger.warning(f"Failed to capture operation from {func.__name__}: {e}")
                continue
        
        return operations 