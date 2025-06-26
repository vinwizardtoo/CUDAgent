"""
Operation capture system for intercepting PyTorch operations and extracting full context.
"""

import logging
import inspect
from typing import Dict, Any, List, Optional, Callable, Union
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OperationCapture:
    """
    Captures PyTorch operations with their full context including parameters,
    input tensors, and operation metadata.
    """
    
    def __init__(self):
        """Initialize the operation capture system."""
        self.operation_history = []
        self.capture_enabled = True
    
    def capture_operation(self, operation_func: Callable, *args, **kwargs) -> torch.Tensor:
        """
        Capture a PyTorch operation with its full context.
        
        Args:
            operation_func: The PyTorch operation function (e.g., torch.matmul)
            *args: Positional arguments to the operation
            **kwargs: Keyword arguments to the operation
            
        Returns:
            Result tensor from the operation
        """
        if not self.capture_enabled:
            return operation_func(*args, **kwargs)
        
        try:
            # Extract operation information
            operation_info = self._extract_operation_info(operation_func, args, kwargs)
            
            # Execute the operation
            result = operation_func(*args, **kwargs)
            
            # Add result information
            operation_info['result'] = result
            operation_info['result_shape'] = result.shape
            operation_info['result_dtype'] = result.dtype
            operation_info['result_device'] = result.device
            
            # Store in history
            self.operation_history.append(operation_info)
            
            logger.debug(f"Captured operation: {operation_info['function_name']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to capture operation: {str(e)}")
            # Fallback to direct execution
            return operation_func(*args, **kwargs)
    
    def _extract_operation_info(self, operation_func: Callable, 
                               args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract comprehensive information about the operation."""
        
        # Basic function information
        operation_info = {
            'function': operation_func,
            'function_name': operation_func.__name__,
            'module_name': operation_func.__module__,
            'args': args,
            'kwargs': kwargs,
        }
        
        # Extract tensor information
        tensor_args = []
        tensor_kwargs = {}
        
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_args.append({
                    'index': i,
                    'tensor': arg,
                    'shape': arg.shape,
                    'dtype': arg.dtype,
                    'device': arg.device,
                    'requires_grad': arg.requires_grad,
                    'is_contiguous': arg.is_contiguous(),
                    'memory_format': str(arg.memory_format) if hasattr(arg, 'memory_format') else 'unknown'
                })
        
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                tensor_kwargs[key] = {
                    'tensor': value,
                    'shape': value.shape,
                    'dtype': value.dtype,
                    'device': value.device,
                    'requires_grad': value.requires_grad,
                    'is_contiguous': value.is_contiguous(),
                    'memory_format': str(value.memory_format) if hasattr(value, 'memory_format') else 'unknown'
                }
        
        operation_info['tensor_args'] = tensor_args
        operation_info['tensor_kwargs'] = tensor_kwargs
        operation_info['input_tensors'] = [t['tensor'] for t in tensor_args] + list(tensor_kwargs.values())
        operation_info['input_shapes'] = [t['shape'] for t in tensor_args] + [t['shape'] for t in tensor_kwargs.values()]
        operation_info['input_dtypes'] = [t['dtype'] for t in tensor_args] + [t['dtype'] for t in tensor_kwargs.values()]
        
        # Extract operation-specific parameters
        operation_info['operation_params'] = self._extract_operation_params(
            operation_func, args, kwargs
        )
        
        # Determine device from input tensors
        if tensor_args:
            operation_info['device'] = tensor_args[0]['device']
        elif tensor_kwargs:
            operation_info['device'] = list(tensor_kwargs.values())[0]['device']
        else:
            operation_info['device'] = None
        
        return operation_info
    
    def _extract_operation_params(self, operation_func: Callable, 
                                 args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract operation-specific parameters based on function type."""
        function_name = operation_func.__name__
        
        if function_name == 'matmul':
            return self._extract_matmul_params(args, kwargs)
        elif function_name == 'conv2d':
            return self._extract_conv2d_params(args, kwargs)
        elif function_name == 'add':
            return self._extract_add_params(args, kwargs)
        elif function_name == 'mul':
            return self._extract_mul_params(args, kwargs)
        elif function_name in ['relu', 'sigmoid', 'tanh']:
            return self._extract_activation_params(args, kwargs)
        elif function_name in ['max_pool2d', 'avg_pool2d']:
            return self._extract_pooling_params(args, kwargs)
        else:
            return self._extract_generic_params(args, kwargs)
    
    def _extract_matmul_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract matrix multiplication parameters."""
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            A, B = args[0], args[1]
            return {
                'A_shape': A.shape,
                'B_shape': B.shape,
                'M': A.shape[0] if len(A.shape) >= 2 else 1,
                'K': A.shape[-1] if len(A.shape) >= 2 else A.shape[0],
                'N': B.shape[-1] if len(B.shape) >= 2 else B.shape[0],
                'transpose_A': kwargs.get('transpose', False),
                'transpose_B': kwargs.get('transpose', False),
            }
        return {}
    
    def _extract_conv2d_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract 2D convolution parameters."""
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            input_tensor, weight = args[0], args[1]
            return {
                'input_shape': input_tensor.shape,
                'weight_shape': weight.shape,
                'in_channels': input_tensor.shape[1],
                'out_channels': weight.shape[0],
                'kernel_size': (weight.shape[2], weight.shape[3]),
                'stride': kwargs.get('stride', (1, 1)),
                'padding': kwargs.get('padding', (0, 0)),
                'dilation': kwargs.get('dilation', (1, 1)),
                'groups': kwargs.get('groups', 1),
                'bias': kwargs.get('bias', None),
            }
        return {}
    
    def _extract_add_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract addition parameters."""
        if len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor):
            A, B = args[0], args[1]
            return {
                'A_shape': A.shape,
                'B_shape': B.shape,
                'alpha': kwargs.get('alpha', 1.0),
                'broadcasting': A.shape != B.shape,
            }
        return {}
    
    def _extract_mul_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract multiplication parameters."""
        return self._extract_add_params(args, kwargs)
    
    def _extract_activation_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract activation function parameters."""
        if args and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
            return {
                'input_shape': input_tensor.shape,
                'inplace': kwargs.get('inplace', False),
            }
        return {}
    
    def _extract_pooling_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract pooling parameters."""
        if args and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
            return {
                'input_shape': input_tensor.shape,
                'kernel_size': kwargs.get('kernel_size', (2, 2)),
                'stride': kwargs.get('stride', None),
                'padding': kwargs.get('padding', 0),
                'dilation': kwargs.get('dilation', 1),
                'return_indices': kwargs.get('return_indices', False),
                'ceil_mode': kwargs.get('ceil_mode', False),
            }
        return {}
    
    def _extract_generic_params(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract generic parameters for unknown operations."""
        return {
            'num_args': len(args),
            'num_kwargs': len(kwargs),
            'arg_types': [type(arg).__name__ for arg in args],
            'kwarg_keys': list(kwargs.keys()),
        }
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get the complete operation history."""
        return self.operation_history.copy()
    
    def clear_history(self):
        """Clear the operation history."""
        self.operation_history.clear()
    
    def enable_capture(self):
        """Enable operation capture."""
        self.capture_enabled = True
    
    def disable_capture(self):
        """Disable operation capture."""
        self.capture_enabled = False
    
    def get_last_operation(self) -> Optional[Dict[str, Any]]:
        """Get the most recent operation."""
        return self.operation_history[-1] if self.operation_history else None 