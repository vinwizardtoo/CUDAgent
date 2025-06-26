"""
Kernel validation utilities for checking CUDA kernel correctness and validity.
"""

import logging
import re
from typing import Dict, Any, List, Optional
import torch

logger = logging.getLogger(__name__)


class KernelValidator:
    """
    Validator for checking CUDA kernel correctness and validity.
    """
    
    def __init__(self):
        """Initialize the kernel validator."""
        self.validation_rules = {
            "syntax": self._validate_syntax,
            "semantics": self._validate_semantics,
            "memory": self._validate_memory_access,
            "threading": self._validate_threading,
        }
    
    def validate_kernel(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a CUDA kernel for correctness and validity.
        
        Args:
            kernel_code: CUDA kernel code to validate
            operation_info: Information about the operation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            validation_results = {}
            errors = []
            warnings = []
            
            # Run all validation checks
            for rule_name, validator in self.validation_rules.items():
                result = validator(kernel_code, operation_info)
                validation_results[rule_name] = result
                
                if not result["valid"]:
                    errors.extend(result.get("errors", []))
                if result.get("warnings"):
                    warnings.extend(result["warnings"])
            
            # Overall validation result
            is_valid = len(errors) == 0
            
            return {
                "is_valid": is_valid,
                "validation_results": validation_results,
                "errors": errors,
                "warnings": warnings,
                "operation_type": operation_info.get("operation_type", "unknown"),
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                "is_valid": False,
                "validation_results": {},
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "operation_type": "unknown",
            }
    
    def _validate_syntax(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CUDA kernel syntax."""
        errors = []
        warnings = []
        
        # Check for basic CUDA syntax
        required_patterns = [
            r"__global__\s+void",  # Kernel function declaration
            r"blockIdx\.",         # Block indexing
            r"threadIdx\.",        # Thread indexing
            r"blockDim\.",         # Block dimensions
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, kernel_code):
                errors.append(f"Missing required CUDA syntax: {pattern}")
        
        # Check for common syntax errors
        syntax_errors = [
            (r"__global__\s+int", "Kernel should return void, not int"),
            (r"__global__\s+float", "Kernel should return void, not float"),
            (r"return\s+[^;]+;", "Kernel should not return values"),
        ]
        
        for pattern, error_msg in syntax_errors:
            if re.search(pattern, kernel_code):
                errors.append(error_msg)
        
        # Check for proper bounds checking
        if "if (" not in kernel_code and "if(" not in kernel_code:
            warnings.append("No bounds checking found - may cause out-of-bounds access")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def _validate_semantics(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate kernel semantics and logic."""
        errors = []
        warnings = []
        
        operation_type = operation_info.get("operation_type", "unknown")
        
        # Check for operation-specific requirements
        if operation_type == "matmul":
            if "for" not in kernel_code:
                errors.append("Matrix multiplication should contain loops")
            if "*" not in kernel_code:
                errors.append("Matrix multiplication should contain multiplication")
        
        elif operation_type == "add":
            if "+" not in kernel_code:
                errors.append("Addition operation should contain '+' operator")
        
        elif operation_type == "mul":
            if "*" not in kernel_code:
                errors.append("Multiplication operation should contain '*' operator")
        
        elif operation_type in ["relu", "sigmoid", "tanh"]:
            if operation_type == "relu" and "fmaxf" not in kernel_code and "max" not in kernel_code:
                warnings.append("ReLU operation should use max function")
            elif operation_type == "sigmoid" and "exp" not in kernel_code:
                warnings.append("Sigmoid operation should use exponential function")
            elif operation_type == "tanh" and "tanh" not in kernel_code:
                warnings.append("Tanh operation should use tanh function")
        
        # Check for proper memory access patterns
        if "[" not in kernel_code or "]" not in kernel_code:
            warnings.append("No array indexing found - may not be accessing memory properly")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def _validate_memory_access(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate memory access patterns."""
        errors = []
        warnings = []
        
        # Check for coalesced memory access patterns
        if "threadIdx.x" in kernel_code:
            # Look for patterns that suggest coalesced access
            if re.search(r"\[.*threadIdx\.x.*\]", kernel_code):
                warnings.append("Memory access pattern may not be coalesced")
        
        # Check for shared memory usage
        if "__shared__" not in kernel_code:
            warnings.append("No shared memory usage - may miss optimization opportunities")
        
        # Check for proper memory allocation
        if "malloc" in kernel_code or "new" in kernel_code:
            errors.append("Kernel should not use dynamic memory allocation")
        
        # Check for proper synchronization
        if "__syncthreads" not in kernel_code and "shared" in kernel_code:
            warnings.append("Shared memory used without synchronization")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def _validate_threading(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate threading and parallelization patterns."""
        errors = []
        warnings = []
        
        # Check for proper thread indexing
        if "threadIdx" not in kernel_code:
            errors.append("No thread indexing found")
        
        if "blockIdx" not in kernel_code:
            errors.append("No block indexing found")
        
        # Check for proper grid and block dimensions
        if "blockDim" not in kernel_code:
            warnings.append("No block dimension usage found")
        
        if "gridDim" not in kernel_code:
            warnings.append("No grid dimension usage found")
        
        # Check for proper bounds checking
        bounds_patterns = [
            r"if\s*\(\s*.*<\s*.*\s*\)",
            r"if\s*\(\s*.*<=\s*.*\s*\)",
        ]
        
        has_bounds_check = any(re.search(pattern, kernel_code) for pattern in bounds_patterns)
        if not has_bounds_check:
            warnings.append("No bounds checking found - may cause out-of-bounds access")
        
        # Check for proper thread divergence
        if kernel_code.count("if") > 5:
            warnings.append("Many conditional statements may cause thread divergence")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }
    
    def validate_kernel_configuration(self, kernel_code: str, 
                                    operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate kernel launch configuration.
        
        Args:
            kernel_code: CUDA kernel code
            operation_info: Information about the operation
            
        Returns:
            Dictionary containing configuration validation results
        """
        errors = []
        warnings = []
        
        tensor_info = operation_info.get("tensor_info", {})
        shape = tensor_info.get("shape", [])
        numel = tensor_info.get("numel", 0)
        
        # Extract block size from comments
        block_size_match = re.search(r"Block size:\s*(\d+)", kernel_code)
        if block_size_match:
            block_size = int(block_size_match.group(1))
            
            # Check if block size is reasonable
            if block_size > 1024:
                errors.append(f"Block size {block_size} exceeds maximum of 1024")
            elif block_size < 32:
                warnings.append(f"Block size {block_size} may be too small for efficiency")
        
        # Extract grid size from comments
        grid_size_match = re.search(r"Grid size:\s*(\d+)", kernel_code)
        if grid_size_match and numel > 0:
            grid_size = int(grid_size_match.group(1))
            estimated_grid_size = (numel + 255) // 256  # Assuming 256 threads per block
            
            if grid_size < estimated_grid_size:
                warnings.append(f"Grid size {grid_size} may be too small for tensor with {numel} elements")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommended_block_size": min(256, max(32, numel // 1000)),
            "recommended_grid_size": max(1, (numel + 255) // 256),
        }
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of validation results.
        
        Args:
            validation_result: Validation result dictionary
            
        Returns:
            Summary string
        """
        if validation_result["is_valid"]:
            summary = "✅ Kernel validation passed"
        else:
            summary = "❌ Kernel validation failed"
        
        if validation_result["errors"]:
            summary += f"\nErrors: {len(validation_result['errors'])}"
            for error in validation_result["errors"][:3]:  # Show first 3 errors
                summary += f"\n  - {error}"
        
        if validation_result["warnings"]:
            summary += f"\nWarnings: {len(validation_result['warnings'])}"
            for warning in validation_result["warnings"][:3]:  # Show first 3 warnings
                summary += f"\n  - {warning}"
        
        return summary 