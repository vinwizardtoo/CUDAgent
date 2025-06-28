"""
Kernel Validation Agent

This module provides kernel validation capabilities including syntax checking,
semantic analysis, and performance validation for generated CUDA kernels.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationIssue:
    """Validation issue details."""
    level: ValidationResult
    category: str
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    severity: int = 1  # 1-5, higher is more severe

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    kernel_code: str
    operation_type: str
    validation_level: ValidationLevel
    overall_result: ValidationResult
    issues: List[ValidationIssue]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    validation_time: float
    is_compilable: bool
    is_optimized: bool

class KernelValidationAgent:
    """
    CUDA kernel validation agent.
    
    This agent validates generated kernels for:
    - Syntax correctness
    - Semantic validity
    - Performance characteristics
    - Optimization quality
    - Best practices compliance
    """
    
    def __init__(self):
        self.validation_history = []
        self.validation_patterns = self._initialize_patterns()
        self.best_practices = self._initialize_best_practices()
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize validation patterns."""
        return {
            "syntax_patterns": [
                r"__global__\s+void\s+\w+\s*\(",
                r"__shared__\s+\w+\s+\w+",
                r"__syncthreads\s*\(\s*\)",
                r"threadIdx\.\w+",
                r"blockIdx\.\w+",
                r"blockDim\.\w+",
                r"gridDim\.\w+"
            ],
            "error_patterns": [
                r"int\s+\w+\s*\[\s*\]",  # Variable length arrays
                r"goto\s+\w+",  # Goto statements
                r"recursion",  # Recursion
                r"virtual\s+\w+",  # Virtual functions
                r"try\s*\{",  # Exception handling
                r"catch\s*\("
            ],
            "performance_patterns": [
                r"__shared__",  # Shared memory usage
                r"#pragma\s+unroll",  # Loop unrolling
                r"__ldg\s*\(",  # Read-only cache
                r"__fmaf_rn\s*\(",  # FMA instructions
                r"__shfl_",  # Warp shuffle
                r"atomicAdd\s*\("  # Atomic operations
            ]
        }
    
    def _initialize_best_practices(self) -> Dict[str, List[str]]:
        """Initialize best practices guidelines."""
        return {
            "memory_access": [
                "Use coalesced memory access patterns",
                "Utilize shared memory for frequently accessed data",
                "Minimize global memory transactions",
                "Use appropriate memory hierarchy"
            ],
            "thread_organization": [
                "Use 2D or 3D thread blocks when appropriate",
                "Maximize GPU occupancy",
                "Avoid thread divergence",
                "Use appropriate block sizes"
            ],
            "computation": [
                "Use FMA instructions when possible",
                "Unroll loops for better instruction-level parallelism",
                "Minimize register usage",
                "Use appropriate data types"
            ],
            "synchronization": [
                "Minimize synchronization points",
                "Use warp-level primitives when possible",
                "Avoid unnecessary __syncthreads()",
                "Use appropriate synchronization scope"
            ]
        }
    
    def validate_kernel(self, kernel_code: str, operation_info: Dict[str, Any], 
                       validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> ValidationReport:
        """
        Validate a CUDA kernel comprehensively.
        
        Args:
            kernel_code: CUDA kernel code to validate
            operation_info: Information about the operation
            validation_level: Level of validation to perform
            
        Returns:
            Comprehensive validation report
        """
        try:
            logger.info(f"Starting kernel validation (level: {validation_level.value})")
            start_time = time.time()
            
            operation_type = operation_info.get('operation_type', 'unknown')
            
            # Perform validation checks
            issues = []
            
            # Basic syntax validation
            syntax_issues = self._validate_syntax(kernel_code)
            issues.extend(syntax_issues)
            
            # Semantic validation
            semantic_issues = self._validate_semantics(kernel_code, operation_info)
            issues.extend(semantic_issues)
            
            # Performance validation
            performance_issues = self._validate_performance(kernel_code, operation_info)
            issues.extend(performance_issues)
            
            # Best practices validation
            if validation_level in [ValidationLevel.INTERMEDIATE, ValidationLevel.ADVANCED]:
                practice_issues = self._validate_best_practices(kernel_code, operation_info)
                issues.extend(practice_issues)
            
            # Advanced validation
            if validation_level == ValidationLevel.ADVANCED:
                advanced_issues = self._validate_advanced(kernel_code, operation_info)
                issues.extend(advanced_issues)
            
            # Determine overall result
            overall_result = self._determine_overall_result(issues)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(kernel_code, operation_info)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues, operation_info)
            
            # Check if kernel is compilable and optimized
            is_compilable = not any(issue.level == ValidationResult.ERROR or issue.level == ValidationResult.CRITICAL 
                                  for issue in issues)
            is_optimized = self._check_optimization_quality(kernel_code, operation_info)
            
            validation_time = time.time() - start_time
            
            report = ValidationReport(
                kernel_code=kernel_code,
                operation_type=operation_type,
                validation_level=validation_level,
                overall_result=overall_result,
                issues=issues,
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                validation_time=validation_time,
                is_compilable=is_compilable,
                is_optimized=is_optimized
            )
            
            # Store in history
            self.validation_history.append({
                'operation_info': operation_info,
                'report': report,
                'timestamp': time.time()
            })
            
            logger.info(f"Kernel validation completed in {validation_time:.2f}s. Result: {overall_result.value}")
            return report
            
        except Exception as e:
            logger.error(f"Kernel validation failed: {str(e)}")
            return self._create_error_report(kernel_code, operation_info, str(e))
    
    def _validate_syntax(self, kernel_code: str) -> List[ValidationIssue]:
        """Validate kernel syntax."""
        issues = []
        lines = kernel_code.split('\n')
        
        # Check for required CUDA syntax
        if not re.search(r"__global__\s+void\s+\w+\s*\(", kernel_code):
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                category="syntax",
                message="Missing __global__ function declaration",
                suggestion="Add __global__ void kernel_name(...) declaration"
            ))
        
        # Check for basic CUDA constructs
        required_patterns = [
            (r"threadIdx\.\w+", "threadIdx usage"),
            (r"blockIdx\.\w+", "blockIdx usage"),
            (r"blockDim\.\w+", "blockDim usage")
        ]
        
        for pattern, description in required_patterns:
            if not re.search(pattern, kernel_code):
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    category="syntax",
                    message=f"Missing {description}",
                    suggestion=f"Use {description} for proper thread indexing"
                ))
        
        # Check for syntax errors
        error_patterns = [
            (r"int\s+\w+\s*\[\s*\]", "Variable length arrays not supported in CUDA"),
            (r"goto\s+\w+", "Goto statements not supported in CUDA"),
            (r"virtual\s+\w+", "Virtual functions not supported in CUDA"),
            (r"try\s*\{", "Exception handling not supported in CUDA"),
            (r"catch\s*\(", "Exception handling not supported in CUDA")
        ]
        
        for pattern, message in error_patterns:
            if re.search(pattern, kernel_code):
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    category="syntax",
                    message=message,
                    suggestion="Remove unsupported CUDA constructs"
                ))
        
        return issues
    
    def _validate_semantics(self, kernel_code: str, operation_info: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate kernel semantics."""
        issues = []
        operation_type = operation_info.get('operation_type', 'unknown')
        
        # Check for operation-specific requirements
        if operation_type == "matmul":
            issues.extend(self._validate_matmul_semantics(kernel_code))
        elif operation_type == "conv2d":
            issues.extend(self._validate_conv2d_semantics(kernel_code))
        elif operation_type in ["add", "mul", "relu"]:
            issues.extend(self._validate_elementwise_semantics(kernel_code))
        
        # Check for common semantic issues
        if "__syncthreads" in kernel_code and "__shared__" not in kernel_code:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="semantics",
                message="__syncthreads() used without shared memory",
                suggestion="Remove unnecessary synchronization or add shared memory usage"
            ))
        
        # Check for potential race conditions
        if kernel_code.count("atomicAdd") > 0 and kernel_code.count("__syncthreads") == 0:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="semantics",
                message="Atomic operations without synchronization",
                suggestion="Consider adding synchronization if needed"
            ))
        
        return issues
    
    def _validate_matmul_semantics(self, kernel_code: str) -> List[ValidationIssue]:
        """Validate matrix multiplication semantics."""
        issues = []
        
        # Check for proper matrix multiplication structure
        if "for" not in kernel_code and "*" not in kernel_code:
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                category="semantics",
                message="Missing matrix multiplication loop structure",
                suggestion="Implement proper matrix multiplication with nested loops"
            ))
        
        # Check for shared memory usage (recommended for matmul)
        if "__shared__" not in kernel_code:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="semantics",
                message="Matrix multiplication without shared memory tiling",
                suggestion="Consider using shared memory for better performance"
            ))
        
        return issues
    
    def _validate_conv2d_semantics(self, kernel_code: str) -> List[ValidationIssue]:
        """Validate 2D convolution semantics."""
        issues = []
        
        # Check for convolution-specific patterns
        if "kernel" not in kernel_code.lower() and "filter" not in kernel_code.lower():
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="semantics",
                message="Missing convolution-specific patterns",
                suggestion="Ensure proper convolution implementation with kernel/filter handling"
            ))
        
        return issues
    
    def _validate_elementwise_semantics(self, kernel_code: str) -> List[ValidationIssue]:
        """Validate elementwise operation semantics."""
        issues = []
        
        # Check for simple elementwise structure
        if kernel_code.count("for") > 2:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="semantics",
                message="Complex loop structure for elementwise operation",
                suggestion="Simplify to single loop for elementwise operations"
            ))
        
        return issues
    
    def _validate_performance(self, kernel_code: str, operation_info: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate performance characteristics."""
        issues = []
        
        # Check for performance optimizations
        performance_patterns = [
            (r"__shared__", "Shared memory usage", ValidationResult.PASS),
            (r"#pragma\s+unroll", "Loop unrolling", ValidationResult.PASS),
            (r"__ldg\s*\(", "Read-only cache usage", ValidationResult.PASS),
            (r"__fmaf_rn\s*\(", "FMA instructions", ValidationResult.PASS)
        ]
        
        for pattern, description, level in performance_patterns:
            if re.search(pattern, kernel_code):
                issues.append(ValidationIssue(
                    level=level,
                    category="performance",
                    message=f"Good: {description} detected",
                    suggestion="Continue using this optimization"
                ))
        
        # Check for performance issues
        if kernel_code.count("if") > kernel_code.count("*") / 2:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="performance",
                message="High conditional branching may cause warp divergence",
                suggestion="Consider using predication or restructuring conditionals"
            ))
        
        if kernel_code.count("__syncthreads") > 3:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="performance",
                message="Excessive synchronization points",
                suggestion="Review synchronization requirements"
            ))
        
        return issues
    
    def _validate_best_practices(self, kernel_code: str, operation_info: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate best practices compliance."""
        issues = []
        
        # Check memory access patterns
        if "threadIdx.x" in kernel_code and "blockIdx.x" in kernel_code:
            # Check for potential coalescing
            if "[" in kernel_code and "threadIdx.x" in kernel_code:
                issues.append(ValidationIssue(
                    level=ValidationResult.PASS,
                    category="best_practices",
                    message="Good: Potential for coalesced memory access",
                    suggestion="Ensure consecutive threads access consecutive memory"
                ))
        
        # Check thread organization
        if "threadIdx.y" in kernel_code:
            issues.append(ValidationIssue(
                level=ValidationResult.PASS,
                category="best_practices",
                message="Good: Using 2D thread organization",
                suggestion="Consider 3D organization for 3D operations"
            ))
        
        # Check for register usage optimization
        if kernel_code.count("float") > kernel_code.count("int") * 2:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                category="best_practices",
                message="High register usage with float variables",
                suggestion="Consider using __restrict__ or reducing variable scope"
            ))
        
        return issues
    
    def _validate_advanced(self, kernel_code: str, operation_info: Dict[str, Any]) -> List[ValidationIssue]:
        """Perform advanced validation checks."""
        issues = []
        
        # Check for advanced optimizations
        if "__shfl_" in kernel_code:
            issues.append(ValidationIssue(
                level=ValidationResult.PASS,
                category="advanced",
                message="Good: Using warp shuffle operations",
                suggestion="Consider other warp-level primitives"
            ))
        
        # Check for potential vectorization
        if "float4" in kernel_code or "int4" in kernel_code:
            issues.append(ValidationIssue(
                level=ValidationResult.PASS,
                category="advanced",
                message="Good: Using vectorized data types",
                suggestion="Consider other vectorization opportunities"
            ))
        
        return issues
    
    def _determine_overall_result(self, issues: List[ValidationIssue]) -> ValidationResult:
        """Determine overall validation result."""
        if any(issue.level == ValidationResult.CRITICAL for issue in issues):
            return ValidationResult.CRITICAL
        elif any(issue.level == ValidationResult.ERROR for issue in issues):
            return ValidationResult.ERROR
        elif any(issue.level == ValidationResult.WARNING for issue in issues):
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASS
    
    def _calculate_performance_metrics(self, kernel_code: str, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the kernel."""
        metrics = {
            "shared_memory_usage": "__shared__" in kernel_code,
            "loop_unrolling": "#pragma unroll" in kernel_code,
            "fma_instructions": "__fmaf_rn" in kernel_code,
            "warp_shuffle": "__shfl_" in kernel_code,
            "vectorization": "float4" in kernel_code or "int4" in kernel_code,
            "synchronization_points": kernel_code.count("__syncthreads"),
            "conditional_branches": kernel_code.count("if"),
            "loop_count": kernel_code.count("for"),
            "optimization_score": 0.0
        }
        
        # Calculate optimization score
        score = 0.0
        if metrics["shared_memory_usage"]:
            score += 0.2
        if metrics["loop_unrolling"]:
            score += 0.15
        if metrics["fma_instructions"]:
            score += 0.15
        if metrics["warp_shuffle"]:
            score += 0.1
        if metrics["vectorization"]:
            score += 0.1
        
        # Penalize excessive synchronization and branching
        if metrics["synchronization_points"] > 3:
            score -= 0.1
        if metrics["conditional_branches"] > 5:
            score -= 0.1
        
        metrics["optimization_score"] = max(0.0, min(1.0, score))
        
        return metrics
    
    def _generate_recommendations(self, issues: List[ValidationIssue], 
                                operation_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        # Group issues by category
        error_issues = [issue for issue in issues if issue.level in [ValidationResult.ERROR, ValidationResult.CRITICAL]]
        warning_issues = [issue for issue in issues if issue.level == ValidationResult.WARNING]
        
        # Prioritize error fixes
        for issue in error_issues:
            if issue.suggestion:
                recommendations.append(f"Fix: {issue.suggestion}")
        
        # Add warning improvements
        for issue in warning_issues:
            if issue.suggestion:
                recommendations.append(f"Improve: {issue.suggestion}")
        
        # Add general recommendations
        operation_type = operation_info.get('operation_type', 'unknown')
        if operation_type == "matmul" and "__shared__" not in operation_info.get('kernel_code', ''):
            recommendations.append("Consider implementing shared memory tiling for matrix multiplication")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _check_optimization_quality(self, kernel_code: str, operation_info: Dict[str, Any]) -> bool:
        """Check if the kernel shows good optimization quality."""
        optimization_indicators = [
            "__shared__" in kernel_code,
            "#pragma unroll" in kernel_code,
            "__fmaf_rn" in kernel_code,
            "threadIdx.y" in kernel_code,  # 2D thread organization
            kernel_code.count("__syncthreads") <= 3,  # Reasonable synchronization
            kernel_code.count("if") <= 5  # Reasonable branching
        ]
        
        # Consider optimized if majority of indicators are positive
        return sum(optimization_indicators) >= 3
    
    def _create_error_report(self, kernel_code: str, operation_info: Dict[str, Any], 
                           error_message: str) -> ValidationReport:
        """Create error report when validation fails."""
        return ValidationReport(
            kernel_code=kernel_code,
            operation_type=operation_info.get('operation_type', 'unknown'),
            validation_level=ValidationLevel.BASIC,
            overall_result=ValidationResult.ERROR,
            issues=[ValidationIssue(
                level=ValidationResult.ERROR,
                category="validation",
                message=f"Validation failed: {error_message}"
            )],
            performance_metrics={},
            recommendations=["Fix validation error and retry"],
            validation_time=0.0,
            is_compilable=False,
            is_optimized=False
        )
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def clear_history(self):
        """Clear validation history."""
        self.validation_history.clear()
    
    def generate_validation_summary(self, report: ValidationReport) -> str:
        """Generate a human-readable validation summary."""
        summary = f"""
Kernel Validation Summary
========================
Operation Type: {report.operation_type}
Validation Level: {report.validation_level.value}
Overall Result: {report.overall_result.value}
Compilable: {report.is_compilable}
Optimized: {report.is_optimized}
Validation Time: {report.validation_time:.2f}s

Performance Metrics:
- Shared Memory Usage: {report.performance_metrics.get('shared_memory_usage', False)}
- Loop Unrolling: {report.performance_metrics.get('loop_unrolling', False)}
- FMA Instructions: {report.performance_metrics.get('fma_instructions', False)}
- Optimization Score: {report.performance_metrics.get('optimization_score', 0):.2f}

Issues Found: {len(report.issues)}
- Errors: {len([i for i in report.issues if i.level in [ValidationResult.ERROR, ValidationResult.CRITICAL]])}
- Warnings: {len([i for i in report.issues if i.level == ValidationResult.WARNING])}
- Passes: {len([i for i in report.issues if i.level == ValidationResult.PASS])}

Top Recommendations:
{chr(10).join(f"- {rec}" for rec in report.recommendations[:3])}
"""
        return summary

# Add backward compatibility alias
ValidationAgent = KernelValidationAgent 