"""
Performance Advisor Agent

This module provides intelligent performance analysis and recommendations
for CUDA kernel optimization and system-level performance tuning.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceRecommendation:
    """Performance optimization recommendation."""
    category: str  # memory, compute, parallelization, etc.
    title: str
    description: str
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    expected_improvement: float
    implementation_steps: List[str]
    code_examples: Optional[List[str]] = None

    def __post_init__(self):
        if self.code_examples is None:
            self.code_examples = []

@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis."""
    operation_type: str
    current_performance: Dict[str, Any]
    bottlenecks: List[str]
    recommendations: List[PerformanceRecommendation]
    optimization_potential: float
    priority_actions: List[str]

class PerformanceAdvisorAgent:
    """
    Intelligent performance advisor for CUDA optimization.
    
    This agent provides:
    - Performance bottleneck analysis
    - Optimization recommendations
    - Code improvement suggestions
    - System-level performance insights
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.analysis_history = []
        self.recommendation_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize recommendation templates for different operation types."""
        return {
            "matmul": {
                "memory_optimizations": [
                    "Shared memory tiling",
                    "Coalesced memory access",
                    "Memory layout optimization"
                ],
                "compute_optimizations": [
                    "Loop unrolling",
                    "Register tiling",
                    "Instruction-level parallelism"
                ],
                "parallelization_optimizations": [
                    "2D thread block optimization",
                    "Occupancy maximization",
                    "Warp-level optimizations"
                ]
            },
            "conv2d": {
                "memory_optimizations": [
                    "Input tiling with shared memory",
                    "Weight caching",
                    "Memory access pattern optimization"
                ],
                "compute_optimizations": [
                    "Loop unrolling",
                    "Register blocking",
                    "FMA instruction utilization"
                ],
                "parallelization_optimizations": [
                    "3D thread block optimization",
                    "Channel-level parallelism",
                    "Spatial parallelism"
                ]
            },
            "elementwise": {
                "memory_optimizations": [
                    "Coalesced memory access",
                    "Vectorized loads/stores",
                    "Memory bandwidth optimization"
                ],
                "compute_optimizations": [
                    "Loop unrolling",
                    "Instruction pipelining",
                    "Branch optimization"
                ],
                "parallelization_optimizations": [
                    "1D thread block optimization",
                    "Warp-level parallelism",
                    "Occupancy optimization"
                ]
            }
        }
    
    def analyze_performance(self, operation_info: Dict[str, Any], 
                          current_kernel: Optional[str] = None,
                          performance_metrics: Optional[Dict[str, Any]] = None) -> PerformanceAnalysis:
        """
        Perform comprehensive performance analysis.
        
        Args:
            operation_info: Information about the operation
            current_kernel: Current kernel code (optional)
            performance_metrics: Current performance metrics (optional)
            
        Returns:
            Comprehensive performance analysis
        """
        try:
            logger.info(f"Starting performance analysis for {operation_info.get('operation_type', 'unknown')}")
            
            operation_type = operation_info.get('operation_type', 'unknown')
            
            # Analyze current performance
            current_performance = self._analyze_current_performance(operation_info, performance_metrics)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(operation_info, current_kernel, current_performance)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(operation_info, bottlenecks, current_performance)
            
            # Calculate optimization potential
            optimization_potential = self._calculate_optimization_potential(recommendations)
            
            # Prioritize actions
            priority_actions = self._prioritize_actions(recommendations)
            
            analysis = PerformanceAnalysis(
                operation_type=operation_type,
                current_performance=current_performance,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                optimization_potential=optimization_potential,
                priority_actions=priority_actions
            )
            
            # Store in history
            self.analysis_history.append({
                'operation_info': operation_info,
                'analysis': analysis,
                'timestamp': time.time()
            })
            
            logger.info(f"Performance analysis completed. Optimization potential: {optimization_potential:.2f}x")
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return self._create_error_analysis(operation_info, str(e))
    
    def _analyze_current_performance(self, operation_info: Dict[str, Any], 
                                   performance_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current performance characteristics."""
        analysis = {
            "compute_efficiency": 0.0,
            "memory_efficiency": 0.0,
            "parallelization_efficiency": 0.0,
            "occupancy": 0.0,
            "memory_bandwidth_utilization": 0.0,
            "compute_intensity": 0.0
        }
        
        operation_type = operation_info.get('operation_type', 'unknown')
        tensor_info = operation_info.get('tensor_info', {})
        
        # Estimate compute efficiency based on operation type
        if operation_type == "matmul":
            analysis["compute_efficiency"] = 0.7  # Typically good for matmul
            analysis["memory_efficiency"] = 0.6   # Can be improved with tiling
            analysis["compute_intensity"] = 2.0   # High compute intensity
        elif operation_type == "conv2d":
            analysis["compute_efficiency"] = 0.6  # Moderate
            analysis["memory_efficiency"] = 0.5   # Can be improved
            analysis["compute_intensity"] = 1.5   # Medium compute intensity
        elif operation_type in ["add", "mul", "relu"]:
            analysis["compute_efficiency"] = 0.8  # High for simple ops
            analysis["memory_efficiency"] = 0.7   # Good for elementwise
            analysis["compute_intensity"] = 0.5   # Low compute intensity
        
        # Estimate parallelization efficiency
        analysis["parallelization_efficiency"] = 0.75  # Assume good parallelization
        
        # Estimate occupancy (simplified)
        analysis["occupancy"] = 0.6  # Assume moderate occupancy
        
        # Estimate memory bandwidth utilization
        analysis["memory_bandwidth_utilization"] = 0.5  # Assume moderate utilization
        
        # Override with actual metrics if available
        if performance_metrics:
            analysis.update(performance_metrics)
        
        return analysis
    
    def _identify_bottlenecks(self, operation_info: Dict[str, Any], 
                            current_kernel: Optional[str], 
                            current_performance: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        operation_type = operation_info.get('operation_type', 'unknown')
        
        # Analyze performance metrics
        if current_performance.get('memory_efficiency', 1.0) < 0.6:
            bottlenecks.append("Memory access inefficiency")
        
        if current_performance.get('compute_efficiency', 1.0) < 0.6:
            bottlenecks.append("Low compute utilization")
        
        if current_performance.get('occupancy', 1.0) < 0.5:
            bottlenecks.append("Low GPU occupancy")
        
        if current_performance.get('memory_bandwidth_utilization', 1.0) < 0.4:
            bottlenecks.append("Underutilized memory bandwidth")
        
        # Operation-specific bottlenecks
        if operation_type == "matmul":
            if current_performance.get('memory_efficiency', 1.0) < 0.7:
                bottlenecks.append("Missing shared memory tiling")
        
        elif operation_type == "conv2d":
            if current_performance.get('compute_efficiency', 1.0) < 0.7:
                bottlenecks.append("Inefficient convolution implementation")
        
        elif operation_type in ["add", "mul", "relu"]:
            if current_performance.get('parallelization_efficiency', 1.0) < 0.8:
                bottlenecks.append("Suboptimal thread block configuration")
        
        # Kernel-specific bottlenecks (if available)
        if current_kernel:
            kernel_bottlenecks = self._analyze_kernel_bottlenecks(current_kernel)
            bottlenecks.extend(kernel_bottlenecks)
        
        return bottlenecks
    
    def _analyze_kernel_bottlenecks(self, kernel_code: str) -> List[str]:
        """Analyze kernel code for specific bottlenecks."""
        bottlenecks = []
        
        # Check for common issues
        if "__shared__" not in kernel_code:
            bottlenecks.append("No shared memory usage")
        
        if kernel_code.count("__syncthreads") > 3:
            bottlenecks.append("Excessive synchronization")
        
        if "if" in kernel_code and kernel_code.count("if") > 5:
            bottlenecks.append("High conditional branching")
        
        if kernel_code.count("for") > 3:
            bottlenecks.append("Multiple nested loops without unrolling")
        
        return bottlenecks
    
    def _generate_recommendations(self, operation_info: Dict[str, Any], 
                                bottlenecks: List[str], 
                                current_performance: Dict[str, Any]) -> List[PerformanceRecommendation]:
        """Generate performance optimization recommendations."""
        recommendations = []
        operation_type = operation_info.get('operation_type', 'unknown')
        
        # Get templates for this operation type
        templates = self.recommendation_templates.get(operation_type, 
                                                     self.recommendation_templates.get("elementwise", {}))
        
        # Memory optimizations
        if "Memory access inefficiency" in bottlenecks or "Missing shared memory tiling" in bottlenecks:
            for opt in templates.get("memory_optimizations", []):
                recommendations.append(self._create_memory_recommendation(opt, operation_type))
        
        # Compute optimizations
        if "Low compute utilization" in bottlenecks or "Inefficient convolution implementation" in bottlenecks:
            for opt in templates.get("compute_optimizations", []):
                recommendations.append(self._create_compute_recommendation(opt, operation_type))
        
        # Parallelization optimizations
        if "Low GPU occupancy" in bottlenecks or "Suboptimal thread block configuration" in bottlenecks:
            for opt in templates.get("parallelization_optimizations", []):
                recommendations.append(self._create_parallelization_recommendation(opt, operation_type))
        
        # General recommendations
        if "Underutilized memory bandwidth" in bottlenecks:
            recommendations.append(self._create_bandwidth_recommendation(operation_type))
        
        if "Excessive synchronization" in bottlenecks:
            recommendations.append(self._create_synchronization_recommendation())
        
        if "High conditional branching" in bottlenecks:
            recommendations.append(self._create_branching_recommendation())
        
        return recommendations
    
    def _create_memory_recommendation(self, optimization: str, operation_type: str) -> PerformanceRecommendation:
        """Create memory optimization recommendation."""
        if optimization == "Shared memory tiling":
            return PerformanceRecommendation(
                category="memory",
                title="Implement Shared Memory Tiling",
                description="Use shared memory to cache frequently accessed data and reduce global memory accesses",
                impact="high",
                effort="medium",
                expected_improvement=2.0,
                implementation_steps=[
                    "Identify data reuse patterns",
                    "Design tiling strategy",
                    "Implement shared memory loading",
                    "Add synchronization points",
                    "Update memory access patterns"
                ],
                code_examples=[
                    "__shared__ float tile_A[TILE_SIZE][TILE_SIZE];",
                    "__shared__ float tile_B[TILE_SIZE][TILE_SIZE];"
                ]
            )
        elif optimization == "Coalesced memory access":
            return PerformanceRecommendation(
                category="memory",
                title="Optimize for Coalesced Memory Access",
                description="Ensure consecutive threads access consecutive memory locations",
                impact="high",
                effort="low",
                expected_improvement=1.5,
                implementation_steps=[
                    "Analyze memory access patterns",
                    "Restructure loops for coalescing",
                    "Use appropriate thread block sizes",
                    "Consider memory layout changes"
                ]
            )
        else:
            return PerformanceRecommendation(
                category="memory",
                title=f"Implement {optimization}",
                description=f"Apply {optimization} to improve memory performance",
                impact="medium",
                effort="medium",
                expected_improvement=1.3,
                implementation_steps=[
                    f"Research {optimization} techniques",
                    "Implement optimization",
                    "Test performance impact"
                ]
            )
    
    def _create_compute_recommendation(self, optimization: str, operation_type: str) -> PerformanceRecommendation:
        """Create compute optimization recommendation."""
        if optimization == "Loop unrolling":
            return PerformanceRecommendation(
                category="compute",
                title="Implement Loop Unrolling",
                description="Unroll loops to reduce loop overhead and improve instruction-level parallelism",
                impact="medium",
                effort="low",
                expected_improvement=1.2,
                implementation_steps=[
                    "Identify unrollable loops",
                    "Choose appropriate unroll factor",
                    "Implement unrolled version",
                    "Balance code size vs performance"
                ]
            )
        else:
            return PerformanceRecommendation(
                category="compute",
                title=f"Implement {optimization}",
                description=f"Apply {optimization} to improve compute efficiency",
                impact="medium",
                effort="medium",
                expected_improvement=1.3,
                implementation_steps=[
                    f"Research {optimization} techniques",
                    "Implement optimization",
                    "Test performance impact"
                ]
            )
    
    def _create_parallelization_recommendation(self, optimization: str, operation_type: str) -> PerformanceRecommendation:
        """Create parallelization optimization recommendation."""
        if optimization == "2D thread block optimization":
            return PerformanceRecommendation(
                category="parallelization",
                title="Optimize Thread Block Configuration",
                description="Use 2D thread blocks to better match data layout and improve occupancy",
                impact="high",
                effort="low",
                expected_improvement=1.4,
                implementation_steps=[
                    "Calculate optimal block dimensions",
                    "Update kernel launch parameters",
                    "Adjust thread indexing",
                    "Test different configurations"
                ]
            )
        else:
            return PerformanceRecommendation(
                category="parallelization",
                title=f"Implement {optimization}",
                description=f"Apply {optimization} to improve parallelization efficiency",
                impact="medium",
                effort="medium",
                expected_improvement=1.3,
                implementation_steps=[
                    f"Research {optimization} techniques",
                    "Implement optimization",
                    "Test performance impact"
                ]
            )
    
    def _create_bandwidth_recommendation(self, operation_type: str) -> PerformanceRecommendation:
        """Create memory bandwidth optimization recommendation."""
        return PerformanceRecommendation(
            category="memory",
            title="Optimize Memory Bandwidth Utilization",
            description="Increase memory bandwidth utilization through better access patterns and vectorization",
            impact="high",
            effort="medium",
            expected_improvement=1.6,
            implementation_steps=[
                "Use vectorized loads/stores",
                "Optimize memory access patterns",
                "Consider memory coalescing",
                "Profile memory bandwidth usage"
            ]
        )
    
    def _create_synchronization_recommendation(self) -> PerformanceRecommendation:
        """Create synchronization optimization recommendation."""
        return PerformanceRecommendation(
            category="parallelization",
            title="Reduce Synchronization Overhead",
            description="Minimize synchronization points to reduce thread waiting time",
            impact="medium",
            effort="high",
            expected_improvement=1.3,
            implementation_steps=[
                "Analyze synchronization requirements",
                "Identify unnecessary sync points",
                "Restructure algorithm if possible",
                "Use warp-level primitives where appropriate"
            ]
        )
    
    def _create_branching_recommendation(self) -> PerformanceRecommendation:
        """Create branching optimization recommendation."""
        return PerformanceRecommendation(
            category="compute",
            title="Optimize Conditional Branching",
            description="Reduce warp divergence by minimizing conditional branching",
            impact="medium",
            effort="medium",
            expected_improvement=1.2,
            implementation_steps=[
                "Identify divergent branches",
                "Use predication where possible",
                "Restructure conditionals",
                "Consider branch-free alternatives"
            ]
        )
    
    def _calculate_optimization_potential(self, recommendations: List[PerformanceRecommendation]) -> float:
        """Calculate overall optimization potential."""
        if not recommendations:
            return 1.0
        
        # Calculate cumulative improvement potential
        total_improvement = 1.0
        for rec in recommendations:
            if rec.impact == "high":
                total_improvement *= rec.expected_improvement
            elif rec.impact == "medium":
                total_improvement *= (1.0 + (rec.expected_improvement - 1.0) * 0.7)
            else:  # low
                total_improvement *= (1.0 + (rec.expected_improvement - 1.0) * 0.3)
        
        return total_improvement
    
    def _prioritize_actions(self, recommendations: List[PerformanceRecommendation]) -> List[str]:
        """Prioritize optimization actions."""
        # Sort by impact and effort
        sorted_recs = sorted(recommendations, 
                           key=lambda r: (r.impact == "high", r.effort == "low"), 
                           reverse=True)
        
        priority_actions = []
        for rec in sorted_recs[:5]:  # Top 5 recommendations
            priority_actions.append(f"{rec.title} (Impact: {rec.impact}, Effort: {rec.effort})")
        
        return priority_actions
    
    def _create_error_analysis(self, operation_info: Dict[str, Any], error_message: str) -> PerformanceAnalysis:
        """Create error analysis when analysis fails."""
        return PerformanceAnalysis(
            operation_type=operation_info.get('operation_type', 'unknown'),
            current_performance={},
            bottlenecks=[f"Analysis failed: {error_message}"],
            recommendations=[],
            optimization_potential=1.0,
            priority_actions=["Fix analysis error and retry"]
        )
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get performance analysis history."""
        return self.analysis_history.copy()
    
    def clear_history(self):
        """Clear performance analysis history."""
        self.analysis_history.clear()
    
    def generate_performance_report(self, analysis: PerformanceAnalysis) -> str:
        """Generate a human-readable performance report."""
        report = f"""
Performance Analysis Report
==========================
Operation Type: {analysis.operation_type}

Current Performance Metrics:
- Compute Efficiency: {analysis.current_performance.get('compute_efficiency', 0):.2f}
- Memory Efficiency: {analysis.current_performance.get('memory_efficiency', 0):.2f}
- Parallelization Efficiency: {analysis.current_performance.get('parallelization_efficiency', 0):.2f}
- GPU Occupancy: {analysis.current_performance.get('occupancy', 0):.2f}

Identified Bottlenecks:
{chr(10).join(f"- {bottleneck}" for bottleneck in analysis.bottlenecks)}

Optimization Potential: {analysis.optimization_potential:.2f}x speedup

Top Recommendations:
{chr(10).join(f"{i+1}. {action}" for i, action in enumerate(analysis.priority_actions))}

Detailed Recommendations:
"""
        
        for i, rec in enumerate(analysis.recommendations, 1):
            report += f"""
{i}. {rec.title}
   Category: {rec.category}
   Impact: {rec.impact}, Effort: {rec.effort}
   Expected Improvement: {rec.expected_improvement:.2f}x
   Description: {rec.description}
   
   Implementation Steps:
   {chr(10).join(f"   {j+1}. {step}" for j, step in enumerate(rec.implementation_steps))}
"""
        
        return report 