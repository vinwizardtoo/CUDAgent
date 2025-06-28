"""
CUDAgent AI Agent Framework

This module provides AI-powered optimization agents for CUDA kernel generation
and performance tuning using Large Language Models (LLMs).
"""

from .config_manager import ConfigManager, LLMConfig
from .llm_agent import LLMOptimizationAgent, OptimizationRequest, OptimizationResponse, LLMAgent
from .kernel_optimizer import KernelOptimizationAgent, KernelOptimizer
from .performance_advisor import PerformanceAdvisorAgent, PerformanceAdvisor
from .validation_agent import KernelValidationAgent, ValidationAgent

__all__ = [
    'ConfigManager',
    'LLMConfig',
    'LLMOptimizationAgent',
    'LLMAgent',  # Backward compatibility alias
    'OptimizationRequest',
    'OptimizationResponse',
    'KernelOptimizationAgent',
    'KernelOptimizer',  # Backward compatibility alias
    'PerformanceAdvisorAgent',
    'PerformanceAdvisor',  # Backward compatibility alias
    'KernelValidationAgent',
    'ValidationAgent'  # Backward compatibility alias
] 