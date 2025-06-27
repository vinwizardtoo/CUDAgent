"""
LLM Optimization Agent

This module provides the core LLM integration for CUDA kernel optimization
using various language model providers (OpenAI, Anthropic, local models, etc.).
"""

import json
import logging
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from .config_manager import ConfigManager, LLMConfig

logger = logging.getLogger(__name__)

@dataclass
class OptimizationRequest:
    """Request for kernel optimization."""
    operation_type: str
    operation_info: Dict[str, Any]
    current_kernel: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    optimization_goals: List[str] = None
    constraints: Dict[str, Any] = None

@dataclass
class OptimizationResponse:
    """Response from LLM optimization."""
    optimized_kernel: str
    optimization_explanations: List[str]
    performance_predictions: Dict[str, Any]
    confidence_score: float
    suggested_parameters: Dict[str, Any]
    warnings: List[str] = None
    errors: List[str] = None
    provider_used: str = "unknown"
    generation_time: float = 0.0

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, config: LLMConfig) -> str:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def validate_response(self, response: str) -> bool:
        """Validate the response format."""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self):
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_response(self, prompt: str, config: LLMConfig) -> str:
        """Generate response using OpenAI API."""
        try:
            client = self.openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def validate_response(self, response: str) -> bool:
        """Basic validation of OpenAI response."""
        return bool(response and len(response.strip()) > 0)

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self):
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def generate_response(self, prompt: str, config: LLMConfig) -> str:
        """Generate response using Anthropic API."""
        try:
            client = self.anthropic.Anthropic(api_key=config.api_key)
            response = client.messages.create(
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def validate_response(self, response: str) -> bool:
        """Basic validation of Anthropic response."""
        return bool(response and len(response.strip()) > 0)

class LocalProvider(LLMProvider):
    """Local LLM provider (placeholder for future implementation)."""
    
    def __init__(self):
        logger.warning("Local LLM provider not yet implemented. Using mock responses.")
    
    def generate_response(self, prompt: str, config: LLMConfig) -> str:
        """Generate mock response for local provider."""
        # Mock response for development
        return self._generate_mock_response(prompt)
    
    def validate_response(self, response: str) -> bool:
        """Validate mock response."""
        return bool(response and len(response.strip()) > 0)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock CUDA kernel response."""
        if "matmul" in prompt.lower():
            return self._mock_matmul_kernel()
        elif "conv2d" in prompt.lower():
            return self._mock_conv2d_kernel()
        else:
            return self._mock_generic_kernel()
    
    def _mock_matmul_kernel(self) -> str:
        return '''
// Optimized Matrix Multiplication Kernel
__global__ void optimized_matmul_kernel(
    const float* A, const float* B, float* C,
    const int M, const int N, const int K
) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
'''
    
    def _mock_conv2d_kernel(self) -> str:
        return '''
// Optimized 2D Convolution Kernel
__global__ void optimized_conv2d_kernel(
    const float* input, const float* weight, float* output,
    const int batch_size, const int in_channels, const int out_channels,
    const int height, const int width, const int kernel_size,
    const int stride, const int padding
) {
    int out_h = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (out_h >= (height + 2 * padding - kernel_size) / stride + 1 ||
        out_w >= (width + 2 * padding - kernel_size) / stride + 1 ||
        out_c >= out_channels) {
        return;
    }
    
    float sum = 0.0f;
    
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int in_h = out_h * stride + kh - padding;
                int in_w = out_w * stride + kw - padding;
                
                if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {
                    float input_val = input[in_c * height * width + in_h * width + in_w];
                    float weight_val = weight[out_c * in_channels * kernel_size * kernel_size + 
                                            in_c * kernel_size * kernel_size + kh * kernel_size + kw];
                    sum += input_val * weight_val;
                }
            }
        }
    }
    
    output[out_c * ((height + 2 * padding - kernel_size) / stride + 1) * 
           ((width + 2 * padding - kernel_size) / stride + 1) + out_h * 
           ((width + 2 * padding - kernel_size) / stride + 1) + out_w] = sum;
}
'''
    
    def _mock_generic_kernel(self) -> str:
        return '''
// Generic Optimized Kernel
__global__ void optimized_kernel(
    const float* input, float* output, const int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Apply optimization strategy
        output[idx] = input[idx] * 2.0f;  // Example operation
    }
}
'''

class LLMOptimizationAgent:
    """
    LLM-based optimization agent for CUDA kernel generation and optimization.
    
    This agent uses Large Language Models to:
    - Generate optimized CUDA kernels
    - Provide optimization suggestions
    - Analyze performance bottlenecks
    - Suggest parameter tuning
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.providers = self._initialize_providers()
        self.optimization_history = []
        
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Initialize LLM providers."""
        providers = {}
        
        # Initialize available providers
        available_providers = self.config_manager.get_available_providers()
        
        if 'openai' in available_providers:
            try:
                providers['openai'] = OpenAIProvider()
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {str(e)}")
        
        if 'anthropic' in available_providers:
            try:
                providers['anthropic'] = AnthropicProvider()
                logger.info("Anthropic provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {str(e)}")
        
        # Local provider is always available
        providers['local'] = LocalProvider()
        logger.info("Local provider initialized")
        
        return providers
    
    def optimize_kernel(self, request: OptimizationRequest, 
                       preferred_provider: Optional[str] = None) -> OptimizationResponse:
        """
        Optimize a CUDA kernel using LLM.
        
        Args:
            request: Optimization request containing operation details
            preferred_provider: Preferred LLM provider (optional)
            
        Returns:
            OptimizationResponse with optimized kernel and analysis
        """
        provider_name = None  # Ensure always defined
        try:
            logger.info(f"Starting LLM optimization for {request.operation_type}")
            
            # Get best available provider (no arguments)
            provider_name = self.config_manager.get_best_provider()
            if not provider_name:
                return self._create_error_response("No LLM providers available")
            
            provider = self.providers.get(provider_name)
            if not provider:
                return self._create_error_response(f"Provider {provider_name} not initialized")
            
            # Get provider configuration
            provider_config = self.config_manager.get_provider_config(provider_name)
            if not provider_config:
                return self._create_error_response(f"Configuration not found for {provider_name}")
            
            # Generate optimization prompt
            prompt = self._generate_optimization_prompt(request)
            
            # Get LLM response
            start_time = time.time()
            response_text = provider.generate_response(prompt, provider_config)
            generation_time = time.time() - start_time
            
            # Parse and validate response
            parsed_response = self._parse_llm_response(response_text, request)
            parsed_response.provider_used = provider_name
            parsed_response.generation_time = generation_time
            
            # Add to history
            self.optimization_history.append({
                'request': request,
                'response': parsed_response,
                'provider_used': provider_name,
                'generation_time': generation_time,
                'timestamp': time.time()
            })
            
            # Save kernel and log to disk
            self.save_kernel_with_log(request, parsed_response)
            
            logger.info(f"LLM optimization completed in {generation_time:.2f}s using {provider_name}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"LLM optimization failed: {str(e)}")
            # Update provider status if it's an API error
            if provider_name and "API" in str(e):
                # self.config_manager.update_provider_status(provider_name, ProviderStatus.ERROR, str(e))
                pass  # Remove or comment out if ProviderStatus is not implemented
            return self._create_error_response(str(e))
    
    def _generate_optimization_prompt(self, request: OptimizationRequest) -> str:
        """Generate a comprehensive prompt for kernel optimization."""
        
        prompt = f"""
You are an expert CUDA kernel optimization specialist. Your task is to generate an optimized CUDA kernel for the following operation:

OPERATION DETAILS:
- Operation Type: {request.operation_type}
- Operation Info: {json.dumps(request.operation_info, indent=2)}

{self._format_current_kernel(request.current_kernel) if request.current_kernel else ""}

{self._format_performance_metrics(request.performance_metrics) if request.performance_metrics else ""}

OPTIMIZATION GOALS:
{chr(10).join(f"- {goal}" for goal in (request.optimization_goals or ['maximize performance', 'minimize memory usage']))}

CONSTRAINTS:
{json.dumps(request.constraints or {}, indent=2)}

REQUIREMENTS:
1. Generate a complete, compilable CUDA kernel
2. Include all necessary includes and defines
3. Use shared memory where beneficial
4. Optimize for coalesced memory access
5. Consider loop unrolling and register usage
6. Add detailed comments explaining optimizations
7. Provide confidence score (0-1) for the optimization
8. List specific optimization techniques used
9. Predict performance improvements

RESPONSE FORMAT:
```json
{{
    "optimized_kernel": "// Complete CUDA kernel code here",
    "optimization_explanations": ["explanation1", "explanation2"],
    "performance_predictions": {{
        "expected_speedup": 2.5,
        "memory_bandwidth_improvement": "30%",
        "compute_efficiency": "85%"
    }},
    "confidence_score": 0.85,
    "suggested_parameters": {{
        "block_size": [16, 16],
        "grid_size": [32, 32],
        "shared_memory_size": 4096
    }},
    "warnings": ["warning1", "warning2"],
    "errors": []
}}
```

Generate the optimized kernel now:
"""
        return prompt
    
    def _format_current_kernel(self, kernel: str) -> str:
        """Format current kernel for prompt."""
        return f"""
CURRENT KERNEL:
```cuda
{kernel}
```
"""
    
    def _format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics for prompt."""
        return f"""
PERFORMANCE METRICS:
{json.dumps(metrics, indent=2)}
"""
    
    def _parse_llm_response(self, response_text: str, request: OptimizationRequest) -> OptimizationResponse:
        """Parse and validate LLM response."""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                return OptimizationResponse(
                    optimized_kernel=parsed.get('optimized_kernel', ''),
                    optimization_explanations=parsed.get('optimization_explanations', []),
                    performance_predictions=parsed.get('performance_predictions', {}),
                    confidence_score=parsed.get('confidence_score', 0.5),
                    suggested_parameters=parsed.get('suggested_parameters', {}),
                    warnings=parsed.get('warnings', []),
                    errors=parsed.get('errors', [])
                )
            else:
                # Fallback: treat entire response as kernel
                return OptimizationResponse(
                    optimized_kernel=response_text,
                    optimization_explanations=["Generated kernel from LLM response"],
                    performance_predictions={"expected_speedup": 1.5},
                    confidence_score=0.6,
                    suggested_parameters={},
                    warnings=["Could not parse structured response"]
                )
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return OptimizationResponse(
                optimized_kernel=response_text,
                optimization_explanations=["Generated kernel from LLM response"],
                performance_predictions={"expected_speedup": 1.5},
                confidence_score=0.5,
                suggested_parameters={},
                warnings=["Failed to parse structured response"]
            )
    
    def _create_error_response(self, error_message: str) -> OptimizationResponse:
        """Create error response when optimization fails."""
        return OptimizationResponse(
            optimized_kernel="",
            optimization_explanations=[],
            performance_predictions={},
            confidence_score=0.0,
            suggested_parameters={},
            warnings=[],
            errors=[error_message],
            provider_used="error"
        )
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimization_history.clear()
    
    def analyze_performance_bottlenecks(self, operation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential performance bottlenecks."""
        prompt = f"""
Analyze the following CUDA operation for potential performance bottlenecks:

{json.dumps(operation_info, indent=2)}

Provide a detailed analysis of:
1. Memory access patterns
2. Compute intensity
3. Parallelization efficiency
4. Potential bottlenecks
5. Optimization opportunities

Format as JSON with keys: bottlenecks, recommendations, priority_level
"""
        
        try:
            # Use the best available provider
            provider_name = self.config_manager.get_best_provider()
            if not provider_name:
                return {"bottlenecks": [], "recommendations": [], "priority_level": "unknown"}
            
            provider = self.providers.get(provider_name)
            provider_config = self.config_manager.get_provider_config(provider_name)
            
            if provider and provider_config:
                response = provider.generate_response(prompt, provider_config)
                return json.loads(response)
            else:
                return {"bottlenecks": [], "recommendations": [], "priority_level": "unknown"}
        except Exception as e:
            logger.error(f"Bottleneck analysis failed: {str(e)}")
            return {"bottlenecks": [], "recommendations": [], "priority_level": "unknown"}
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary from the config manager."""
        return self.config_manager.get_configuration_summary()
    
    def print_configuration_summary(self):
        """Print configuration summary."""
        self.config_manager.print_configuration_summary()

    def save_kernel(self, response: OptimizationResponse, directory: str):
        """Save the optimized kernel to disk with a timestamp and metadata."""
        if not response.optimized_kernel:
            logger.warning("Cannot save an empty kernel.")
            return

        # Create the directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Generate a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{timestamp}_{response.provider_used}_kernel.cu"

        # Construct the full path
        full_path = Path(directory) / filename

        # Save the kernel to file
        with open(full_path, 'w') as f:
            f.write(response.optimized_kernel)

        logger.info(f"Kernel saved to {full_path}")

    def save_optimization_log(self, request: OptimizationRequest, response: OptimizationResponse, 
                            directory: str = "logs/optimizations"):
        """Save detailed optimization log with metadata."""
        # Create the directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Generate timestamp and filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_filename = f"{timestamp}_{response.provider_used}_{request.operation_type}_log.json"

        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "operation_type": request.operation_type,
            "provider_used": response.provider_used,
            "generation_time": response.generation_time,
            "confidence_score": response.confidence_score,
            "request": {
                "operation_info": request.operation_info,
                "optimization_goals": request.optimization_goals,
                "constraints": request.constraints
            },
            "response": {
                "optimization_explanations": response.optimization_explanations,
                "performance_predictions": response.performance_predictions,
                "suggested_parameters": response.suggested_parameters,
                "warnings": response.warnings,
                "errors": response.errors,
                "kernel_length": len(response.optimized_kernel)
            },
            "kernel_file": f"{timestamp}_{response.provider_used}_kernel.cu"
        }

        # Save log to JSON file
        log_path = Path(directory) / log_filename
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=2)

        logger.info(f"Optimization log saved to {log_path}")

    def save_kernel_with_log(self, request: OptimizationRequest, response: OptimizationResponse, 
                           kernel_directory: str = "generated_kernels", 
                           log_directory: str = "logs/optimizations"):
        """Save both the kernel and detailed log."""
        # Save the kernel
        self.save_kernel(response, kernel_directory)
        
        # Save the detailed log
        self.save_optimization_log(request, response, log_directory) 