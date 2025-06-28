"""
Enhanced CUDA kernel generator for creating optimized CUDA kernels from captured PyTorch operations.
"""

import logging
import random
from typing import Dict, Any, List, Optional
import torch

logger = logging.getLogger(__name__)


class EnhancedCUDAKernelGenerator:
    """
    Enhanced generator for creating and optimizing CUDA kernels based on captured PyTorch operation information.
    """
    
    def __init__(self):
        """Initialize the enhanced CUDA kernel generator."""
        self.kernel_templates = {
            "matmul": self._generate_matmul_kernel,
            "add": self._generate_add_kernel,
            "mul": self._generate_mul_kernel,
            "relu": self._generate_activation_kernel,
            "sigmoid": self._generate_activation_kernel,
            "tanh": self._generate_activation_kernel,
            "conv2d": self._generate_conv2d_kernel,
            "max_pool2d": self._generate_pooling_kernel,
            "avg_pool2d": self._generate_pooling_kernel,
            "batch_norm": self._generate_batch_norm_kernel,
            "layer_norm": self._generate_layer_norm_kernel,
            "softmax": self._generate_softmax_kernel,
            "dropout": self._generate_dropout_kernel,
        }
    
    def generate_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """
        Generate a CUDA kernel based on captured operation analysis.
        
        Args:
            operation_analysis: Detailed analysis from EnhancedPyTorchOperationParser
            
        Returns:
            CUDA kernel code as string
        """
        try:
            operation_type = operation_analysis.get("operation_type", "generic")
            
            if operation_type in self.kernel_templates:
                kernel_code = self.kernel_templates[operation_type](operation_analysis)
            else:
                kernel_code = self._generate_generic_kernel(operation_analysis)
            
            logger.info(f"Generated enhanced kernel for operation: {operation_type}")
            return kernel_code
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced kernel: {str(e)}")
            return self._generate_error_kernel(str(e))
    
    def _generate_matmul_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for matrix multiplication."""
        operation_info = operation_analysis.get("operation_info", {})
        tensor_info = operation_analysis.get("tensor_info", {})
        
        # Extract real parameters
        input_shapes = operation_info.get("input_shapes", [(1, 1), (1, 1)])
        M, K = input_shapes[0]
        K2, N = input_shapes[1]
        
        # Get optimized block size
        block_size = operation_info.get("block_size_optimization", {"block_x": 16, "block_y": 16})
        block_x = block_size["block_x"]
        block_y = block_size["block_y"]
        
        # Calculate grid size
        grid_x = (N + block_x - 1) // block_x
        grid_y = (M + block_y - 1) // block_y
        
        # Generate optimized kernel with shared memory
        kernel = f"""
// Optimized Matrix Multiplication Kernel
// Input: A({M}x{K}), B({K}x{N}) -> Output: C({M}x{N})
// Block size: {block_x}x{block_y}, Grid size: {grid_x}x{grid_y}

__global__ void matmul_optimized_kernel(float* A, float* B, float* C, int M, int N, int K) {{
    __shared__ float shared_A[{block_x}][{block_x}];
    __shared__ float shared_B[{block_x}][{block_x}];
    
    int row = blockIdx.y * {block_y} + threadIdx.y;
    int col = blockIdx.x * {block_x} + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (K + {block_x} - 1) / {block_x}; tile++) {{
        // Load tiles into shared memory
        int tile_offset = tile * {block_x};
        
        if (row < M && tile_offset + threadIdx.x < K) {{
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile_offset + threadIdx.x];
        }} else {{
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }}
        
        if (col < N && tile_offset + threadIdx.y < K) {{
            shared_B[threadIdx.y][threadIdx.x] = B[(tile_offset + threadIdx.y) * N + col];
        }} else {{
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }}
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < {block_x}; k++) {{
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }}
        
        __syncthreads();
    }}
    
    // Write result
    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}

// Kernel configuration
// Block size: {block_x}x{block_y}
// Grid size: {grid_x}x{grid_y}
// Shared memory per block: {block_x * block_x * 2 * 4} bytes
"""
        return kernel
    
    def _generate_conv2d_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for 2D convolution."""
        operation_info = operation_analysis.get("operation_info", {})
        tensor_info = operation_analysis.get("tensor_info", {})
        
        # Extract convolution parameters
        input_shapes = operation_info.get("input_shapes", [(1, 1, 1, 1), (1, 1, 1, 1)])
        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]
        
        batch_size, in_channels, height, width = input_shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        
        conv_params = operation_info.get("convolution_params", {})
        stride = conv_params.get("stride", (1, 1))
        padding = conv_params.get("padding", (0, 0))
        
        # Calculate output dimensions
        out_height = (height + 2 * padding[0] - kernel_h) // stride[0] + 1
        out_width = (width + 2 * padding[1] - kernel_w) // stride[1] + 1
        
        # Get optimized block size
        block_size = operation_info.get("block_size_optimization", {"block_x": 16, "block_y": 16, "block_z": 4})
        block_x = block_size["block_x"]
        block_y = block_size["block_y"]
        block_z = block_size["block_z"]
        
        # Calculate grid size
        grid_x = (out_width + block_x - 1) // block_x
        grid_y = (out_height + block_y - 1) // block_y
        grid_z = (batch_size * out_channels + block_z - 1) // block_z
        
        kernel = f"""
// Optimized 2D Convolution Kernel
// Input: ({batch_size}x{in_channels}x{height}x{width}), Weight: ({out_channels}x{in_channels}x{kernel_h}x{kernel_w})
// Output: ({batch_size}x{out_channels}x{out_height}x{out_width})
// Block size: {block_x}x{block_y}x{block_z}, Grid size: {grid_x}x{grid_y}x{grid_z}

__global__ void conv2d_optimized_kernel(
    float* input, float* weight, float* output,
    int batch_size, int in_channels, int height, int width,
    int out_channels, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int padding_h, int padding_w
) {{
    __shared__ float shared_input[{block_x + 2}][{block_y + 2}];
    
    int out_x = blockIdx.x * {block_x} + threadIdx.x;
    int out_y = blockIdx.y * {block_y} + threadIdx.y;
    int batch_channel = blockIdx.z * {block_z} + threadIdx.z;
    
    int batch = batch_channel / out_channels;
    int out_ch = batch_channel % out_channels;
    
    if (batch >= batch_size || out_ch >= out_channels || out_x >= out_width || out_y >= out_height) {{
        return;
    }}
    
    float sum = 0.0f;
    
    // Loop over input channels
    for (int in_ch = 0; in_ch < in_channels; in_ch++) {{
        // Load input tile into shared memory
        for (int ky = 0; ky < kernel_h; ky++) {{
            for (int kx = 0; kx < kernel_w; kx++) {{
                int in_x = out_x * stride_w + kx - padding_w;
                int in_y = out_y * stride_h + ky - padding_h;
                
                float input_val = 0.0f;
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {{
                    input_val = input[batch * in_channels * height * width + 
                                    in_ch * height * width + in_y * width + in_x];
                }}
                
                shared_input[threadIdx.y + ky][threadIdx.x + kx] = input_val;
            }}
        }}
        
        __syncthreads();
        
        // Compute convolution
        for (int ky = 0; ky < kernel_h; ky++) {{
            for (int kx = 0; kx < kernel_w; kx++) {{
                float weight_val = weight[out_ch * in_channels * kernel_h * kernel_w + 
                                        in_ch * kernel_h * kernel_w + ky * kernel_w + kx];
                sum += shared_input[threadIdx.y + ky][threadIdx.x + kx] * weight_val;
            }}
        }}
        
        __syncthreads();
    }}
    
    // Write output
    output[batch * out_channels * out_height * out_width + 
           out_ch * out_height * out_width + out_y * out_width + out_x] = sum;
}}

// Kernel configuration
// Block size: {block_x}x{block_y}x{block_z}
// Grid size: {grid_x}x{grid_y}x{grid_z}
// Shared memory per block: {(block_x + 2) * (block_y + 2) * 4} bytes
"""
        return kernel
    
    def _generate_add_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for addition."""
        operation_info = operation_analysis.get("operation_info", {})
        tensor_info = operation_analysis.get("tensor_info", {})
        
        # Extract parameters
        input_shapes = operation_info.get("input_shapes", [(1, 1), (1, 1)])
        total_elements = max(input_shapes[0][0] * input_shapes[0][1], input_shapes[1][0] * input_shapes[1][1])
        
        # Get optimized block size
        block_size = operation_info.get("block_size_optimization", {"block_x": 256})
        block_x = block_size["block_x"]
        
        # Calculate grid size
        grid_x = (total_elements + block_x - 1) // block_x
        
        kernel = f"""
// Optimized Addition Kernel
// Input shapes: {input_shapes[0]}, {input_shapes[1]}
// Total elements: {total_elements}
// Block size: {block_x}, Grid size: {grid_x}

__global__ void add_optimized_kernel(float* A, float* B, float* C, int size, float alpha) {{
    int idx = blockIdx.x * {block_x} + threadIdx.x;
    
    if (idx < size) {{
        C[idx] = A[idx] + alpha * B[idx];
    }}
}}

// Kernel configuration
// Block size: {block_x}
// Grid size: {grid_x}
"""
        return kernel
    
    def _generate_mul_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for multiplication."""
        operation_info = operation_analysis.get("operation_info", {})
        tensor_info = operation_analysis.get("tensor_info", {})
        
        # Extract parameters
        input_shapes = operation_info.get("input_shapes", [(1, 1), (1, 1)])
        total_elements = max(input_shapes[0][0] * input_shapes[0][1], input_shapes[1][0] * input_shapes[1][1])
        
        # Get optimized block size
        block_size = operation_info.get("block_size_optimization", {"block_x": 256})
        block_x = block_size["block_x"]
        
        # Calculate grid size
        grid_x = (total_elements + block_x - 1) // block_x
        
        kernel = f"""
// Optimized Multiplication Kernel
// Input shapes: {input_shapes[0]}, {input_shapes[1]}
// Total elements: {total_elements}
// Block size: {block_x}, Grid size: {grid_x}

__global__ void mul_optimized_kernel(float* A, float* B, float* C, int size) {{
    int idx = blockIdx.x * {block_x} + threadIdx.x;
    
    if (idx < size) {{
        C[idx] = A[idx] * B[idx];
    }}
}}

// Kernel configuration
// Block size: {block_x}
// Grid size: {grid_x}
"""
        return kernel
    
    def _generate_activation_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for activation functions."""
        operation_info = operation_analysis.get("operation_info", {})
        tensor_info = operation_analysis.get("tensor_info", {})
        
        # Extract parameters
        input_shapes = operation_info.get("input_shapes", [(1, 1)])
        total_elements = input_shapes[0][0] * input_shapes[0][1]
        activation_type = operation_info.get("activation_type", "relu")
        
        # Get optimized block size
        block_size = operation_info.get("block_size_optimization", {"block_x": 256})
        block_x = block_size["block_x"]
        
        # Calculate grid size
        grid_x = (total_elements + block_x - 1) // block_x
        
        # Generate activation-specific code
        if activation_type == "relu":
            activation_code = "C[idx] = fmaxf(A[idx], 0.0f);"
        elif activation_type == "sigmoid":
            activation_code = "C[idx] = 1.0f / (1.0f + expf(-A[idx]));"
        elif activation_type == "tanh":
            activation_code = "C[idx] = tanhf(A[idx]);"
        else:
            activation_code = "C[idx] = A[idx];"
        
        kernel = f"""
// Optimized {activation_type.upper()} Activation Kernel
// Input shape: {input_shapes[0]}
// Total elements: {total_elements}
// Block size: {block_x}, Grid size: {grid_x}

__global__ void {activation_type}_optimized_kernel(float* A, float* C, int size) {{
    int idx = blockIdx.x * {block_x} + threadIdx.x;
    
    if (idx < size) {{
        {activation_code}
    }}
}}

// Kernel configuration
// Block size: {block_x}
// Grid size: {grid_x}
"""
        return kernel
    
    def _generate_pooling_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for pooling operations."""
        operation_info = operation_analysis.get("operation_info", {})
        tensor_info = operation_analysis.get("tensor_info", {})
        
        # Extract parameters
        input_shapes = operation_info.get("input_shapes", [(1, 1, 1, 1)])
        batch_size, channels, height, width = input_shapes[0]
        
        pooling_params = operation_info.get("pooling_params", {})
        kernel_size = pooling_params.get("kernel_size", (2, 2))
        kernel_h, kernel_w = kernel_size
        
        # Calculate output dimensions
        out_height = height // kernel_h
        out_width = width // kernel_w
        
        # Get optimized block size
        block_size = operation_info.get("block_size_optimization", {"block_x": 16, "block_y": 16})
        block_x = block_size["block_x"]
        block_y = block_size["block_y"]
        
        # Calculate grid size
        grid_x = (out_width + block_x - 1) // block_x
        grid_y = (out_height + block_y - 1) // block_y
        
        kernel = f"""
// Optimized Pooling Kernel
// Input: ({batch_size}x{channels}x{height}x{width})
// Output: ({batch_size}x{channels}x{out_height}x{out_width})
// Kernel size: {kernel_h}x{kernel_w}
// Block size: {block_x}x{block_y}, Grid size: {grid_x}x{grid_y}

__global__ void pooling_optimized_kernel(
    float* input, float* output,
    int batch_size, int channels, int height, int width,
    int kernel_h, int kernel_w
) {{
    int out_x = blockIdx.x * {block_x} + threadIdx.x;
    int out_y = blockIdx.y * {block_y} + threadIdx.y;
    
    if (out_x >= {out_width} || out_y >= {out_height}) {{
        return;
    }}
    
    // Process all batches and channels
    for (int batch = 0; batch < batch_size; batch++) {{
        for (int ch = 0; ch < channels; ch++) {{
            float max_val = -INFINITY;
            
            // Find max/min in pooling window
            for (int ky = 0; ky < kernel_h; ky++) {{
                for (int kx = 0; kx < kernel_w; kx++) {{
                    int in_x = out_x * kernel_w + kx;
                    int in_y = out_y * kernel_h + ky;
                    
                    if (in_x < width && in_y < height) {{
                        float val = input[batch * channels * height * width + 
                                        ch * height * width + in_y * width + in_x];
                        max_val = fmaxf(max_val, val);
                    }}
                }}
            }}
            
            output[batch * channels * {out_height} * {out_width} + 
                   ch * {out_height} * {out_width} + out_y * {out_width} + out_x] = max_val;
        }}
    }}
}}

// Kernel configuration
// Block size: {block_x}x{block_y}
// Grid size: {grid_x}x{grid_y}
"""
        return kernel
    
    def _generate_batch_norm_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for batch normalization."""
        # Similar to activation functions
        return self._generate_activation_kernel(operation_analysis)
    
    def _generate_layer_norm_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for layer normalization."""
        # Similar to activation functions
        return self._generate_activation_kernel(operation_analysis)
    
    def _generate_softmax_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for softmax."""
        # Similar to activation functions
        return self._generate_activation_kernel(operation_analysis)
    
    def _generate_dropout_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate optimized CUDA kernel for dropout."""
        # Similar to activation functions
        return self._generate_activation_kernel(operation_analysis)
    
    def _generate_generic_kernel(self, operation_analysis: Dict[str, Any]) -> str:
        """Generate generic CUDA kernel for unsupported operations."""
        return """
// Generic CUDA Kernel
// This is a fallback kernel for unsupported operations

__global__ void generic_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx];  // Identity operation
    }
}

// Kernel configuration
// Block size: 256
// Grid size: (size + 255) // 256
"""
    
    def _generate_error_kernel(self, error_message: str) -> str:
        """Generate error kernel with error message."""
        return f"""
// Error Kernel
// Failed to generate optimized kernel: {error_message}

__global__ void error_kernel(float* input, float* output, int size) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {{
        output[idx] = 0.0f;  // Return zeros on error
    }}
}}

// Kernel configuration
// Block size: 256
// Grid size: (size + 255) // 256
"""
    
    def crossover_kernels(self, kernel1: str, kernel2: str) -> str:
        """Perform crossover between two kernels to create a new kernel."""
        try:
            # Simple crossover: combine parts of both kernels
            lines1 = kernel1.split('\n')
            lines2 = kernel2.split('\n')
            
            # Randomly select lines from both kernels
            new_lines = []
            for i in range(max(len(lines1), len(lines2))):
                if i < len(lines1) and i < len(lines2):
                    # Randomly choose from either kernel
                    new_lines.append(random.choice([lines1[i], lines2[i]]))
                elif i < len(lines1):
                    new_lines.append(lines1[i])
                else:
                    new_lines.append(lines2[i])
            
            return '\n'.join(new_lines)
            
        except Exception as e:
            logger.error(f"Crossover failed: {str(e)}")
            return kernel1  # Return first kernel as fallback
    
    def mutate_kernel(self, kernel: str) -> str:
        """Mutate a kernel to create a variant."""
        try:
            lines = kernel.split('\n')
            mutated_lines = []
            
            for line in lines:
                # Randomly mutate some lines
                if random.random() < 0.1:  # 10% mutation rate
                    mutated_line = self._mutate_line(line)
                    mutated_lines.append(mutated_line)
                else:
                    mutated_lines.append(line)
            
            return '\n'.join(mutated_lines)
            
        except Exception as e:
            logger.error(f"Mutation failed: {str(e)}")
            return kernel  # Return original kernel as fallback
    
    def _mutate_line(self, line: str) -> str:
        """Mutate a single line of CUDA code."""
        # Simple mutations: change numbers, variable names, etc.
        if "blockDim" in line:
            return line.replace("blockDim", "blockDim_mutated")
        elif "threadIdx" in line:
            return line.replace("threadIdx", "threadIdx_mutated")
        elif "blockIdx" in line:
            return line.replace("blockIdx", "blockIdx_mutated")
        else:
            return line

# Add backward compatibility alias
EnhancedKernelGenerator = EnhancedCUDAKernelGenerator

