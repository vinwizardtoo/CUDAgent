"""
Example: Basic PyTorch to CUDA Kernel Optimization
"""
from cudagent import CUDAgent

# Initialize CUDAgent
agent = CUDAgent()

# Optimize a matrix multiplication operation
result = agent.optimize_operation(
    operation_type="matmul",
    input_shape="1024,1024",
    output_shape="1024,1024"
)

print("Generated kernel path:", result['kernel_path'])
print("Optimization summary:", result.get('optimization_summary')) 