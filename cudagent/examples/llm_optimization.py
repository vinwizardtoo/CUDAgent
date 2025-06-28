"""
Example: LLM-Based Kernel Optimization
"""
from cudagent import CUDAgent

# Initialize CUDAgent with LLM support (requires API key)
agent = CUDAgent(use_llm=True, llm_provider="openai")

# Optimize a convolution operation using LLM
result = agent.optimize_operation(
    operation_type="conv2d",
    input_shape="32,3,224,224",
    output_shape="32,64,112,112"
)

print("Generated kernel path:", result['kernel_path'])
print("LLM optimization summary:", result.get('optimization_summary')) 