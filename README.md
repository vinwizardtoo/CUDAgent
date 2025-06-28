# CUDAgent 🚀

**AI-Powered CUDA Kernel Optimization for GPU Engineers**

Transform your PyTorch code into highly optimized CUDA kernels using advanced AI techniques. CUDAgent automatically analyzes your operations, generates optimized CUDA kernels, and provides performance insights.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/cudagent.svg)](https://badge.fury.io/py/cudagent)

## 🎯 Features

- **🤖 AI-Powered Optimization**: Uses LLMs to generate and optimize CUDA kernels
- **🔍 PyTorch Integration**: Automatically captures and analyzes PyTorch operations
- **⚡ Performance Analysis**: Benchmarks and validates generated kernels
- **🛠️ Multi-Provider Support**: OpenAI, Anthropic, and local/mock providers
- **📊 Real-time Validation**: Ensures kernel correctness and performance
- **🎨 Easy-to-Use API**: Simple interface for quick integration

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install cudagent

# With GPU support
pip install cudagent[gpu]

# With LLM support
pip install cudagent[llm]

# Full installation
pip install cudagent[all]
```

### Basic Usage

```python
from cudagent import CUDAgent

# Initialize CUDAgent
agent = CUDAgent()

# Optimize a matrix multiplication kernel
optimized_kernel = agent.optimize(
    operation_type="matmul",
    input_shapes=[(1000, 1000), (1000, 1000)]
)

print(optimized_kernel)
```

### PyTorch Integration

```python
import torch
from cudagent import CUDAgent

# Define your PyTorch operations
def my_model(x, y):
    return torch.matmul(x, y) + torch.relu(x)

# Initialize CUDAgent
agent = CUDAgent()

# Capture and optimize operations
optimized_kernels = agent.capture_and_optimize([my_model])

# Use the optimized kernels
for kernel in optimized_kernels:
    print(f"Generated kernel:\n{kernel}")
```

## 🛠️ Setup

### 1. Install CUDAgent

```bash
pip install cudagent[all]
```

### 2. Configure API Keys

```bash
# Interactive setup
cudagent-setup

# Or non-interactive
cudagent-setup --openai-key YOUR_OPENAI_KEY --anthropic-key YOUR_ANTHROPIC_KEY
```

### 3. Test Installation

```bash
# Basic functionality test
cudagent-test --test-type=basic

# Full integration test
cudagent-test --test-type=full --verbose
```

## 📚 Examples

### Example 1: Matrix Multiplication

```python
from cudagent import CUDAgent

agent = CUDAgent()

# Optimize matrix multiplication
kernel = agent.optimize(
    operation_type="matmul",
    input_shapes=[(1024, 1024), (1024, 1024)],
    dtype="float32"
)

# Benchmark the operation
results = agent.benchmark(
    operation_type="matmul",
    input_shapes=[(1024, 1024), (1024, 1024)],
    iterations=100
)

print(f"Performance: {results['gflops']:.2f} GFLOPS")
```

### Example 2: Convolution Operation

```python
from cudagent import CUDAgent

agent = CUDAgent()

# Optimize 2D convolution
kernel = agent.optimize(
    operation_type="conv2d",
    input_shapes=[(1, 64, 224, 224), (128, 64, 3, 3)],
    dtype="float32"
)

# Validate the kernel
validation = agent.validate(
    kernel_code=kernel,
    operation_type="conv2d",
    input_shapes=[(1, 64, 224, 224), (128, 64, 3, 3)]
)

print(f"Validation: {validation['is_valid']}")
```

### Example 3: Custom PyTorch Model

```python
import torch
import torch.nn as nn
from cudagent import CUDAgent

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1000, 500)
        self.linear2 = nn.Linear(500, 100)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Initialize model and CUDAgent
model = MyModel()
agent = CUDAgent()

# Capture operations from model
def forward_pass(x):
    return model(x)

# Optimize the operations
kernels = agent.capture_and_optimize([forward_pass])

print(f"Generated {len(kernels)} optimized kernels")
```

## 🔧 Configuration

### API Keys

CUDAgent supports multiple LLM providers. Configure them in `config/api_keys.json`:

```json
{
  "openai": {
    "api_key": "your-openai-api-key",
    "model": "gpt-4",
    "temperature": 0.1,
    "max_tokens": 4000
  },
  "anthropic": {
    "api_key": "your-anthropic-api-key",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.1,
    "max_tokens": 4000
  },
  "local": {
    "model": "mock",
    "enabled": true
  }
}
```

### Environment Variables

You can also use environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 🧪 Testing

### Run Tests

```bash
# Basic tests (no GPU required)
cudagent-test --test-type=basic

# LLM tests (requires API keys)
cudagent-test --test-type=llm

# GPU tests (requires CUDA)
cudagent-test --test-type=gpu

# Full integration test
cudagent-test --test-type=full --verbose
```

### Test with Python

```python
from cudagent import CUDAgent

# Test basic functionality
agent = CUDAgent()

# Test kernel generation
kernel = agent.optimize("matmul", [(100, 100), (100, 100)])
assert kernel is not None

# Test benchmarking
results = agent.benchmark("matmul", [(100, 100), (100, 100)])
assert results is not None

print("✅ All tests passed!")
```

## 📖 API Reference

### Main Classes

#### `CUDAgent`

The main interface for CUDAgent functionality.

```python
from cudagent import CUDAgent

agent = CUDAgent(config_file="config/api_keys.json")
```

**Methods:**

- `optimize(operation_type, input_shapes, output_shape=None, dtype="float32")`: Optimize a CUDA kernel
- `capture_and_optimize(functions)`: Capture PyTorch operations and optimize them
- `benchmark(operation_type, input_shapes, iterations=100)`: Benchmark an operation
- `validate(kernel_code, operation_type, input_shapes)`: Validate a CUDA kernel

#### `ConfigManager`

Manages API keys and provider configuration.

```python
from cudagent import ConfigManager

config = ConfigManager("config/api_keys.json")
```

#### `LLMAgent`

Handles LLM interactions for kernel optimization.

```python
from cudagent import LLMAgent, ConfigManager

config = ConfigManager()
llm_agent = LLMAgent(config)
```

### Supported Operations

- **Matrix Operations**: `matmul`, `add`, `sub`, `mul`, `div`
- **Convolution**: `conv1d`, `conv2d`, `conv3d`
- **Pooling**: `max_pool2d`, `avg_pool2d`
- **Activation**: `relu`, `sigmoid`, `tanh`
- **Normalization**: `batch_norm`, `layer_norm`
- **Custom**: Any custom operation can be defined

## 🏗️ Architecture

CUDAgent consists of several key components:

```
cudagent/
├── agents/           # AI agents for optimization
├── core/            # Core optimization logic
├── parsers/         # PyTorch operation parsing
├── profiling/       # Performance benchmarking
├── utils/           # Utility functions
├── cli/             # Command-line interface
└── examples/        # Usage examples
```

### Component Overview

- **Agents**: LLM integration, kernel optimization, performance analysis
- **Parsers**: PyTorch operation capture and analysis
- **Generators**: CUDA kernel generation from specifications
- **Validators**: Kernel correctness and performance validation
- **Benchmarkers**: Performance measurement and analysis

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/cudagent/cudagent.git
cd cudagent

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Run linting
black cudagent/
flake8 cudagent/
mypy cudagent/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Sakana AI's approach to AI-powered optimization
- Built with PyTorch and CUDA
- Uses OpenAI and Anthropic APIs for LLM-powered optimization

## 📞 Support

- **Documentation**: [https://docs.cudagent.ai](https://docs.cudagent.ai)
- **Issues**: [GitHub Issues](https://github.com/cudagent/cudagent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cudagent/cudagent/discussions)
- **Email**: contact@cudagent.ai

---

**Made with ❤️ for the GPU computing community** 