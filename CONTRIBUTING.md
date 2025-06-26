# Contributing to CUDAgent

Thank you for your interest in contributing to CUDAgent! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **🐛 Bug Reports**: Report issues and bugs
- **✨ Feature Requests**: Suggest new features and improvements
- **📝 Documentation**: Improve documentation and examples
- **🔧 Code Contributions**: Submit code improvements and optimizations
- **🧪 Testing**: Add tests and improve test coverage
- **📊 Performance**: Optimize existing kernels and algorithms
- **🌐 Community**: Help with discussions and support

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA Toolkit 11.0+
- NVIDIA GPU with CUDA support
- Git

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/CUDAgent.git
   cd CUDAgent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Run tests to ensure everything works**
   ```bash
   pytest tests/
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Follow the coding style guidelines below
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

Use conventional commit messages:

```bash
git commit -m "feat: add new optimization strategy for matrix multiplication"
git commit -m "fix: resolve memory leak in kernel compilation"
git commit -m "docs: update installation instructions"
```

### 4. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] **Tests pass**: All existing and new tests should pass
- [ ] **Code style**: Follow the project's coding standards
- [ ] **Documentation**: Update relevant documentation
- [ ] **Performance**: Ensure no performance regressions
- [ ] **Security**: No security vulnerabilities introduced

### Pull Request Template

When creating a pull request, please use the following template:

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Test addition

## Testing
- [ ] Added tests for new functionality
- [ ] All existing tests pass
- [ ] Performance benchmarks show no regression

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] No breaking changes introduced

## Related Issues
Closes #(issue number)
```

## 🎨 Coding Standards

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use meaningful variable and function names

### CUDA Code Style

- Follow NVIDIA CUDA C++ best practices
- Use consistent indentation (4 spaces)
- Add comprehensive comments for complex algorithms
- Include error checking for CUDA API calls

### Example Code Style

```python
from typing import Optional, Tuple
import torch
import numpy as np


def optimize_kernel(
    operation: torch.Tensor,
    target_device: Optional[str] = None
) -> Tuple[str, float]:
    """
    Optimize a PyTorch operation into a CUDA kernel.
    
    Args:
        operation: The PyTorch operation to optimize
        target_device: Target CUDA device (default: current device)
        
    Returns:
        Tuple of (kernel_code, speedup_factor)
        
    Raises:
        ValueError: If operation is not supported
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Implementation here
    pass
```

## 🧪 Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Include both unit tests and integration tests
- Test edge cases and error conditions
- Use descriptive test names

### Test Structure

```python
import pytest
import torch
from cudagent import optimize_kernel


class TestKernelOptimization:
    """Test suite for kernel optimization functionality."""
    
    def test_basic_matrix_multiplication(self):
        """Test optimization of basic matrix multiplication."""
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        
        kernel_code, speedup = optimize_kernel(z)
        
        assert kernel_code is not None
        assert speedup > 1.0
        assert "matmul" in kernel_code.lower()
    
    def test_unsupported_operation(self):
        """Test handling of unsupported operations."""
        with pytest.raises(ValueError, match="Unsupported operation"):
            optimize_kernel(torch.tensor([1, 2, 3]))
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cudagent

# Run specific test file
pytest tests/test_kernel_optimization.py

# Run tests in parallel
pytest -n auto
```

## 📊 Performance Guidelines

### Benchmarking

- Always benchmark your changes against the baseline
- Use consistent hardware and software configurations
- Report speedup factors and memory usage
- Include both small and large input sizes

### Performance Testing

```python
import time
import torch
from cudagent import optimize_kernel


def benchmark_operation(operation, iterations=100):
    """Benchmark an operation against optimized version."""
    # Warm up
    for _ in range(10):
        _ = operation()
    
    # Benchmark original
    start = time.time()
    for _ in range(iterations):
        result_original = operation()
    original_time = time.time() - start
    
    # Benchmark optimized
    kernel_code, _ = optimize_kernel(operation)
    start = time.time()
    for _ in range(iterations):
        result_optimized = kernel_code()
    optimized_time = time.time() - start
    
    speedup = original_time / optimized_time
    return speedup, result_original, result_optimized
```

## 🐛 Bug Reports

### Before Reporting

1. Check if the issue has already been reported
2. Try to reproduce the issue with the latest version
3. Check the documentation and existing issues

### Bug Report Template

```markdown
## Bug Description
Clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.9.7]
- CUDA: [e.g. 11.8]
- GPU: [e.g. RTX 3080]
- CUDAgent version: [e.g. 0.1.0]

## Additional Context
Any other context about the problem.
```

## 💡 Feature Requests

### Before Requesting

1. Check if the feature already exists
2. Consider if it aligns with project goals
3. Think about implementation complexity

### Feature Request Template

```markdown
## Feature Description
Clear and concise description of the feature.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
How would you like to see this implemented?

## Alternatives Considered
Any alternative solutions you've considered.

## Additional Context
Any other context or screenshots.
```

## 🏷️ Issue Labels

We use the following labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `performance`: Performance improvements
- `question`: Further information is requested
- `wontfix`: This will not be worked on

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

### Code of Conduct

We are committed to providing a welcoming and inspiring community for all. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## 🎉 Recognition

Contributors will be recognized in:

- The project's README file
- Release notes
- Contributor hall of fame (if applicable)

## 📄 License

By contributing to CUDAgent, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CUDAgent! 🚀 