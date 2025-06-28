# CUDAgent Packaging Guide

This guide explains how to build, test, and distribute the CUDAgent package.

## Overview

CUDAgent is packaged as a Python package with the following structure:
- **Core package**: `cudagent/` - Main library code
- **CLI tools**: Command-line interface for easy usage
- **Optional dependencies**: GPU, LLM, and development support
- **Build tools**: Automated build and distribution scripts

## Package Structure

```
cudagent/
├── __init__.py              # Main package entry point
├── _version.py              # Version information (auto-generated)
├── cli/                     # Command-line interface
│   ├── __init__.py          # CLI main entry point
│   ├── setup.py             # Setup and configuration
│   ├── test.py              # System testing
│   ├── optimize.py          # Kernel optimization
│   └── benchmark.py         # Performance benchmarking
├── core/                    # Core functionality
├── agents/                  # AI agents
├── parsers/                 # PyTorch parsing
├── utils/                   # Utility functions
├── profiling/               # Performance profiling
└── examples/                # Usage examples
```

## Building the Package

### Prerequisites

1. **Python 3.8+** installed
2. **setuptools** and **wheel** for building
3. **twine** for package checking and uploading

```bash
pip install setuptools wheel twine
```

### Quick Build

Use the automated build script:

```bash
# Full build process (clean, test, build, install, test CLI)
python build.py

# Individual steps
python build.py --clean      # Clean build artifacts
python build.py --test       # Run tests only
python build.py --build      # Build package only
python build.py --install    # Install package only
python build.py --cli        # Test CLI only
```

### Manual Build

```bash
# Clean previous builds
make clean

# Install in development mode
make install

# Run tests
make test

# Build package
make build

# Check package
twine check dist/*
```

### Using Makefile

The Makefile provides convenient commands:

```bash
# Show all available commands
make help

# Development workflow
make dev

# Full build process
make full

# Release preparation
make release
```

## Package Configuration

### setup.py

Main package configuration with:
- Package metadata (name, version, description)
- Dependencies and optional dependencies
- Entry points for CLI commands
- Package data inclusion

### pyproject.toml

Modern Python packaging configuration with:
- Build system requirements
- Project metadata
- Development tool configurations (black, mypy, pytest)

### MANIFEST.in

Specifies additional files to include in the package:
- Configuration files
- Examples
- Templates
- Documentation

## CLI Commands

The package provides several CLI commands:

### Main CLI
```bash
cudagent --help                    # Show main help
cudagent setup --help              # Setup configuration
cudagent test --help               # Run tests
cudagent optimize --help           # Optimize kernels
cudagent benchmark --help          # Benchmark kernels
```

### Individual Commands
```bash
cudagent-setup --help              # Setup API keys and environment
cudagent-test --help               # Run system tests
cudagent-optimize --help           # Optimize PyTorch operations
cudagent-benchmark --help          # Benchmark CUDA kernels
```

## Installation Options

### Basic Installation
```bash
pip install cudagent
```

### With Optional Dependencies
```bash
# GPU support (PyTorch)
pip install cudagent[gpu]

# LLM support (OpenAI, Anthropic)
pip install cudagent[llm]

# Development tools
pip install cudagent[dev]

# Plotting support
pip install cudagent[plotting]

# All optional dependencies
pip install cudagent[all]
```

### Development Installation
```bash
# Clone repository
git clone https://github.com/cudagent/cudagent.git
cd cudagent

# Install in development mode
pip install -e .

# Or with all dependencies
pip install -e .[all]
```

## Testing the Package

### Running Tests
```bash
# All tests
make test

# GPU tests only
make test-gpu

# LLM tests only
make test-llm

# Using pytest directly
pytest tests/ -v
pytest tests/ -m gpu -v
pytest tests/ -m llm -v
```

### Testing CLI
```bash
# Test all CLI commands
make cli-test

# Test individual commands
make cli-setup
make cli-optimize
make cli-benchmark
```

### Integration Testing
```bash
# Test package installation and basic functionality
python -c "import cudagent; print(cudagent.__version__)"

# Test CLI availability
cudagent --help
```

## Distribution

### Local Distribution
```bash
# Build distribution
make dist

# Install from local distribution
pip install dist/cudagent-*.whl
```

### PyPI Distribution

#### Test PyPI (for testing)
```bash
# Upload to test PyPI
twine upload --repository testpypi dist/*

# Install from test PyPI
pip install --index-url https://test.pypi.org/simple/ cudagent
```

#### Production PyPI
```bash
# Upload to production PyPI
twine upload dist/*

# Install from PyPI
pip install cudagent
```

### GitHub Releases

1. Create a release on GitHub
2. Upload the built distribution files
3. Users can install directly from GitHub:
   ```bash
   pip install git+https://github.com/cudagent/cudagent.git@v0.1.0
   ```

## Version Management

### Automatic Versioning
The package uses `setuptools_scm` for automatic versioning:
- Version is derived from git tags
- Development versions include commit hash
- No manual version updates needed

### Manual Versioning
To set a specific version:
```bash
# Create a git tag
git tag v0.1.0
git push origin v0.1.0

# Or edit _version.py manually
echo '__version__ = "0.1.0"' > cudagent/_version.py
```

## Quality Assurance

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Type checking
mypy cudagent/
```

### Package Quality
```bash
# Check package
twine check dist/*

# Validate setup.py
python setup.py check

# Test installation
pip install dist/cudagent-*.whl
```

## Troubleshooting

### Common Issues

1. **Import errors after installation**
   - Ensure package is installed in the correct environment
   - Check for conflicting package names

2. **CLI commands not found**
   - Verify entry points are correctly configured
   - Reinstall package after changes

3. **Missing dependencies**
   - Install optional dependencies as needed
   - Check requirements.txt for base dependencies

4. **Build failures**
   - Clean build artifacts: `make clean`
   - Update build tools: `pip install --upgrade setuptools wheel`

### Debug Mode
```bash
# Install with verbose output
pip install -e . -v

# Run tests with debug output
pytest tests/ -v -s

# Check package contents
tar -tzf dist/cudagent-*.tar.gz
```

## Best Practices

1. **Always test before releasing**
   - Run full test suite
   - Test CLI functionality
   - Verify installation works

2. **Use semantic versioning**
   - Follow semver.org guidelines
   - Update changelog with releases

3. **Document changes**
   - Update README.md for user-facing changes
   - Update this guide for packaging changes

4. **Automate where possible**
   - Use CI/CD for automated testing
   - Use build scripts for consistency

## Next Steps

1. **Documentation**: Add comprehensive API documentation
2. **CI/CD**: Set up automated testing and deployment
3. **Packaging**: Add more distribution formats (conda, docker)
4. **Monitoring**: Add usage analytics and error reporting 