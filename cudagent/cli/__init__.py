"""
CUDAgent CLI - Command Line Interface
Provides command-line tools for CUDA kernel optimization and benchmarking.
"""

from .setup import main as setup_main
from .test import main as test_main
from .optimize import main as optimize_main
from .benchmark import main as benchmark_main

__all__ = [
    'setup_main',
    'test_main', 
    'optimize_main',
    'benchmark_main'
]

def main():
    """Main CLI entry point that provides a unified interface."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CUDAgent - AI-Powered CUDA Kernel Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  setup     - Configure API keys and environment
  test      - Run system tests and validation
  optimize  - Optimize PyTorch operations to CUDA kernels
  benchmark - Benchmark kernels and compare performance

For detailed help on any command, use:
  cudagent <command> --help
        """
    )
    
    parser.add_argument(
        'command',
        choices=['setup', 'test', 'optimize', 'benchmark'],
        help='Command to execute'
    )
    
    # Parse only the command, pass remaining args to subcommand
    args, remaining = parser.parse_known_args()
    
    # Route to appropriate command
    if args.command == 'setup':
        sys.argv = ['cudagent-setup'] + remaining
        setup_main()
    elif args.command == 'test':
        sys.argv = ['cudagent-test'] + remaining
        test_main()
    elif args.command == 'optimize':
        sys.argv = ['cudagent-optimize'] + remaining
        optimize_main()
    elif args.command == 'benchmark':
        sys.argv = ['cudagent-benchmark'] + remaining
        benchmark_main()

if __name__ == "__main__":
    main() 