#!/usr/bin/env python3
"""
Build script for CUDAgent package distribution.
Automates the build, test, and distribution process.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    if capture_output:
        return result.stdout.strip()
    return result

def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed: {path}")
            elif path.is_file():
                path.unlink()
                print(f"Removed: {path}")

def run_tests():
    """Run the test suite."""
    print("Running tests...")
    try:
        run_command("python -m pytest tests/ -v")
        print("✅ Tests passed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Tests failed")
        return False

def build_package():
    """Build the package."""
    print("Building package...")
    try:
        run_command("python setup.py sdist bdist_wheel")
        print("✅ Package built successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Package build failed")
        return False

def check_package():
    """Check the built package."""
    print("Checking package...")
    try:
        run_command("twine check dist/*")
        print("✅ Package check passed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Package check failed")
        return False

def install_package():
    """Install the package in development mode."""
    print("Installing package in development mode...")
    try:
        run_command("pip install -e .")
        print("✅ Package installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Package installation failed")
        return False

def test_cli():
    """Test CLI commands."""
    print("Testing CLI commands...")
    try:
        # Test main CLI
        output = run_command("cudagent --help", capture_output=True)
        if "CUDAgent - AI-Powered CUDA Kernel Optimization" in output:
            print("✅ Main CLI works")
        else:
            print("❌ Main CLI failed")
            return False
        
        # Test individual commands
        commands = [
            "cudagent-setup --help",
            "cudagent-test --help", 
            "cudagent-optimize --help",
            "cudagent-benchmark --help"
        ]
        
        for cmd in commands:
            try:
                output = run_command(cmd, capture_output=True)
                print(f"✅ {cmd.split()[0]} works")
            except subprocess.CalledProcessError:
                print(f"❌ {cmd.split()[0]} failed")
                return False
        
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Main build process."""
    print("🚀 Starting CUDAgent build process...")
    print("=" * 50)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Build CUDAgent package")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--build", action="store_true", help="Build package only")
    parser.add_argument("--install", action="store_true", help="Install package only")
    parser.add_argument("--cli", action="store_true", help="Test CLI only")
    parser.add_argument("--full", action="store_true", help="Run full build process")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_build()
        return
    
    if args.test:
        if not run_tests():
            sys.exit(1)
        return
    
    if args.build:
        if not build_package():
            sys.exit(1)
        return
    
    if args.install:
        if not install_package():
            sys.exit(1)
        return
    
    if args.cli:
        if not test_cli():
            sys.exit(1)
        return
    
    # Full build process
    if args.full or not any([args.clean, args.test, args.build, args.install, args.cli]):
        print("Running full build process...")
        
        # Clean
        clean_build()
        
        # Run tests
        if not run_tests():
            print("❌ Build failed at testing stage")
            sys.exit(1)
        
        # Build package
        if not build_package():
            print("❌ Build failed at package building stage")
            sys.exit(1)
        
        # Check package
        if not check_package():
            print("❌ Build failed at package checking stage")
            sys.exit(1)
        
        # Install package
        if not install_package():
            print("❌ Build failed at package installation stage")
            sys.exit(1)
        
        # Test CLI
        if not test_cli():
            print("❌ Build failed at CLI testing stage")
            sys.exit(1)
        
        print("\n🎉 Full build process completed successfully!")
        print("=" * 50)
        print("Package is ready for distribution!")
        print("Generated files:")
        for path in Path('dist').glob('*'):
            print(f"  - {path}")

if __name__ == "__main__":
    main() 