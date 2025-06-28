#!/usr/bin/env python3
"""
CUDAgent Setup CLI

Interactive setup for API keys and configuration.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cudagent.agents.config_manager import ConfigManager


def main():
    """Main CLI entry point for setup."""
    parser = argparse.ArgumentParser(
        description="CUDAgent Setup - Configure API keys and settings"
    )
    parser.add_argument(
        "--config-file",
        default="config/api_keys.json",
        help="Path to configuration file (default: config/api_keys.json)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Run in interactive mode (default: True)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (can also use OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--anthropic-key", 
        help="Anthropic API key (can also use ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check current configuration without making changes"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_configuration()
        return
    
    if args.interactive:
        interactive_setup(args.config_file)
    else:
        non_interactive_setup(args)


def check_configuration():
    """Check current configuration."""
    print("🔍 Checking CUDAgent Configuration")
    print("=" * 40)
    
    config_manager = ConfigManager()
    config_manager.print_configuration_summary()


def interactive_setup(config_file):
    """Interactive setup process."""
    print("🔑 CUDAgent Interactive Setup")
    print("=" * 40)
    print()
    print("This will help you configure API keys for CUDAgent.")
    print("Your API keys will be stored securely and used for LLM-powered optimization.")
    print()
    
    # Ensure config directory exists
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config if it exists
    config = {}
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("📁 Found existing configuration file")
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing config: {e}")
    
    print()
    print("Let's set up your API keys:")
    print()
    
    # OpenAI setup
    print("🤖 OpenAI Configuration")
    print("-" * 25)
    openai_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if openai_key:
        config['openai'] = {
            'api_key': openai_key,
            'model': 'gpt-4',
            'temperature': 0.1,
            'max_tokens': 4000
        }
        print("✅ OpenAI configured")
    else:
        config['openai'] = {
            'api_key': 'your-openai-api-key-here',
            'model': 'gpt-4',
            'temperature': 0.1,
            'max_tokens': 4000
        }
        print("⏭️  Skipped OpenAI configuration")
    
    print()
    
    # Anthropic setup
    print("🧠 Anthropic Configuration")
    print("-" * 25)
    anthropic_key = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    
    if anthropic_key:
        config['anthropic'] = {
            'api_key': anthropic_key,
            'model': 'claude-3-sonnet-20240229',
            'temperature': 0.1,
            'max_tokens': 4000
        }
        print("✅ Anthropic configured")
    else:
        config['anthropic'] = {
            'api_key': 'your-anthropic-api-key-here',
            'model': 'claude-3-sonnet-20240229',
            'temperature': 0.1,
            'max_tokens': 4000
        }
        print("⏭️  Skipped Anthropic configuration")
    
    print()
    
    # Local configuration
    config['local'] = {
        'model': 'mock',
        'enabled': True
    }
    print("🏠 Local/Mock provider enabled (for testing)")
    
    # Save configuration
    try:
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print()
        print("✅ Configuration saved successfully!")
        print()
        print("🔒 Security Note:")
        print("- Your API keys are stored locally")
        print("- The config file is automatically ignored by git")
        print("- Keep this file secure and don't share it")
        print()
        
        # Test the configuration
        print("🧪 Testing configuration...")
        config_manager = ConfigManager(config_file)
        config_manager.print_configuration_summary()
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        sys.exit(1)


def non_interactive_setup(args):
    """Non-interactive setup using command line arguments."""
    print("🔧 CUDAgent Non-Interactive Setup")
    print("=" * 40)
    
    config = {}
    
    # Set OpenAI key
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if openai_key:
        config['openai'] = {
            'api_key': openai_key,
            'model': 'gpt-4',
            'temperature': 0.1,
            'max_tokens': 4000
        }
        print("✅ OpenAI configured")
    
    # Set Anthropic key
    anthropic_key = args.anthropic_key or os.getenv('ANTHROPIC_API_KEY')
    if anthropic_key:
        config['anthropic'] = {
            'api_key': anthropic_key,
            'model': 'claude-3-sonnet-20240229',
            'temperature': 0.1,
            'max_tokens': 4000
        }
        print("✅ Anthropic configured")
    
    # Always enable local provider
    config['local'] = {
        'model': 'mock',
        'enabled': True
    }
    print("🏠 Local/Mock provider enabled")
    
    # Save configuration
    config_path = Path(args.config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {config_path}")
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 