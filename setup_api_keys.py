#!/usr/bin/env python3
"""
API Key Setup Script for CUDAgent

This script helps you set up API keys for the AI Agent Framework.
It will create or update the config/api_keys.json file with your API keys.
"""

import json
import os
import getpass
from pathlib import Path

def setup_api_keys():
    """Interactive setup for API keys."""
    print("🔑 CUDAgent API Key Setup")
    print("=" * 40)
    print()
    print("This script will help you set up API keys for the AI Agent Framework.")
    print("Your API keys will be stored in config/api_keys.json")
    print()
    
    # Ensure config directory exists
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "api_keys.json"
    
    # Load existing config if it exists
    config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
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
    openai_key = getpass.getpass("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
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
    anthropic_key = getpass.getpass("Enter your Anthropic API key (or press Enter to skip): ").strip()
    
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
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print()
        print("✅ Configuration saved to config/api_keys.json")
        print()
        print("🔒 Security Note:")
        print("- The config/api_keys.json file is already in .gitignore")
        print("- Your API keys will not be committed to version control")
        print("- Keep this file secure and don't share it")
        print()
        
        # Test the configuration
        print("🧪 Testing configuration...")
        try:
            from cudagent.agents.config_manager import ConfigManager
            cm = ConfigManager()
            cm.print_configuration_summary()
        except ImportError:
            print("⚠️  Could not test configuration (cudagent not installed)")
            print("   Run 'python test_ai_agents.py' to test after installation")
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False
    
    return True

def main():
    """Main function."""
    try:
        success = setup_api_keys()
        if success:
            print()
            print("🎉 Setup complete!")
            print()
            print("Next steps:")
            print("1. Test your configuration: python test_ai_agents.py")
            print("2. Start using the AI Agent Framework")
            print("3. If you need to update keys later, run this script again")
        else:
            print("❌ Setup failed. Please try again.")
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 