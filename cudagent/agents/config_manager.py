"""
Configuration Manager for AI Agent Framework

This module manages API keys, provider selection, and routing for different LLM providers.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60  # Timeout in seconds for API requests
    base_url: Optional[str] = None

class ConfigManager:
    """
    Manages API keys and provider configuration for the AI Agent Framework.
    
    Supports multiple providers:
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - Anthropic (Claude)
    - Local/Mock (for testing)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/api_keys.json"
        self.providers = {}
        self.best_provider = None
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from environment variables and JSON file."""
        # Try to load from JSON file first
        json_config = self._load_json_config()
        
        # Load providers with priority: env vars > JSON file > defaults
        self._load_openai_config(json_config)
        self._load_anthropic_config(json_config)
        self._load_local_config(json_config)
        
        # Determine best available provider
        self._select_best_provider()
    
    def _load_json_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.info(f"Config file not found: {self.config_file}")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
            return {}
    
    def _load_openai_config(self, json_config: Dict[str, Any]):
        """Load OpenAI configuration."""
        # Priority: environment variables > JSON config
        api_key = os.getenv('OPENAI_API_KEY') or json_config.get('openai', {}).get('api_key')
        model = os.getenv('OPENAI_MODEL') or json_config.get('openai', {}).get('model', 'gpt-4')
        temperature = float(os.getenv('OPENAI_TEMPERATURE', 
                                    json_config.get('openai', {}).get('temperature', 0.1)))
        max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', 
                                 json_config.get('openai', {}).get('max_tokens', 4000)))
        
        if api_key and api_key != "your-openai-api-key-here":
            self.providers['openai'] = LLMConfig(
                provider='openai',
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"openai provider: Configured with model {model}")
        else:
            logger.info("openai provider: No API key provided")
    
    def _load_anthropic_config(self, json_config: Dict[str, Any]):
        """Load Anthropic configuration."""
        # Priority: environment variables > JSON config
        api_key = os.getenv('ANTHROPIC_API_KEY') or json_config.get('anthropic', {}).get('api_key')
        model = os.getenv('ANTHROPIC_MODEL') or json_config.get('anthropic', {}).get('model', 'claude-3-sonnet-20240229')
        temperature = float(os.getenv('ANTHROPIC_TEMPERATURE', 
                                    json_config.get('anthropic', {}).get('temperature', 0.1)))
        max_tokens = int(os.getenv('ANTHROPIC_MAX_TOKENS', 
                                 json_config.get('anthropic', {}).get('max_tokens', 4000)))
        
        if api_key and api_key != "your-anthropic-api-key-here":
            self.providers['anthropic'] = LLMConfig(
                provider='anthropic',
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"anthropic provider: Configured with model {model}")
        else:
            logger.info("anthropic provider: No API key provided")
    
    def _load_local_config(self, json_config: Dict[str, Any]):
        """Load local/mock configuration."""
        model = os.getenv('LOCAL_MODEL') or json_config.get('local', {}).get('model', 'mock')
        enabled = json_config.get('local', {}).get('enabled', True)
        
        if enabled:
            self.providers['local'] = LLMConfig(
                provider='local',
                model=model,
                temperature=0.1,
                max_tokens=4000
            )
            logger.info("Local provider available (mock responses)")
    
    def _select_best_provider(self):
        """Select the best available provider based on priority."""
        # Priority order: openai > anthropic > local
        priority_order = ['openai', 'anthropic', 'local']
        
        for provider in priority_order:
            if provider in self.providers:
                self.best_provider = provider
                logger.info(f"Using best provider: {provider}")
                return
        
        # Fallback to local if nothing else is available
        if 'local' in self.providers:
            self.best_provider = 'local'
            logger.info("Using fallback provider: local")
        else:
            logger.warning("No providers available!")
    
    def get_provider_config(self, provider: Optional[str] = None) -> Optional[LLMConfig]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider: Provider name (openai, anthropic, local). If None, returns best provider.
            
        Returns:
            LLMConfig for the provider, or None if not available
        """
        if provider is None:
            provider = self.best_provider
        
        return self.providers.get(provider)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a specific provider is available."""
        return provider in self.providers
    
    def get_best_provider(self) -> Optional[str]:
        """Get the best available provider name."""
        return self.best_provider
    
    def print_configuration_summary(self):
        """Print a summary of the current configuration."""
        print("=" * 50)
        print("LLM Configuration Summary")
        print("=" * 50)
        print(f"Available Providers: {len(self.providers)}/{len(['openai', 'anthropic', 'local'])}")
        print(f"Best Provider: {self.best_provider or 'None'}")
        print()
        print("Provider Details:")
        
        all_providers = ['openai', 'anthropic', 'local']
        for provider in all_providers:
            if provider in self.providers:
                config = self.providers[provider]
                print(f"  ✅ {provider}: available")
                if provider != 'local':
                    print(f"    Model: {config.model}")
                    print(f"    Temperature: {config.temperature}")
            else:
                print(f"  ❌ {provider}: unavailable")
                if provider == 'openai':
                    print("    Error: No API key provided")
                elif provider == 'anthropic':
                    print("    Error: No API key provided")
                else:
                    print("    Error: Not configured")
        
        print()
        print("Setup Guide:")
        print()
        print("LLM API Key Setup Guide")
        print("=" * 22)
        print()
        print("To use the AI Agent Framework with real LLM providers, set the following environment variables:")
        print()
        print("1. OpenAI (Recommended):")
        print("   export OPENAI_API_KEY=\"sk-your-openai-key-here\"")
        print("   export OPENAI_MODEL=\"gpt-4\"  # Optional, defaults to gpt-4")
        print("   export OPENAI_TEMPERATURE=\"0.1\"  # Optional, defaults to 0.1")
        print()
        print("2. Anthropic (Alternative):")
        print("   export ANTHROPIC_API_KEY=\"sk-ant-your-anthropic-key-here\"")
        print("   export ANTHROPIC_MODEL=\"claude-3-sonnet-20240229\"  # Optional")
        print("   export ANTHROPIC_TEMPERATURE=\"0.1\"  # Optional")
        print()
        print("3. Local/Mock (Fallback):")
        print("   export LOCAL_MODEL=\"mock\"  # Optional, defaults to mock responses")
        print()
        print("Example setup:")
        print("   export OPENAI_API_KEY=\"sk-1234567890abcdef...\"")
        print("   export ANTHROPIC_API_KEY=\"sk-ant-1234567890abcdef...\"")
        print()
        print("The system will automatically:")
        print("- Detect available providers based on API keys")
        print("- Use the best available provider")
        print("- Fallback to mock responses if no APIs are available")
        print("- Handle rate limiting and errors gracefully")
        print()
        print("To test your setup, run:")
        print("   python -c \"from cudagent.agents.config_manager import ConfigManager; cm = ConfigManager(); cm.print_configuration_summary()\"")
    
    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate an API key format (basic validation).
        
        Args:
            provider: Provider name
            api_key: API key to validate
            
        Returns:
            True if key format is valid, False otherwise
        """
        if not api_key or api_key == "your-openai-api-key-here" or api_key == "your-anthropic-api-key-here":
            return False
        
        if provider == 'openai':
            return api_key.startswith('sk-') and len(api_key) > 20
        elif provider == 'anthropic':
            return api_key.startswith('sk-ant-') and len(api_key) > 20
        else:
            return True  # Local provider doesn't need validation
    
    def update_provider_config(self, provider: str, **kwargs) -> bool:
        """
        Update configuration for a provider.
        
        Args:
            provider: Provider name
            **kwargs: Configuration parameters to update
            
        Returns:
            True if update was successful, False otherwise
        """
        if provider not in self.providers:
            return False
        
        config = self.providers[provider]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return True 