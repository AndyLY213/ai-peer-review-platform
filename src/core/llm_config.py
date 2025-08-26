"""
Centralized LLM Configuration System.

This module provides a unified interface for configuring different LLM providers
including Ollama (local), OpenAI, Anthropic, and others.
"""

import os
from typing import Dict, List, Any, Optional
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")
load_dotenv(".env.local", override=True)  # Local overrides

class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"

class LLMConfig:
    """
    Centralized LLM configuration manager.
    
    Supports easy switching between different LLM providers.
    """
    
    def __init__(self):
        """Initialize LLM configuration."""
        self.provider = self._get_provider()
        self.config = self._create_config()
    
    def _get_provider(self) -> LLMProvider:
        """
        Determine which LLM provider to use based on environment variables.
        
        Returns:
            LLMProvider enum value
        """
        provider_str = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        try:
            return LLMProvider(provider_str)
        except ValueError:
            print(f"Warning: Unknown LLM provider '{provider_str}', defaulting to Ollama")
            return LLMProvider.OLLAMA
    
    def _create_config(self) -> Dict[str, Any]:
        """
        Create LLM configuration based on the selected provider.
        
        Returns:
            Dictionary with LLM configuration for AutoGen
        """
        if self.provider == LLMProvider.OLLAMA:
            return self._create_ollama_config()
        elif self.provider == LLMProvider.OPENAI:
            return self._create_openai_config()
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_config()
        elif self.provider == LLMProvider.AZURE_OPENAI:
            return self._create_azure_openai_config()
        elif self.provider == LLMProvider.GEMINI:
            return self._create_gemini_config()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _create_ollama_config(self) -> Dict[str, Any]:
        """Create Ollama configuration."""
        config_list = [
            {
                "model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
                "client_host": os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
                "api_type": "ollama"
            }
        ]
        
        return {
            "config_list": config_list,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
        }
    
    def _create_openai_config(self) -> Dict[str, Any]:
        """Create OpenAI configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        
        config_list = [
            {
                "model": os.getenv("OPENAI_MODEL", "gpt-4"),
                "api_key": api_key,
                "api_type": "openai"
            }
        ]
        
        return {
            "config_list": config_list,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
        }
    
    def _create_anthropic_config(self) -> Dict[str, Any]:
        """Create Anthropic configuration."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
        
        config_list = [
            {
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                "api_key": api_key,
                "api_type": "anthropic"
            }
        ]
        
        return {
            "config_list": config_list,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
        }
    
    def _create_azure_openai_config(self) -> Dict[str, Any]:
        """Create Azure OpenAI configuration."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_base = os.getenv("AZURE_OPENAI_API_BASE")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        
        if not api_key or not api_base:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_BASE are required for Azure OpenAI")
        
        config_list = [
            {
                "model": os.getenv("AZURE_OPENAI_MODEL", "gpt-4"),
                "api_key": api_key,
                "api_base": api_base,
                "api_version": api_version,
                "api_type": "azure"
            }
        ]
        
        return {
            "config_list": config_list,
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "timeout": int(os.getenv("LLM_TIMEOUT", "120")),
        }
    
    def _create_gemini_config(self) -> Dict[str, Any]:
        """Create Gemini configuration."""
        from src.core.autogen_gemini import create_autogen_gemini_config
        return create_autogen_gemini_config()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the LLM configuration.
        
        Returns:
            Dictionary with LLM configuration for AutoGen
        """
        return self.config
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.provider.value,
            "model": self.config["config_list"][0]["model"],
            "temperature": self.config["temperature"],
            "timeout": self.config["timeout"]
        }

# Global instance for easy access
_llm_config_instance = None

def get_llm_config() -> Dict[str, Any]:
    """
    Get the global LLM configuration.
    
    Returns:
        Dictionary with LLM configuration for AutoGen
    """
    global _llm_config_instance
    if _llm_config_instance is None:
        _llm_config_instance = LLMConfig()
    return _llm_config_instance.get_config()

def get_llm_provider_info() -> Dict[str, Any]:
    """
    Get information about the current LLM provider.
    
    Returns:
        Dictionary with provider information
    """
    global _llm_config_instance
    if _llm_config_instance is None:
        _llm_config_instance = LLMConfig()
    return _llm_config_instance.get_provider_info()

def reload_llm_config():
    """Reload the LLM configuration (useful after changing environment variables)."""
    global _llm_config_instance
    _llm_config_instance = None

# Legacy function for backward compatibility
def create_ollama_config() -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.
    
    Returns:
        Dictionary with LLM configuration
    """
    return get_llm_config()