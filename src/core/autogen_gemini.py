"""
AutoGen-compatible Gemini client.

This module provides an AutoGen-compatible interface for Google's Gemini API.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.gemini_client import GeminiClient

# Load environment variables
load_dotenv("config.env")
load_dotenv(".env.local", override=True)


class AutoGenGeminiClient:
    """
    AutoGen-compatible wrapper for Gemini API.
    
    This class provides an interface that's compatible with AutoGen's
    expected LLM client behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen Gemini client.
        
        Args:
            config: Configuration dictionary with model, api_key, etc.
        """
        self.config = config
        self.model = config.get("model", "gemini-2.0-flash")
        self.api_key = config.get("api_key")
        self.temperature = config.get("temperature", 0.7)
        
        # Initialize the underlying Gemini client
        self.gemini_client = GeminiClient(
            api_key=self.api_key,
            model=self.model
        )
    
    def create(self, **kwargs) -> Dict[str, Any]:
        """
        Create a completion using Gemini API.
        
        This method mimics the OpenAI client interface that AutoGen expects.
        
        Args:
            **kwargs: Keyword arguments including messages, temperature, etc.
            
        Returns:
            Dictionary with completion response in OpenAI-compatible format
        """
        messages = kwargs.get("messages", [])
        temperature = kwargs.get("temperature", self.temperature)
        
        try:
            # Use the Gemini client to generate response
            response_text = self.gemini_client.chat_completion(messages, temperature)
            
            # Return in OpenAI-compatible format
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Gemini doesn't provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "model": self.model
            }
            
        except Exception as e:
            # Return error in compatible format
            return {
                "error": {
                    "message": str(e),
                    "type": "gemini_api_error"
                }
            }


class GeminiLLMConfig:
    """
    Configuration class for Gemini LLM integration with AutoGen.
    """
    
    @staticmethod
    def create_config() -> Dict[str, Any]:
        """
        Create AutoGen-compatible configuration for Gemini.
        
        Returns:
            Configuration dictionary for AutoGen
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        
        # Create a custom client configuration
        config = {
            "config_list": [
                {
                    "model": model,
                    "api_key": api_key,
                    "api_type": "google",  # Use 'google' instead of 'gemini'
                    "base_url": None,  # Not used for Gemini
                }
            ],
            "temperature": temperature,
            "timeout": timeout,
        }
        
        return config


def create_autogen_gemini_config() -> Dict[str, Any]:
    """
    Create AutoGen-compatible Gemini configuration.
    
    Returns:
        Configuration dictionary for AutoGen
    """
    return GeminiLLMConfig.create_config()


# Test function
if __name__ == "__main__":
    try:
        print("Testing AutoGen Gemini integration...")
        
        config = create_autogen_gemini_config()
        print(f"Config created: {config}")
        
        # Test the client directly
        client_config = config["config_list"][0]
        client = AutoGenGeminiClient(client_config)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Japan?"}
        ]
        
        response = client.create(messages=messages)
        print(f"Response: {response}")
        
        if "choices" in response:
            print("✓ AutoGen Gemini client working!")
        else:
            print("✗ AutoGen Gemini client failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()