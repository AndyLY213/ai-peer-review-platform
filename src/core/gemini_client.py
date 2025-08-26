"""
Gemini API Client for LLM Integration

This module provides a client for Google's Gemini API to be used with the
peer review simulation system.
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")
load_dotenv(".env.local", override=True)


class GeminiClient:
    """Client for Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key (if not provided, will use GEMINI_API_KEY env var)
            model: Model name (if not provided, will use GEMINI_MODEL env var)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
    
    def generate_content(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: The input prompt
            temperature: Temperature for generation (0.0 to 1.0)
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If API call fails
        """
        url = f"{self.base_url}/models/{self.model}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": 2048,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the generated text from the response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            raise Exception(f"Unexpected response format: {result}")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Gemini API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Gemini API response: {str(e)}")
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Generate a chat completion using Gemini API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Temperature for generation
            
        Returns:
            Generated response text
        """
        # Convert chat messages to a single prompt for Gemini
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add a final prompt for the assistant to respond
        prompt_parts.append("Assistant:")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        return self.generate_content(full_prompt, temperature)
    
    def test_connection(self) -> bool:
        """
        Test the connection to Gemini API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.generate_content("Hello, this is a test. Please respond with 'Test successful.'")
            return "test successful" in response.lower()
        except Exception:
            return False


def create_gemini_client() -> GeminiClient:
    """
    Create a Gemini client with configuration from environment variables.
    
    Returns:
        Configured GeminiClient instance
    """
    return GeminiClient()


# Test function
if __name__ == "__main__":
    try:
        client = create_gemini_client()
        print(f"Testing Gemini client with model: {client.model}")
        
        if client.test_connection():
            print("✓ Gemini API connection successful!")
            
            # Test chat completion
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
            
            response = client.chat_completion(messages)
            print(f"Chat response: {response}")
            
        else:
            print("✗ Gemini API connection failed!")
            
    except Exception as e:
        print(f"Error testing Gemini client: {e}")