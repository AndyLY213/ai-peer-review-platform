#!/usr/bin/env python3
"""
Simple test script to verify Ollama connection and model availability.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")

def test_ollama_connection():
    """Test basic Ollama connection."""
    base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    
    print(f"Testing Ollama connection...")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print("-" * 50)
    
    try:
        # Test 1: Check if Ollama server is running
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            models = response.json().get("models", [])
            print(f"Available models: {len(models)}")
            for model_info in models:
                print(f"  - {model_info.get('name', 'Unknown')}")
        else:
            print(f"❌ Ollama server responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama server")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False
    
    try:
        # Test 2: Try to use the model with ollama library
        import ollama
        
        print(f"\nTesting model '{model}'...")
        
        # Create client with explicit host
        client = ollama.Client(host=base_url)
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Hello! Please respond with just 'OK' to confirm you're working."}],
            options={"num_predict": 10}
        )
        
        if response and 'message' in response:
            print(f"✅ Model response: {response['message']['content']}")
            return True
        else:
            print("❌ No response from model")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    if success:
        print("\n🎉 Ollama is working correctly!")
    else:
        print("\n💥 Ollama connection failed!")