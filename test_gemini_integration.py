#!/usr/bin/env python3
"""
Test Gemini integration with the funding system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.llm_config import get_llm_config, get_llm_provider_info
from src.core.gemini_client import create_gemini_client

def test_llm_config():
    """Test LLM configuration loading."""
    print("Testing LLM configuration...")
    
    try:
        config = get_llm_config()
        provider_info = get_llm_provider_info()
        
        print(f"Provider: {provider_info['provider']}")
        print(f"Model: {provider_info['model']}")
        print(f"Temperature: {provider_info['temperature']}")
        print(f"Config: {config}")
        
        return True
    except Exception as e:
        print(f"Error loading LLM config: {e}")
        return False

def test_gemini_client():
    """Test Gemini client directly."""
    print("\nTesting Gemini client...")
    
    try:
        client = create_gemini_client()
        
        if client.test_connection():
            print("✓ Gemini client connection successful!")
            
            # Test a simple generation
            response = client.generate_content("What is 2+2? Answer briefly.")
            print(f"Response: {response}")
            
            return True
        else:
            print("✗ Gemini client connection failed!")
            return False
            
    except Exception as e:
        print(f"Error testing Gemini client: {e}")
        return False

def test_funding_system():
    """Test funding system basic functionality."""
    print("\nTesting funding system...")
    
    try:
        from src.enhancements.funding_system import FundingSystem
        
        funding_system = FundingSystem()
        
        # Test basic functionality
        agencies = funding_system.get_agencies_by_type(funding_system.agencies[list(funding_system.agencies.keys())[0]].agency_type)
        print(f"Found {len(agencies)} agencies")
        
        # Test creating a cycle
        if agencies:
            agency = agencies[0]
            cycle_id = funding_system.create_funding_cycle(
                agency_id=agency.agency_id,
                cycle_name="Test Cycle 2024"
            )
            print(f"Created funding cycle: {cycle_id}")
            
            cycle = funding_system.get_cycle(cycle_id)
            print(f"Cycle details: {cycle.cycle_name}, Status: {cycle.status}")
        
        print("✓ Funding system basic functionality works!")
        return True
        
    except Exception as e:
        print(f"Error testing funding system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Gemini Integration Test ===")
    
    success = True
    
    success &= test_llm_config()
    success &= test_gemini_client()
    success &= test_funding_system()
    
    if success:
        print("\n✓ All tests passed! Gemini integration is working.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")