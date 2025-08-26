#!/usr/bin/env python3

from src.enhancements.funding_system import FundingSystem

try:
    print("Creating FundingSystem...")
    fs = FundingSystem()
    print(f"FundingSystem created")
    print(f"Has _create_default_agencies: {hasattr(fs, '_create_default_agencies')}")
    print(f"Agencies before: {len(fs.agencies)}")
    
    if hasattr(fs, '_create_default_agencies'):
        print("Calling _create_default_agencies manually...")
        fs._create_default_agencies()
        print(f"Agencies after: {len(fs.agencies)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()