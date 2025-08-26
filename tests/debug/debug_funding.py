#!/usr/bin/env python3

import sys
import traceback

try:
    print("Importing funding_system module...")
    import src.enhancements.funding_system as fs
    print(f"Module imported successfully. Available attributes: {dir(fs)}")
    
    print("Trying to access FundingSystem class...")
    if hasattr(fs, 'FundingSystem'):
        print("FundingSystem class found!")
        system = fs.FundingSystem()
        print("FundingSystem instance created successfully!")
    else:
        print("FundingSystem class not found in module")
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()