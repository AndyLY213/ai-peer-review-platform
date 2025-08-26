#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    print("Attempting to import the module...")
    import src.enhancements.reproducibility_tracker as rt
    print(f"Module imported successfully. Contents: {[x for x in dir(rt) if not x.startswith('_')]}")
    
    print("Attempting to import ReproducibilityTracker class...")
    from src.enhancements.reproducibility_tracker import ReproducibilityTracker
    print("ReproducibilityTracker imported successfully!")
    
    print("Creating instance...")
    tracker = ReproducibilityTracker()
    print("Instance created successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()