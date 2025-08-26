#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

print("Attempting to execute the reproducibility_tracker.py file...")

try:
    with open('src/enhancements/reproducibility_tracker.py', 'r') as f:
        code = f.read()
    
    print(f"File read successfully. Length: {len(code)} characters")
    
    # Try to compile the code
    print("Compiling code...")
    compiled_code = compile(code, 'src/enhancements/reproducibility_tracker.py', 'exec')
    print("Code compiled successfully")
    
    # Try to execute the code
    print("Executing code...")
    namespace = {}
    exec(compiled_code, namespace)
    print("Code executed successfully")
    
    print(f"Namespace contents: {[k for k in namespace.keys() if not k.startswith('__')]}")
    
    if 'ReproducibilityTracker' in namespace:
        print("✓ ReproducibilityTracker found in namespace")
        tracker_class = namespace['ReproducibilityTracker']
        print("✓ Creating instance...")
        tracker = tracker_class()
        print("✓ Instance created successfully")
    else:
        print("✗ ReproducibilityTracker not found in namespace")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()