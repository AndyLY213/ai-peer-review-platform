#!/usr/bin/env python3

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    print("Testing individual imports...")
    
    print("1. Testing json import...")
    import json
    print("   ✓ json imported")
    
    print("2. Testing uuid import...")
    import uuid
    print("   ✓ uuid imported")
    
    print("3. Testing datetime imports...")
    from datetime import date, datetime, timedelta
    print("   ✓ datetime imports successful")
    
    print("4. Testing dataclasses import...")
    from dataclasses import dataclass, asdict
    print("   ✓ dataclasses imported")
    
    print("5. Testing enum import...")
    from enum import Enum
    print("   ✓ enum imported")
    
    print("6. Testing pathlib import...")
    from pathlib import Path
    print("   ✓ pathlib imported")
    
    print("7. Testing typing imports...")
    from typing import Dict, List, Optional, Tuple, Any, Set
    print("   ✓ typing imports successful")
    
    print("8. Testing math import...")
    import math
    print("   ✓ math imported")
    
    print("9. Testing custom exceptions import...")
    from src.core.exceptions import ValidationError, PeerReviewError
    print("   ✓ custom exceptions imported")
    
    print("10. Testing enum definitions...")
    
    class ReplicationOutcome(Enum):
        SUCCESS = "success"
        FAILURE = "failure"
        PARTIAL = "partial"
        INCONCLUSIVE = "inconclusive"
    
    print("   ✓ ReplicationOutcome enum created")
    
    class QuestionablePractice(Enum):
        P_HACKING = "p_hacking"
        DATA_FABRICATION = "data_fabrication"
    
    print("   ✓ QuestionablePractice enum created")
    
    print("11. Testing dataclass creation...")
    
    @dataclass
    class TestClass:
        test_field: str
        
        def __post_init__(self):
            if not self.test_field:
                raise ValidationError("test_field", self.test_field, "non-empty string")
    
    print("   ✓ Test dataclass created")
    
    test_instance = TestClass("test")
    print("   ✓ Test dataclass instance created")
    
    print("All imports and basic functionality work. The issue might be in the file structure.")
    
except Exception as e:
    print(f"Error during testing: {e}")
    traceback.print_exc()