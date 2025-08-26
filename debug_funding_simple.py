#!/usr/bin/env python3

print("Starting debug...")

try:
    print("1. Testing basic imports...")
    import uuid
    from datetime import datetime, date, timedelta
    from typing import Dict, List, Optional, Any, Tuple
    from dataclasses import dataclass, field
    from enum import Enum
    import logging
    print("   Basic imports OK")
    
    print("2. Testing ValidationError...")
    from src.core.exceptions import ValidationError
    print("   ValidationError OK")
    
    print("3. Testing ResearcherLevel...")
    from src.data.enhanced_models import ResearcherLevel
    print("   ResearcherLevel OK")
    
    print("4. Testing enum creation...")
    class TestEnum(Enum):
        TEST = "test"
    print("   Enum creation OK")
    
    print("5. Testing dataclass creation...")
    @dataclass
    class TestClass:
        name: str = "test"
    print("   Dataclass creation OK")
    
    print("6. Testing module import...")
    import src.enhancements.funding_system
    print(f"   Module imported, dir: {dir(src.enhancements.funding_system)}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()