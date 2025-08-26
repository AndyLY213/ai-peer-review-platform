#!/usr/bin/env python3

print("Testing imports...")

try:
    from src.core.exceptions import ValidationError, SystemError
    print("✓ Core exceptions imported")
except Exception as e:
    print(f"✗ Core exceptions failed: {e}")

try:
    from src.core.logging_config import get_logger
    print("✓ Logging config imported")
except Exception as e:
    print(f"✗ Logging config failed: {e}")

try:
    from src.data.enhanced_models import EnhancedResearcher, PublicationRecord
    print("✓ Enhanced models imported")
except Exception as e:
    print(f"✗ Enhanced models failed: {e}")

print("All imports tested.")