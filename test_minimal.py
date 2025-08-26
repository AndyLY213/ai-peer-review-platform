#!/usr/bin/env python3

print("Starting minimal test...")

try:
    from src.core.logging_config import get_logger
    print("✓ Logger imported")
    logger = get_logger(__name__)
    print("✓ Logger created")
except Exception as e:
    print(f"✗ Logger failed: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")