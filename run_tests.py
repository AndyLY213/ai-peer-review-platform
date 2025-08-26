#!/usr/bin/env python3
"""
Test runner for the AI Peer Review Platform.

This script provides an easy way to run different types of tests.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}")
        print(f"Exit code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for AI Peer Review Platform")
    parser.add_argument("--type", choices=["unit", "integration", "all"], default="all",
                       help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true",
                       help="Run with coverage report")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run a quick test (just unit tests)")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        base_cmd.append("-v")
    
    if args.coverage:
        base_cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    success = True
    
    if args.quick:
        # Quick test - just run a simple test
        cmd = base_cmd + ["tests/unit/test_simple.py", "-x"]
        success &= run_command(cmd, "Quick Test (Simple Unit Test)")
    
    elif args.type == "unit":
        # Run unit tests
        cmd = base_cmd + ["tests/unit/"]
        success &= run_command(cmd, "Unit Tests")
    
    elif args.type == "integration":
        # Run integration tests
        cmd = base_cmd + ["tests/integration/"]
        success &= run_command(cmd, "Integration Tests")
    
    else:  # all
        # Run unit tests first
        cmd = base_cmd + ["tests/unit/"]
        success &= run_command(cmd, "Unit Tests")
        
        # Then integration tests
        if success:
            cmd = base_cmd + ["tests/integration/"]
            success &= run_command(cmd, "Integration Tests")
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check the output above.")
    print(f"{'='*60}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())