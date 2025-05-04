"""
Main entry point for the Peer Review Simulation System.

This script serves as the primary entry point for running the simulation.
"""

import os
import sys

# Add the project root directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the simulation module
from src.simulation.peer_review_simulation import main

if __name__ == "__main__":
    # Run the simulation
    main() 