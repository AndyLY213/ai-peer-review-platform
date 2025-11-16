"""
Script to run simulation with automated inputs
"""
import subprocess
import sys

# Prepare inputs: mode 2, 10 rounds, 10 interactions
inputs = "2\n10\n10\n"

# Run the simulation
process = subprocess.Popen(
    [sys.executable, "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# Send inputs and get output
output, _ = process.communicate(input=inputs)

# Write output to file
with open("simulation_run.md", "w", encoding='utf-8') as f:
    f.write(output)

print("Simulation complete. Output saved to simulation_run.md")
