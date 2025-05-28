# Make the Benchmark directory a Python package

# Add root directory to Python path for package imports
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
