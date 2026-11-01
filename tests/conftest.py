import os
import matplotlib
# Ensure non-interactive backend for headless environments
if not os.getenv('MPLBACKEND'):
    matplotlib.use('Agg')

# Prevent pytest from collecting non-Python test data files
collect_ignore = [
    "test_output.txt",
    "full_test_output.txt",
]
