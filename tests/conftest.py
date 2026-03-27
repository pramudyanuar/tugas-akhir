"""Pytest configuration for tugas-akhir project."""

import sys
import os
from pathlib import Path

# Add the project root to Python path so pytest can find src module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
