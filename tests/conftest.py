"""Pytest configuration for tugas-akhir project."""

import sys
import os
from pathlib import Path


# Add the project root to Python path so pytest can find src module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.planning.repack_trial import RepackTrial


@pytest.fixture
def repacker():
	"""Legacy fixture for repack strategy tests."""
	return RepackTrial(container_dims=(59, 23, 23), time_limit=1.0)


@pytest.fixture
def test_items():
	"""Sample items for repack tests."""
	return [(10, 10, 10), (8, 8, 8), (6, 6, 6)]


@pytest.fixture
def test_positions():
	"""Sample positions for repack tests."""
	return [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
