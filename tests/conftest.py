"""Shared pytest configuration and fixtures for the GRPO test suite."""

import os
import sys

# Ensure src/ is on sys.path so that both ``import grpo`` and legacy bare
# imports (``import grpo_trainer``) resolve correctly.
_SRC_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "src")
_SRC_DIR = os.path.abspath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
