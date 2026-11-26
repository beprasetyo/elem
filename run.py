"""Run the configured FEM stress-integral pipeline with ONE command: python run.py.

Edit config/project.json, config/search_domain.json, and config/materials.json.
All stages use disk artifacts of previous stages; no hidden CLI arguments.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from kimlee.runner import run_project

if __name__ == '__main__':
    run_project(ROOT)
