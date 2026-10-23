"""option-order-check: how much does your multiple-choice LLM eval depend on option order?"""
import os
import sys

from .core import main as _main

__version__ = "0.1.0"
DEMO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.jsonl")


def main():
    """Console entry point. `--demo` runs the bundled 12-question demo set."""
    if "--demo" in sys.argv:
        i = sys.argv.index("--demo")
        sys.argv[i] = DEMO_PATH
    _main()
