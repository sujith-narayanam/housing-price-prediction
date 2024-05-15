# Add vendored libs to the path
import os
import sys

__version__ = 0.3

_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_HERE)
