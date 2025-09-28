import os
import sys

# Ensure project root (rl_vo) is on sys.path when running scripts/* directly
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
