import os
import sys

# Ensure project root (rl_vo) is importable during pytest collection
ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
