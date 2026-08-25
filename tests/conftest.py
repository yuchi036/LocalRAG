import os
import sys

# Make the localrag package importable when running `pytest` from the repo root
# without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
