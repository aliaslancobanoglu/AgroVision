import sys
from pathlib import Path

# the project scripts live in the repository root (no package), so put it on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
