#!/usr/bin/env python3
"""Former name of check_deeptile_legality.py.

TileCNN was renamed DeepTile. This wrapper keeps existing commands working and
will be removed in a later release; run tools/check_deeptile_legality.py.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    print("check_tilecnn_legality.py is deprecated; use check_deeptile_legality.py",
          file=sys.stderr)
    runpy.run_path(str(Path(__file__).with_name("check_deeptile_legality.py")),
                   run_name="__main__")
