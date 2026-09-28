#!/usr/bin/env python3
"""Former name of export_deeptile_graph.py.

TileCNN was renamed DeepTile. This wrapper keeps existing commands working and
will be removed in a later release; run tools/export_deeptile_graph.py.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    print("export_tilecnn_graph.py is deprecated; use export_deeptile_graph.py",
          file=sys.stderr)
    runpy.run_path(str(Path(__file__).with_name("export_deeptile_graph.py")),
                   run_name="__main__")
