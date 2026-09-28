"""Former name of :mod:`fixquant.export.deeptile_exporter`.

TileCNN was renamed DeepTile. This module keeps existing imports working and
will be removed in a later release; import ``fixquant.export.deeptile_exporter``.
"""
import warnings

from .deeptile_exporter import *  # noqa: F401,F403
from .deeptile_exporter import (  # noqa: F401
    DeepTileGraphExporter,
    _check_shift_legality,
    _deeptile_conv2d,
    _deeptile_gap,
    _deeptile_linear,
    _deeptile_maxpool,
    _deeptile_residual_add,
    _relu6_max,
)

TileCNNGraphExporter = DeepTileGraphExporter
_tilecnn_conv2d = _deeptile_conv2d
_tilecnn_linear = _deeptile_linear
_tilecnn_residual_add = _deeptile_residual_add
_tilecnn_maxpool = _deeptile_maxpool
_tilecnn_gap = _deeptile_gap

warnings.warn(
    "fixquant.export.tilecnn_exporter is deprecated; "
    "use fixquant.export.deeptile_exporter",
    DeprecationWarning,
    stacklevel=2,
)
