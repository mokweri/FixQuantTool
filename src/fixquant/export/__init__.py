from .deeptile_exporter import DeepTileGraphExporter

# Former name, kept for existing callers.
TileCNNGraphExporter = DeepTileGraphExporter

__all__ = ['DeepTileGraphExporter', 'TileCNNGraphExporter']
