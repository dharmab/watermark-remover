"""Watermark Remover - Remove watermarks using YOLO + LaMa."""

__version__ = "0.1.0"

from .detector import WatermarkDetector
from .inpainter import WatermarkInpainter

__all__ = ["WatermarkDetector", "WatermarkInpainter"]
