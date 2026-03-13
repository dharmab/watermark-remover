"""Watermark detection using YOLOv8."""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


class WatermarkDetector:
    """Watermark detector using YOLOv8."""

    def __init__(self, model_path: str | None = None, confidence: float = 0.5):
        """
        Initialize the detector.

        Args:
            model_path: Path to the YOLO model. If None, uses yolov8n.pt
            confidence: Minimum confidence threshold (0.0 - 1.0)
        """
        from ultralytics import YOLO

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use base YOLOv8 nano model
            self.model = YOLO("yolov8n.pt")

        self.confidence = confidence

    def detect(self, image: Image.Image) -> list[dict]:
        """
        Detect objects in the image that could be watermarks.

        Args:
            image: PIL Image in RGB mode

        Returns:
            List of detections with bbox [x1, y1, x2, y2] and confidence
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf),
                    "class": int(box.cls) if box.cls is not None else None,
                })

        return detections

    def create_mask(
        self,
        image_size: tuple[int, int],
        detections: list[dict],
        padding: int = 10,
    ) -> Image.Image:
        """
        Create a binary mask from the detections.

        Args:
            image_size: Image size (width, height)
            detections: List of detections with bbox
            padding: Extra pixels around each detection

        Returns:
            PIL mask in L mode (white = area to remove)
        """
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Apply padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_size[0], x2 + padding)
            y2 = min(image_size[1], y2 + padding)
            draw.rectangle([x1, y1, x2, y2], fill=255)

        return mask


def create_corner_mask(
    image_size: tuple[int, int],
    corner: str = "bottom-right",
    width_ratio: float = 0.15,
    height_ratio: float = 0.08,
    padding: int = 10,
) -> Image.Image:
    """
    Create a mask in a corner of the image (fallback when YOLO doesn't detect).

    Args:
        image_size: Image size (width, height)
        corner: Corner to mask ("bottom-right", "bottom-left", "top-right", "top-left")
        width_ratio: Ratio of image width for the mask
        height_ratio: Ratio of image height for the mask
        padding: Extra margin pixels

    Returns:
        PIL mask in L mode
    """
    width, height = image_size
    mask_width = int(width * width_ratio)
    mask_height = int(height * height_ratio)

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    if corner == "bottom-right":
        x1 = width - mask_width - padding
        y1 = height - mask_height - padding
        x2 = width - padding
        y2 = height - padding
    elif corner == "bottom-left":
        x1 = padding
        y1 = height - mask_height - padding
        x2 = mask_width + padding
        y2 = height - padding
    elif corner == "top-right":
        x1 = width - mask_width - padding
        y1 = padding
        x2 = width - padding
        y2 = mask_height + padding
    elif corner == "top-left":
        x1 = padding
        y1 = padding
        x2 = mask_width + padding
        y2 = mask_height + padding
    else:
        raise ValueError(f"Invalid corner: {corner}")

    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask
