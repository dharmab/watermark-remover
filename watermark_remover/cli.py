"""CLI for removing watermarks from images."""

import click
from pathlib import Path
from PIL import Image

from .detector import WatermarkDetector, create_corner_mask
from .inpainter import WatermarkInpainter


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Output path. Default: <name>_clean.<ext>"
)
@click.option(
    "--confidence",
    default=0.5,
    type=float,
    help="YOLO confidence threshold (0.0-1.0)"
)
@click.option(
    "--padding",
    default=10,
    type=int,
    help="Extra pixels around the detected watermark"
)
@click.option(
    "--fallback-corner",
    is_flag=True,
    default=True,
    help="Use bottom-right corner if YOLO detects nothing (default: enabled)"
)
@click.option(
    "--no-fallback",
    is_flag=True,
    help="Disable corner fallback"
)
@click.option(
    "--corner",
    type=click.Choice(["bottom-right", "bottom-left", "top-right", "top-left"]),
    default="bottom-right",
    help="Corner for fallback"
)
@click.option(
    "--corner-width",
    default=0.12,
    type=float,
    help="Width ratio for the corner mask (0.0-1.0)"
)
@click.option(
    "--corner-height",
    default=0.08,
    type=float,
    help="Height ratio for the corner mask (0.0-1.0)"
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Show detailed information"
)
@click.option(
    "--force-corner",
    is_flag=True,
    help="Always use the corner mask, skip YOLO detection"
)
@click.option(
    "--method",
    type=click.Choice(["lama", "opencv"]),
    default="lama",
    help="Inpainting method: lama (better quality) or opencv (faster)"
)
def main(
    input_path: str,
    output: str | None,
    confidence: float,
    padding: int,
    fallback_corner: bool,
    no_fallback: bool,
    corner: str,
    corner_width: float,
    corner_height: float,
    verbose: bool,
    force_corner: bool,
    method: str,
):
    """
    Remove watermarks from images using AI.

    Uses YOLO to detect the watermark and LaMa to remove it.
    If YOLO detects nothing, defaults to the bottom-right corner.

    Examples:

        watermark-remover image.png

        watermark-remover image.png -o clean.png --verbose

        watermark-remover image.png --force-corner --corner-width 0.15
    """
    path = Path(input_path)

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = path.parent / f"{path.stem}_clean{path.suffix}"

    if verbose:
        click.echo(f"Processing: {path}")

    # Load image
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        click.echo(f"Error loading image: {e}", err=True)
        raise SystemExit(1)

    if verbose:
        click.echo(f"Size: {image.size[0]}x{image.size[1]}")

    mask = None
    use_fallback = no_fallback is False and fallback_corner

    # Detect watermark with YOLO (unless corner is forced)
    if not force_corner:
        if verbose:
            click.echo("Detecting watermarks with YOLO...")

        try:
            detector = WatermarkDetector(confidence=confidence)
            detections = detector.detect(image)

            if detections:
                if verbose:
                    click.echo(f"Detected {len(detections)} watermark(s)")
                    for i, det in enumerate(detections):
                        click.echo(f"  {i+1}. Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}")

                mask = detector.create_mask(image.size, detections, padding)
            elif verbose:
                click.echo("YOLO did not detect any watermarks")

        except Exception as e:
            if verbose:
                click.echo(f"YOLO detection error: {e}")
            # Continue with fallback if enabled

    # Fallback: use corner
    if mask is None:
        if force_corner or use_fallback:
            if verbose:
                click.echo(f"Using corner mask: {corner}")

            mask = create_corner_mask(
                image.size,
                corner=corner,
                width_ratio=corner_width,
                height_ratio=corner_height,
                padding=padding,
            )
        else:
            click.echo("No watermarks detected and fallback is disabled", err=True)
            raise SystemExit(1)

    # Inpainting
    if verbose:
        click.echo(f"Applying inpainting with {method.upper()}...")

    try:
        inpainter = WatermarkInpainter(method=method)
        result = inpainter.inpaint(image, mask)
    except Exception as e:
        click.echo(f"Inpainting error: {e}", err=True)
        raise SystemExit(1)

    # Save result
    try:
        result.save(output_path)
        click.echo(f"Saved: {output_path}")
    except Exception as e:
        click.echo(f"Error saving: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
