"""Image inpainting using LaMa or OpenCV."""

import os
from PIL import Image
import numpy as np

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class WatermarkInpainter:
    """Remove watermarks using LaMa or OpenCV."""

    def __init__(self, method: str = "lama"):
        """
        Initialize the inpainter.

        Args:
            method: "lama" for better quality, "opencv" for compatibility
        """
        self.method = method
        self._lama_model = None

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Remove the watermark from the image using inpainting.

        Args:
            image: PIL Image in RGB mode
            mask: PIL mask in L mode (white = area to remove)

        Returns:
            PIL Image with the watermark removed
        """
        if self.method == "lama":
            return self._inpaint_lama(image, mask)
        else:
            return self._inpaint_opencv(image, mask)

    def _get_lama_model(self):
        """Download and load the LaMa model (TorchScript)."""
        if self._lama_model is not None:
            return self._lama_model

        import torch
        from torch.hub import download_url_to_file, get_dir

        # Model path
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "big-lama.pt")

        # Download if not present
        if not os.path.exists(model_path):
            url = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
            print("Downloading LaMa model...")
            download_url_to_file(url, model_path, progress=True)

        # Load TorchScript model
        print("Loading LaMa model...")
        model = torch.jit.load(model_path, map_location=torch.device("cpu"))
        model.eval()

        self._lama_model = model
        return model

    def _inpaint_lama(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpainting with LaMa."""
        import torch

        model = self._get_lama_model()

        # Preprocess image
        img_np = np.array(image.convert("RGB")).astype(np.float32)
        mask_np = np.array(mask.convert("L")).astype(np.float32)

        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0) / 255.0

        # Binarize mask (>0.5 = area to remove)
        mask_tensor = (mask_tensor > 0.5).float()

        # Pad to multiple of 8 (required by LaMa's encoder/decoder architecture)
        _, _, h, w = img_tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # Inference
        with torch.no_grad():
            result = model(img_tensor, mask_tensor)

        # Crop back to original size and postprocess
        result = result[:, :, :h, :w]
        result_np = result[0].permute(1, 2, 0).cpu().numpy()
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(result_np)

    def _inpaint_opencv(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpainting with OpenCV (fast fallback)."""
        import cv2

        # Convert to numpy
        img_np = np.array(image)
        mask_np = np.array(mask)

        # OpenCV uses BGR
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Inpainting with Navier-Stokes
        result_bgr = cv2.inpaint(img_bgr, mask_np, inpaintRadius=7, flags=cv2.INPAINT_NS)

        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(result_rgb)


class OpenCVInpainter:
    """Simple inpainter using OpenCV."""

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Remove the watermark using OpenCV inpainting."""
        import cv2

        img_np = np.array(image)
        mask_np = np.array(mask)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.inpaint(img_bgr, mask_np, inpaintRadius=7, flags=cv2.INPAINT_NS)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(result_rgb)
