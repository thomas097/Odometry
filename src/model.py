from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


@dataclass(frozen=True)
class DepthAnything3Output:
    """
    Container for DepthAnything3 inference outputs.
    """
    predicted_depth: NDArray[np.float32]
    confidence: NDArray[np.float32]
    extrinsics: NDArray[np.float32]
    intrinsics: NDArray[np.float32]


class DepthAnything3:
    """
    ONNX Runtime wrapper for the DepthAnything3 depth and camera pose estimation model.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        normalize: bool = False
        ) -> None:
        """
        Initializes a DepthAnything3 inference wrapper.

        Args:
            session (onnxruntime.InferenceSession): Preloaded ONNX Runtime inference session.
            normalize (bool, optional): Whether to normalize input images.
        """
        self.session = session
        self.normalize = normalize

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, device: Literal['cpu', 'cuda'] = 'cpu', normalize: bool = False) -> "DepthAnything3":
        """
        Loads a DepthAnything3 ONNX model from a directory.

        The directory must contain a `.onnx` file and a corresponding `.onnx_data` file.

        Args:
            model_dir (str): Path to the model directory.
            device (Literal['cpu', 'cuda'], optional): Runtime device.
            normalize (bool, optional): Whether to normalize inputs by dividing by 255.0.

        Returns:
            Initialized DepthAnything3 instance.

        Raises:
            FileNotFoundError: If no ONNX model is found or associated `.onnx_data` file is missing.
        """
        model_dir = Path(model_dir)

        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in '{model_dir}'")
        
        if not list(model_dir.glob("*.onnx_data")):
            raise FileNotFoundError(f"No .onnx_data file found in '{model_dir}'")

        # Set CUDAExecutionProvider as preferred provider if `device='cuda'`
        providers = ["CPUExecutionProvider"]
        if device.startswith('cuda'):
            providers.insert(0, "CUDAExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_files[0].as_posix(),
            providers=providers,
            sess_options=sess_options
        )

        return cls(session=session, normalize=normalize)

    def infer(self, images: Iterable[NDArray[np.uint8]]) -> DepthAnything3Output:
        """
        Runs depth inference on a sequence of images.

        Args:
            images: Sequence of RGB images with shape (H, W, 3).

        Returns:
            DepthAnything3Output: Object containing predicted depth, confidence,
                camera extrinsics, and intrinsics.

        Raises:
            ValueError: If no images are provided or input shapes are invalid.
        """
        pixel_values = self._prepare_pixel_values(images)
        outputs = self.session.run(
            output_names=['predicted_depth', 'confidence', 'extrinsics', 'intrinsics'],
            input_feed={
                'pixel_values': pixel_values
                }
        )
        return DepthAnything3Output(*outputs)

    def _prepare_pixel_values(self, images: Iterable[NDArray]) -> NDArray[np.float32]:
        """
        Converts input images into a model-compatible tensor.

        Args:
            images: Sequence of images with shape (H, W, 3).

        Returns:
            Float32 tensor of shape (1, N, 3, H, W).

        Raises:
            ValueError: If images are empty or have invalid shapes.
        """
        images = list(images)
        if not images:
            raise ValueError("At least one image must be provided")

        processed: List[NDArray[np.float32]] = []

        for idx, img in enumerate(images):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"Image {idx} must have shape (H, W, 3), got {img.shape}"
                )

            img = img.astype(np.float32)
            if self.normalize:
                img /= 255.0

            processed.append(img)

        imgs = np.stack(processed, axis=0)          # (N, H, W, 3)
        imgs = np.transpose(imgs, (0, 3, 1, 2))     # (N, 3, H, W)
        imgs = np.expand_dims(imgs, axis=0)         # (1, N, 3, H, W)

        return imgs
