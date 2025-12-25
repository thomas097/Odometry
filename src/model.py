from __future__ import annotations

import os
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


class DepthAnything3Model:
    """
    ONNX Runtime wrapper for the DepthAnything3 depth and camera pose estimation model.
    """

    def __init__(self, session: ort.InferenceSession) -> None:
        """
        Initializes a DepthAnything3 inference wrapper.

        Args:
            session (onnxruntime.InferenceSession): Preloaded ONNX Runtime inference session.
        """
        self.session = session

    @classmethod
    def from_pretrained(
        cls, 
        model_dir: str | Path, 
        device: Literal['cpu', 'cuda'] = 'cpu', 
        quant: str | None = None
        ) -> 'DepthAnything3Model':
        """
        Loads a DepthAnything3 ONNX model from a directory.

        Args:
            model_dir (str): Path to the model directory.
            device (Literal['cpu', 'cuda'], optional): Runtime device.
            quant (str | None, optional): Quantization level of model. Defaults to None.

        Returns:
            Initialized DepthAnything3 instance.

        Raises:
            FileNotFoundError: If no compatible ONNX model is found.
        """
        suffix = '_' + quant.lower() if quant is not None else ''
        model_file = os.path.join(model_dir, f"model{suffix}.onnx")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"No model{suffix}.onnx file found in '{model_dir}'")

        # Set CUDAExecutionProvider as preferred provider if `device='cuda'`
        providers = ["CPUExecutionProvider"]
        if device.startswith('cuda'):
            providers.insert(0, "CUDAExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            path_or_bytes=model_file,
            providers=providers,
            sess_options=sess_options
        )
        return cls(session=session)
    
    def _prepare_pixel_values(self, images: Iterable[NDArray]) -> NDArray[np.float32]:
        """
        Converts input images into a model-compatible tensor.

        Args:
            images (Iterable[NDArray]): Sequence of images with shape (H, W, 3).

        Returns:
            NDArray: Tensor of shape (1, N, 3, H, W).

        Raises:
            ValueError: If image sequence is empty or images have an invalid shape.
        """
        if not images:
            raise ValueError("At least one image must be provided")

        processed_images: List[NDArray[np.float32]] = []

        for idx, img in enumerate(images):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Image {idx} must have shape (H, W, 3), got {img.shape}")

            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            elif np.issubdtype(img.dtype, np.floating):
                img = img.astype(np.float32) / np.max(img)
            else:
                raise ValueError(f"Image {idx} with dtype {img.dtype} is not understood.")

            processed_images.append(img)

        # Reshape (N, H, W, 3) -> (1, N, 3, H, W)
        imgs = np.stack(processed_images, axis=0)          
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        imgs = np.expand_dims(imgs, axis=0)

        return imgs

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
        outputs = self.session.run(
            output_names=['predicted_depth', 'confidence', 'extrinsics', 'intrinsics'],
            input_feed={
                'pixel_values': self._prepare_pixel_values(images)
                }
        )
        return DepthAnything3Output(*outputs) #type:ignore