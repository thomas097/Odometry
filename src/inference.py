import numpy as np
from .model import DepthAnything3Model, DepthAnything3Output
from collections import deque
from typing import List
from numpy.typing import NDArray


_NULL_POSE = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0]],
    dtype=np.float32
)


class PoseEstimator:
    def __init__(self, model: DepthAnything3Model, context_size: int = 5) -> None:
        self._model = model
        self._context_size = context_size
        self._queue = deque(maxlen=context_size)

        # We maintain a list of previous solves
        self._n_images = 0
        self._cache = []

    def _estimate_sim3_from_poses(self, X: NDArray, Y: NDArray) -> dict[str, NDArray]:
        """Estimate a similarity transform (Sim(3)) aligning two sets of camera poses.

        Given corresponding camera poses X_i and Y_i living in different coordinate
        frames, this function estimates a single similarity transform (s, R, t)
        such that:

            Y_i ≈ [sR | t] * X_i

        where scale affects translations but not rotations.

        Args:
            X (np.ndarray): Array of shape (N, 3, 4) representing source poses.
                Each pose is [R | t], where R is 3x3 and t is 3x1.
            Y (np.ndarray): Array of shape (N, 3, 4) representing target poses.
                Must be in one-to-one correspondence with X.

        Returns:
            dict: A dictionary with the following keys:
                - 'scale' (float): Estimated scale factor.
                - 'rotation' (np.ndarray): Rotation matrix of shape (3, 3).
                - 'translation' (np.ndarray): Translation vector of shape (3,).
                - 'matrix' (np.ndarray): Homogeneous 4x4 Sim(3) transform.

        Raises:
            ValueError: If fewer than two poses are provided.
        """
        if X.shape[0] < 2:
            raise ValueError("At least two pose correspondences are required.")

        N = X.shape[0]

        # Split rotations and translations
        RX = X[:, :3, :3]
        tX = X[:, :3, 3]
        RY = Y[:, :3, :3]
        tY = Y[:, :3, 3]

        # --- Estimate rotation ---
        M = np.zeros((3, 3))
        for i in range(N):
            M += RY[i] @ RX[i].T

        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Ensure proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        # --- Estimate scale ---
        tX_mean = tX.mean(axis=0)
        tY_mean = tY.mean(axis=0)

        tX_centered = (R @ (tX - tX_mean).T).T
        tY_centered = tY - tY_mean

        numerator = np.sum(tY_centered * tX_centered)
        denominator = np.sum(tX_centered ** 2)
        scale = numerator / denominator

        # --- Estimate translation ---
        t = tY_mean - scale * R @ tX_mean

        # Homogeneous transform
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = t

        return {
            "scale": scale,
            "rotation": R,
            "translation": t,
            "matrix": T,
        }

    def _apply_sim3_to_pose(self, poses: NDArray, sim3: dict) -> NDArray:
        """Apply a Sim(3) transform to one or more camera poses.

        The similarity transform is applied as:
            R_Y = R * R_X
            t_Y = s * R * t_X + t

        Args:
            pose (np.ndarray): Input to transform of shape (N, 3, 4).
            sim3 (dict): Sim(3) transform dictionary returned by
                `estimate_sim3_from_poses`.

        Returns:
            np.ndarray: transformed camera poses

        Raises:
            ValueError: If the input shape is unsupported.
        """
        s = sim3["scale"]
        R = sim3["rotation"]
        t = sim3["translation"]

        if poses.ndim == 3 and poses.shape[1:] == (3, 4):
            RX = poses[:, :, :3]
            tX = poses[:, :, 3]

            RY = R @ RX
            tY = (s * (R @ tX.T)).T + t

            return np.concatenate([RY, tY[:, :, None]], axis=2)
        else:
            raise ValueError(
                "Unsupported pose shape. Expected shape (N, 3, 4)."
            )

    def infer(self, images: List[NDArray] | NDArray) -> NDArray[np.float32]:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        # Chec if the new images fit inside the queue without discarding
        if len(self._queue) + len(images) <= self._context_size:
            pass
        
        # Otherwise, check if a previous context of at least two images can be maintained
        elif (self._context_size - len(images)) < 2:
            raise ValueError(
                f"Number of images ({len(images)}) needs to be less than {self._context_size - 1} with context of size {self._context_size}!"
                )
            
        self._queue.extend(images)
        self._n_images += len(images)
        
        result = self._model.infer(images=self._queue)
        extrinsics = result.extrinsics[0]

        # Base case: images still fit in context
        if self._n_images <= self._context_size:
            self._cache = list(extrinsics)
            return extrinsics[-len(images):].copy()

        # Estimate transform from new reconstruction to previous one
        src_old_poses = extrinsics[:-len(images)]
        tgt_old_poses = np.stack(self._cache[-src_old_poses.shape[0]:], axis=0)

        sim3 = self._estimate_sim3_from_poses(X=src_old_poses, Y=tgt_old_poses)

        # Transform poses of new images to previous coordinate system
        src_new_poses = extrinsics[-len(images):]
        tgt_new_poses = self._apply_sim3_to_pose(src_new_poses, sim3=sim3)

        self._cache.extend(list(tgt_new_poses))
        return src_new_poses
