import numpy as np
import matplotlib.pyplot as plt

# Load poses
poses = np.load("outputs/poses.npy")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

scale = 0.1

for pose in poses:
    R = pose[:, :3]
    t = pose[:, 3]

    # Camera center in world coordinates
    C = -R.T @ t

    # Camera axes in world coordinates
    axes = R.T

    # Draw camera center
    ax.scatter(C[0], C[1], C[2], c="k", s=10)

    # Draw camera axes
    ax.quiver(C[0], C[1], C[2], axes[0, 0], axes[1, 0], axes[2, 0],
              color="r", length=scale)
    ax.quiver(C[0], C[1], C[2], axes[0, 1], axes[1, 1], axes[2, 1],
              color="g", length=scale)
    ax.quiver(C[0], C[1], C[2], axes[0, 2], axes[1, 2], axes[2, 2],
              color="b", length=scale)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.show()
