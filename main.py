import numpy as np
from tqdm import tqdm
from src import (
    DepthAnything3Model, 
    PoseEstimator,
    frames_from_video
)

BATCH_SIZE = 3

model = DepthAnything3Model.from_pretrained(
    "checkpoints/da3-small",
    device='cpu',
    quant='uint8'
    )
estimator = PoseEstimator(model, context_size=5)

image_batch = []
poses = []

for image in tqdm(frames_from_video("assets/living_room.mp4", frames_per_second=0.5)):
    if len(image_batch) < BATCH_SIZE:
        image_batch.append(image)
        continue

    pose = estimator.infer(image_batch)
    poses.append(pose)

    if len(poses) == 12:
        break

with open("outputs/poses.npy", mode='wb') as file: 
    np.save(file=file, arr=np.concatenate(poses, axis=0))