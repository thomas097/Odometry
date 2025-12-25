from src.model import DepthAnything3
from src.video import extract_and_resize_frames

model = DepthAnything3.from_pretrained(
    "checkpoints/da3-small",
    device='cpu',
    quant='uint8',
    normalize=False
)

images = extract_and_resize_frames(
    video_path="assets/living_room.mp4",
    target_width=320
)

output = model.infer(images[:3])

print(output.predicted_depth.shape)
print(output.confidence.shape)
print(output.extrinsics.shape)
print(output.intrinsics.shape)
