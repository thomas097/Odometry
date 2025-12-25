import onnxruntime as ort
import numpy as np

# -------------------------
# Load ONNX model
# -------------------------
model_path = "checkpoints/da3-small/model.onnx"

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"], # or CUDAExecutionProvider if available
    sess_options=sess_options
)

# Inspect inputs (optional, but useful)
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

# -------------------------
# Prepare input images
# -------------------------
# images: list of numpy arrays, each (H, W, 3), RGB
# Example:
# images = [img1, img2, img3]

def prepare_pixel_values(images):
    """
    images: list of (H, W, 3) numpy arrays
    returns: (1, num_images, 3, H, W) float32
    """
    # Convert to float32
    imgs = [img.astype(np.float32) for img in images]

    # Optional normalization (depends on your model!)
    # Uncomment if your model expects normalized inputs
    # imgs = [img / 255.0 for img in imgs]

    # Stack -> (num_images, H, W, 3)
    imgs = np.stack(imgs, axis=0)

    # HWC -> CHW
    imgs = np.transpose(imgs, (0, 3, 1, 2))

    # Add batch dimension
    imgs = np.expand_dims(imgs, axis=0)

    return imgs


from extract_frames import extract_and_resize_frames

# Load frames from video
images = extract_and_resize_frames(
    video_path="assets/living_room.mp4",
    target_width=320
    )
print("Loaded images!")

pixel_values = prepare_pixel_values(images[:3])
print("Preprocessed images!")

# -------------------------
# Run inference
# -------------------------
inputs = {
    "pixel_values": pixel_values
}

outputs = session.run(None, inputs)
print("Ran session!")

# -------------------------
# Unpack outputs
# -------------------------
output_names = [o.name for o in session.get_outputs()]
outputs = dict(zip(output_names, outputs))

predicted_depth = outputs["predicted_depth"]
confidence = outputs["confidence"]
extrinsics = outputs["extrinsics"]
intrinsics = outputs["intrinsics"]

# -------------------------
# Print shapes
# -------------------------
print("predicted_depth:", predicted_depth.shape)
print("confidence:", confidence.shape)
print("extrinsics:", extrinsics.shape)
print("intrinsics:", intrinsics.shape)
