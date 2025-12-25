import onnx

model = onnx.load("checkpoints/da3-small/model.onnx", load_external_data=True)
onnx.save(
    model,
    "checkpoints/da3-small/model_no_external.onnx",
    save_as_external_data=False
)