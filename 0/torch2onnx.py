import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import typer
import torch

OPSET_VERSION = 14

def get_model(model_name: str) -> torch.nn.Module:
    match model_name:
        case "resnet50":
            from torchvision.models import resnet50
            return resnet50(weights=None)
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_inputs(model_name: str):
    match model_name:
        case "resnet50":
            return (torch.randn(1, 3, 224, 224),)
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_io_names(model_name: str):
    match model_name:
        case "resnet50":
            return (["input"], ["output"], {"input": {0: "batch_size"}, "output": {0: "batch_size"}})
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def main(
    model_name: str = typer.Option(..., "-m", help="Model name"),
    input_file: str = typer.Option(..., "-i", help="Input file path"),
    output_file: str = typer.Option(..., "-o", help="Output file path"),
    verify: bool = typer.Option(True, help="Verify the exported ONNX model")
):
    model = get_model(model_name)
    model.load_state_dict(torch.load(input_file))
    model.eval()

    inputs = get_inputs(model_name)
    input_names, output_names, dynamic_axes = get_io_names(model_name)

    torch.onnx.export(
        model,
        inputs,
        output_file,
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Model exported to {output_file}.")

    if verify:
        import onnx
        import onnxruntime as ort

        # Load the exported ONNX model
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        # Run inference with ONNX Runtime
        ort_session = ort.InferenceSession(output_file)
        ort_inputs = {name: input.numpy() for name, input in zip(input_names, inputs)}
        ort_outputs = ort_session.run(output_names, ort_inputs)

        # Run inference with PyTorch
        with torch.no_grad():
            torch_outputs = model(*inputs)

        # Compare outputs
        for i, (torch_output, ort_output) in enumerate(zip(torch_outputs, ort_outputs)):
            if not torch.allclose(torch_output, torch.from_numpy(ort_output), rtol=1e-03, atol=1e-03):
                raise ValueError(f"Output {i} does not match between PyTorch and ONNX Runtime.")

        print("Verification successful: Outputs match between PyTorch and ONNX Runtime.")

if __name__ == "__main__":
    typer.run(main)
