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
    print(f"Model exported to {output_file}")

if __name__ == "__main__":
    typer.run(main)