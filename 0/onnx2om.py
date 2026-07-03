import typer
import subprocess

def get_inputs(model_name: str):
    match model_name:
        case "resnet50":
            return ({"input": [-1, 3, 224, 224]}, "NCHW")
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def main(
    model_name: str = typer.Option(..., "-m", help="Model name"),
    input_file: str = typer.Option(..., "-i", help="Input file path"),
    output_file: str = typer.Option(..., "-o", help="Output file path"),
    batch_sizes: list[int] = typer.Option([1, 2, 4, 8, 16], "-b", help="Batch sizes"),
    soc_version: str = typer.Option("Ascend310P3", help="SOC version"),
):
    input_shapes, input_format = get_inputs(model_name)
    shapes = []
    for input, shape in input_shapes.items():
        shapes.append(f"{input}:{','.join([str(dim) for dim in shape])}")
    shape_str = ";".join(shapes)
    atc_args = [
        "atc",
        "--framework=5",
        f"--model={input_file}",
        f"--output={output_file}",
        f"--input_format={input_format}",
        f"--soc_version={soc_version}",
        f"--input_shape={shape_str}",
    ]
    if batch_sizes:
        atc_args.append(f"--dynamic_batch_size={','.join([str(batch_size) for batch_size in batch_sizes])}")
    print("Running command:", " ".join(atc_args))
    subprocess.run(atc_args, check=True)

if __name__ == "__main__":
    typer.run(main)