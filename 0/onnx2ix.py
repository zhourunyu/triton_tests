import typer
import subprocess

def get_inputs(model_name: str):
    match model_name:
        case "resnet50":
            return {"input": ([1, 3, 224, 224], [1, 3, 224, 224], [16, 3, 224, 224])}
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def main(
    model_name: str = typer.Option(..., "-m", help="Model name"),
    input_file: str = typer.Option(..., "-i", help="Input file path"),
    output_file: str = typer.Option(..., "-o", help="Output file path"),
):
    input_shapes = get_inputs(model_name)
    min_shapes = []
    opt_shapes = []
    max_shapes = []
    for input, (min_shape, opt_shape, max_shape) in input_shapes.items():
        min_shapes.append(f"{input}:{'x'.join([str(dim) for dim in min_shape])}")
        opt_shapes.append(f"{input}:{'x'.join([str(dim) for dim in opt_shape])}")
        max_shapes.append(f"{input}:{'x'.join([str(dim) for dim in max_shape])}")

    atc_args = [
        "ixrtexec",
        "--onnx", input_file,
        "--save_engine", output_file,
        "--min_shape", ','.join(min_shapes),
        "--opt_shape", ','.join(opt_shapes),
        "--max_shape", ','.join(max_shapes),
    ]
    print("Running command:", " ".join(atc_args))
    subprocess.run(atc_args, check=True)

if __name__ == "__main__":
    typer.run(main)