import typer
import subprocess

def get_input_shape(model_name: str):
    match model_name:
        case "resnet50":
            return ({"input": [-1, 3, 224, 224]}, "NCHW")
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_output_shape(model_name: str) -> list[tuple[str, tuple[int, ...]]]:
    match model_name:
        case "resnet50":
            return [("output", (1, 1000))]
        case _:
            raise ValueError(f"Unsupported model: {model_name}")

def get_inputs(model_name: str) -> list[tuple[str, "np.ndarray"]]:
    import numpy as np
    match model_name:
        case "resnet50":
            return [("input", np.random.randn(1, 3, 224, 224).astype(np.float32))]
        case _:
            raise ValueError(f"Unsupported model: {model_name}")

def infer_acl(model_path: str, inputs: list[tuple[str, "np.ndarray"]], output_shapes: list[tuple[str, tuple[int, ...]]]) -> list["np.ndarray"]:
    from acllite import AclLiteModel, AclLiteResource
    import numpy as np

    resource = AclLiteResource(device_id=0)
    model = AclLiteModel(model_path)

    acl_inputs: list[np.ndarray] = [input for _, input in inputs]
    acl_outputs = model.execute(acl_inputs)

    model.destroy()
    return acl_outputs

def infer_onnx(onnx_path: str, inputs: list[tuple[str, "np.ndarray"]], output_names: list[str]) -> list["np.ndarray"]:
    import onnxruntime as ort

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {name: input for name, input in inputs}
    outputs = ort_session.run(output_names, ort_inputs)
    return outputs

def main(
    model_name: str = typer.Option(..., "-m", help="Model name"),
    input_file: str = typer.Option(..., "-i", help="Input file path"),
    output_file: str = typer.Option(..., "-o", help="Output file path"),
    batch_sizes: list[int] = typer.Option([1, 2, 4, 8, 16], "-b", help="Batch sizes"),
    soc_version: str = typer.Option("Ascend310P3", help="SOC version"),
    verify: bool = typer.Option(True, help="Verify the exported model"),
):
    input_shapes, input_format = get_input_shape(model_name)
    shapes = []
    for input, shape in input_shapes.items():
        shapes.append(f"{input}:{','.join([str(dim) for dim in shape])}")
    shape_str = ";".join(shapes)
    output_file = output_file.removesuffix(".om")
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
    output_file = output_file + ".om"
    print(f"Model exported to {output_file}.")

    if verify:
        import numpy as np

        inputs = get_inputs(model_name)
        output_shapes = get_output_shape(model_name)
        output_names = [name for name, _ in output_shapes]
        ort_outputs = infer_onnx(input_file, inputs, output_names)
        acl_outputs = infer_acl(output_file, inputs, output_shapes)

        for i, (ort_output, acl_output) in enumerate(zip(ort_outputs, acl_outputs)):
            if not np.allclose(ort_output, acl_output, rtol=1e-2, atol=1e-2):
                raise ValueError(f"Output {i} does not match between ONNX Runtime and AscendCL.")

        print("Verification successful: Outputs match between ONNX Runtime and AscendCL.")

if __name__ == "__main__":
    typer.run(main)