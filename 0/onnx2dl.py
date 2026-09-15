import dlnne
import typer

max_batch_size = 16

def export_onnx_to_engine(onnx_path, engine_path, input_shapes: list[tuple[str, tuple[int, ...]]]):
    builder = dlnne.Builder()
    builder.config.max_batch_size = max_batch_size
    network = builder.create_network()
    parser = dlnne.Parser()
    for input_name, input_shape in input_shapes:
        assert parser.register_input(input_name, input_shape)
    assert parser.parse(onnx_path, network)
    engine = builder.build_engine(network)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

def get_input_shape(model_name: str) -> list[tuple[str, tuple[int, ...]]]:
    match model_name:
        case "resnet50":
            return [("input", (1, 3, 224, 224))]
        case _:
            raise ValueError(f"Unsupported model: {model_name}")

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

def infer_dlnne(engine_path: str, inputs: list[tuple[str, "np.ndarray"]], output_shapes: list[tuple[str, tuple[int, ...]]]) -> list["np.ndarray"]:
    import pydlcuda as cuda
    import numpy as np

    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    stream = cuda.Stream()
    with open(engine_path, "rb") as f:
        engine = dlnne.deserialize(f.read())
    context = engine.create_execution_context()

    bindings = [0] * engine.num_bindings
    input_buffers = []
    for name, input in inputs:
        index = engine.get_binding_index(name)
        input_size = np.prod(input.shape) * input.itemsize
        input_buffer = cuda.mem_alloc(input_size)
        input_buffers.append(input_buffer)
        bindings[index] = int(input_buffer)
        cuda.py_memcpy_htod_async(input_buffer, input, stream)
    outputs = []
    output_buffers = []
    for name, shape in output_shapes:
        index = engine.get_binding_index(name)
        output = np.empty(shape, dtype=np.float32)
        output_size = np.prod(shape) * output.itemsize
        output_buffer = cuda.mem_alloc(output_size)
        outputs.append(output)
        output_buffers.append(output_buffer)
        bindings[index] = int(output_buffer)

    context.execute_async(1, bindings, stream.handle)
    for output, output_buffer in zip(outputs, output_buffers):
        cuda.memcpy_dtoh_async(output, output_buffer, stream)
    stream.synchronize()
    return outputs

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
    verify: bool = typer.Option(True, help="Verify the exported engine"),
):
    input_shapes = get_input_shape(model_name)
    export_onnx_to_engine(input_file, output_file, input_shapes)
    print(f"Model exported to {output_file}.")

    if verify:
        import numpy as np

        inputs = get_inputs(model_name)
        output_shapes = get_output_shape(model_name)
        output_names = [name for name, _ in output_shapes]
        ort_outputs = infer_onnx(input_file, inputs, output_names)
        dlnne_outputs = infer_dlnne(output_file, inputs, output_shapes)

        for i, (ort_output, dlnne_output) in enumerate(zip(ort_outputs, dlnne_outputs)):
            if not np.allclose(ort_output, dlnne_output, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Output {i} does not match between ONNX Runtime and DLNNE.")

        print("Verification successful: Outputs match between ONNX Runtime and DLNNE.")

if __name__ == "__main__":
    typer.run(main)