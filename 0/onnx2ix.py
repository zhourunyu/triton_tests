import tensorrt as trt
import typer

def export_onnx_to_engine(onnx_path, engine_path, input_shapes: dict[str, tuple[list[int], list[int], list[int]]]):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read())

    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for input_name, (min_shape, opt_shape, max_shape) in input_shapes.items():
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

def get_input_shapes(model_name: str) -> dict[str, tuple[list[int], list[int], list[int]]]:
    match model_name:
        case "resnet50":
            return {"input": ([1, 3, 224, 224], [1, 3, 224, 224], [16, 3, 224, 224])}
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

def infer_ixrt(engine_path: str, inputs: list[tuple[str, "np.ndarray"]], output_shapes: list[tuple[str, tuple[int, ...]]]) -> list["np.ndarray"]:
    import pycuda.driver as cuda
    import numpy as np

    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    stream = cuda.Stream()

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_buffers = []
    for name, input in inputs:
        index = engine.get_binding_index(name)
        input_size = int(np.prod(input.shape) * input.itemsize)
        input_buffer = cuda.mem_alloc(input_size)
        context.set_input_shape(name, input.shape)
        context.set_tensor_address(name, int(input_buffer))
        cuda.memcpy_htod_async(input_buffer, input, stream)
        input_buffers.append(input_buffer)
    outputs = []
    output_buffers = []
    for name, shape in output_shapes:
        index = engine.get_binding_index(name)
        output = np.empty(shape, dtype=np.float32)
        output_size = int(np.prod(shape) * output.itemsize)
        output_buffer = cuda.mem_alloc(output_size)
        context.set_tensor_address(name, int(output_buffer))
        output_buffers.append(output_buffer)

    context.execute_async_v3(stream.handle)
    for output, output_buffer in zip(outputs, output_buffers):
        cuda.memcpy_dtoh_async(output, output_buffer, stream)
    stream.synchronize()

    for buffer in input_buffers + output_buffers:
        buffer.free()
    ctx.pop()
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
    input_shapes = get_input_shapes(model_name)
    export_onnx_to_engine(input_file, output_file, input_shapes)
    print(f"Model exported to {output_file}.")

    if verify:
        import numpy as np

        inputs = get_inputs(model_name)
        output_shapes = get_output_shape(model_name)
        output_names = [name for name, _ in output_shapes]
        ort_outputs = infer_onnx(input_file, inputs, output_names)
        ixrt_outputs = infer_ixrt(output_file, inputs, output_shapes)

        for i, (ort_output, ixrt_output) in enumerate(zip(ort_outputs, ixrt_outputs)):
            if not np.allclose(ort_output, ixrt_output, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Output {i} does not match between ONNX Runtime and ixRT.")

        print("Verification successful: Outputs match between ONNX Runtime and ixRT.")

if __name__ == "__main__":
    typer.run(main)