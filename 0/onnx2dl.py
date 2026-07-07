from dlnne import Builder, Parser
import typer

max_batch_size = 16

def export_onnx_to_engine(onnx_path, engine_path, inputs: list[tuple[str, tuple[int, ...]]]):
    builder = Builder()
    builder.config.max_batch_size = max_batch_size
    network = builder.create_network()
    parser = Parser()
    for input_name, input_shape in inputs:
        assert parser.register_input(input_name, input_shape)
    assert parser.parse(onnx_path, network)
    engine = builder.build_engine(network)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

def export(model_name: str, input_file: str, output_file: str):
    match model_name:
        case "resnet50" | "mobilenet_v2":
            inputs = [("input", (1, 3, 224, 224))]
        case "yolov5s":
            inputs = [("input", (1, 3, 640, 640))]
        case "bert-base-cased":
            inputs = [("input_ids", (1, 32)), ("attention_mask", (1, 32)), ("token_type_ids", (1, 32))]
        case _:
            raise ValueError(f"Unsupported model: {model_name}")
    export_onnx_to_engine(input_file, output_file, inputs)

def main(
    model_name: str = typer.Option(..., "-m", help="Model name"),
    input_file: str = typer.Option(..., "-i", help="Input file path"),
    output_file: str = typer.Option(..., "-o", help="Output file path"),
):
    export(model_name, input_file, output_file)

if __name__ == "__main__":
    typer.run(main)