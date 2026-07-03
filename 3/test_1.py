from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np
import json

model = "resnet50"
url = "localhost:8000"

if __name__ == "__main__":
    client = InferenceServerClient(url=url)
    print(f"Model {model} readiness: {client.is_model_ready(model)}")
    print(f"Loading model {model}...")
    client.load_model(model)
    if client.is_model_ready(model):
        print(f"Model {model} is ready!")

    inputs = [InferInput("input", [1, 3, 224, 224], "FP32").set_data_from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32))]
    outputs = [InferRequestedOutput("output")]
    config = client.get_model_config(model)
    print(f"Current version policy: {config['version_policy']}")
    print(f"Running inference with model {model}")
    client.infer(model_name=model, inputs=inputs, outputs=outputs)
    print("Inference successful!")

    print("=" * 50)
    print("Updating model version policy...")
    client.unload_model(model)
    config["version_policy"] = {"all": {}}
    client.load_model(model, config=json.dumps(config))
    print(f"Current version policy: {client.get_model_config(model)['version_policy']}")
    print(f"Running inference with model {model} version 1...")
    client.infer(model_name=model, inputs=inputs, outputs=outputs, model_version="1")
    print("Inference successful!")
    print(f"Running inference with model {model} version 2...")
    client.infer(model_name=model, inputs=inputs, outputs=outputs, model_version="2")
    print("Inference successful!")