from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput, model_config_pb2
from google.protobuf import text_format
from threading import Thread
import numpy as np
import time

model = "resnet50"
url = "localhost:8001"

def infer():
    client = InferenceServerClient(url=url)
    inputs = [InferInput("input", [1, 3, 224, 224], "FP32").set_data_from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32))]
    outputs = [InferRequestedOutput("output")]
    start_time = time.time()
    while time.time() - start_time < 10:
        client.infer(model_name=model, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    client = InferenceServerClient(url=url)
    if config := client.get_model_config(model, as_json=True):
        print(f"Current max batch size: {config['config']['max_batch_size']}")

    print(f"Updating config.pbtxt of model {model}...")
    with open(f"models/{model}/config.pbtxt", "r") as f:
        config_txt = f.read()
    config = model_config_pb2.ModelConfig()
    text_format.Parse(config_txt, config)
    config.max_batch_size = 4

    t = Thread(target=infer)
    t.start()

    new_config_txt = text_format.MessageToString(config)
    with open(f"models/{model}/config.pbtxt", "w") as f:
        f.write(new_config_txt)

    print(f"Waiting for the server to update the model config...")
    while True:
        if config := client.get_model_config(model, as_json=True):
            if config["config"]["max_batch_size"] == 4:
                print(f"Config updated successfully! Current max batch size: {config['config']['max_batch_size']}")
                break
        time.sleep(1)

    t.join()

    with open(f"models/{model}/config.pbtxt", "w") as f:
        f.write(config_txt)
    print(f"Config restored to original state.")