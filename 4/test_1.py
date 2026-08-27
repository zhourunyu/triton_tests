from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

num_requests = 20
model = "resnet50"
url = "localhost:8000"

inputs = [InferInput("input", [1, 3, 224, 224], "FP32").set_data_from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32))]
outputs = [InferRequestedOutput("output")]

def infer():
    client = InferenceServerClient(url=url)
    client.infer(model_name=model, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    print(f"Sending {num_requests} concurrent requests to model {model}...")
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda _: infer(), range(num_requests))
    print("All requests completed")
