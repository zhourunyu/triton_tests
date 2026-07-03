from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

model = "lstm"
url = "localhost:8000"

input_size = 128

lstm_inputs = [np.random.rand(1, 1, input_size).astype(np.float32) for _ in range(3)]
num_requests = 10

def infer(sequence_id):
    client = InferenceServerClient(url=url)
    for i, input_data in enumerate(lstm_inputs):
        inputs = [InferInput("INPUT", [1, 1, input_size], "FP32").set_data_from_numpy(input_data)]
        outputs = [InferRequestedOutput("OUTPUT")]

        is_start = i == 0
        is_end = i == len(lstm_inputs) - 1

        client.infer(
            model_name=model,
            inputs=inputs,
            outputs=outputs,
            sequence_id=sequence_id + 1000,
            sequence_start=is_start,
            sequence_end=is_end,
        )

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    print(f"Sending {num_requests} concurrent requests to model {model}...")
    with ThreadPoolExecutor() as executor:
        results = executor.map(infer, range(num_requests))
    print("All requests completed")