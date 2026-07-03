import json
import numpy as np
import random
import time
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput, InferResult, InferenceServerException

class LLMClient:
    def __init__(self, url: str):
        self._client = InferenceServerClient(url=url)

    def chat(self, model, prompt, sampling_parameters: dict):
        prompt = self._apply_chat_template(model, prompt)
        inputs, outputs = self._build_request(prompt, sampling_parameters)
        finished = False

        def callback(result: InferResult, error: InferenceServerException | None):
            nonlocal finished
            text_output = result.as_numpy("text_output")
            finish_reason = result.as_numpy("finish_reason")
            if finish_reason is not None and finish_reason[0] != "None":
                finished = True

        self._client.start_stream(callback)
        self._client.async_stream_infer(model_name=model, inputs=inputs, outputs=outputs)
        while not finished:
            time.sleep(1)

    def _apply_chat_template(self, model, prompt):
        if model.startswith("Qwen3"):
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unknown model: {model}")

    def _build_request(self, prompt: str, sampling_parameters: dict):
        inputs: list[InferInput] = []

        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        inputs.append(InferInput("text_input", [1], "BYTES").set_data_from_numpy(prompt_data))

        sampling_parameters_data = np.array(
            [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
        )
        inputs.append(InferInput("sampling_parameters", [1], "BYTES").set_data_from_numpy(sampling_parameters_data))

        stream_data = np.array([True], dtype=bool)
        inputs.append(InferInput("stream", [1], "BOOL").set_data_from_numpy(stream_data))

        exclude_input_in_output_data = np.array([True], dtype=bool)
        inputs.append(InferInput("exclude_input_in_output", [1], "BOOL").set_data_from_numpy(exclude_input_in_output_data))

        return_finish_reason_data = np.array([True], dtype=bool)
        inputs.append(InferInput("return_finish_reason", [1], "BOOL").set_data_from_numpy(return_finish_reason_data))

        outputs: list[InferRequestedOutput] = []
        outputs.append(InferRequestedOutput("text_output"))
        outputs.append(InferRequestedOutput("finish_reason"))

        # Issue the asynchronous sequence inference.
        return inputs, outputs

url = "localhost:8001"
model = "Qwen3-4B"
prompt = "Hello!"
num_requests = 10

def infer():
    client = LLMClient(url)
    sampling_parameters = {
        "max_tokens": 512,
        "seed": random.randint(0, 100),
    }
    client.chat(model, prompt, sampling_parameters)

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    print(f"Sending {num_requests} concurrent requests to model {model}...")
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda _: infer(), range(num_requests))
    print("All requests completed")