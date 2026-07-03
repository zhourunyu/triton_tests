from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

model = "resnet50_ensemble"
url = "localhost:8000"
image = "lion.jpg"

if __name__ == "__main__":
    client = InferenceServerClient(url=url)

    with open(image, 'rb') as f:
        image_data = [f.read()]

    inputs = [InferInput("IMAGE", [1, 1], "BYTES").set_data_from_numpy(np.array([image_data], dtype=np.object_))]
    outputs = [InferRequestedOutput("PREDICTIONS"), InferRequestedOutput("PROBABILITIES")]
    print(f"Running inference with model {model}")
    result = client.infer(model_name=model, inputs=inputs, outputs=outputs)
    predictions = result.as_numpy("PREDICTIONS")
    probabilities = result.as_numpy("PROBABILITIES")
    if predictions is not None and probabilities is not None:
        print(f"Image: {image}")
        print(f"Top-5 predictions:")
        for i in range(5):
            print(f"{predictions[0][i]}: {probabilities[0][i]:.4f}")
        print("Inference successful!")
