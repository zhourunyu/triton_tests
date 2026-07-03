import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger

    def execute(self, requests):
        responses = []
        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "LOGITS").as_numpy()
            probabilties = logits - np.max(logits, axis=1, keepdims=True)
            probabilties = np.exp(probabilties)
            probabilties = probabilties / np.sum(probabilties, axis=1, keepdims=True)
            predictions = np.argsort(probabilties, axis=1)[:, ::-1]
            probabilties = np.sort(probabilties, axis=1)[:, ::-1]
            predictions_tensor = pb_utils.Tensor("PREDICTIONS", predictions.astype(np.int64))
            probabilties_tensor = pb_utils.Tensor("PROBABILITIES", probabilties.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[predictions_tensor, probabilties_tensor]))
        return responses
