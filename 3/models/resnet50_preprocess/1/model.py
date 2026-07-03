import numpy as np
import io
import triton_python_backend_utils as pb_utils

from PIL import Image
from torchvision.models import ResNet50_Weights

class TritonPythonModel:
    def initialize(self, args):
        self.transform = ResNet50_Weights.DEFAULT.transforms()
        self.logger = pb_utils.Logger

    def execute(self, requests):
        responses = []
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()
            output_tensors = []
            for image in images:
                image = Image.open(io.BytesIO(image[0])).convert('RGB')
                output_tensors.append(self.transform(image).numpy())
            out_numpy = np.stack(output_tensors, axis=0)
            out_tensor = pb_utils.Tensor("PREPROCESSED_TENSOR", out_numpy.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
