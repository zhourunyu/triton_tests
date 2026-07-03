import numpy as np
import os

import triton_python_backend_utils as pb_utils

from onnxruntime import InferenceSession
from pydantic import BaseModel, Field, Json

class ModelConfig(BaseModel):
    class Input(BaseModel):
        name: str
        data_type: str
        dims: list[int]
        optional: bool = False

    class Output(BaseModel):
        name: str
        data_type: str
        dims: list[int]

    name: str
    platform: str
    backend: str
    max_batch_size: int
    input: list[Input]
    output: list[Output]

class TritonArguments(BaseModel):
    config: Json[ModelConfig] = Field(alias="model_config")
    instance_kind: str = Field(alias="model_instance_kind")
    instance_name: str = Field(alias="model_instance_name")
    instance_device_id: int = Field(alias="model_instance_device_id")
    repository: str = Field(alias="model_repository")
    version: str = Field(alias="model_version")
    name: str = Field(alias="model_name")

class TritonPythonModel:
    def initialize(self, args):
        # parse the arguments
        args = TritonArguments.model_validate(args)
        self.config = args.config
        model_file = os.path.join(pb_utils.get_model_dir(), f"{args.name}.onnx")
        assert os.path.isfile(model_file), f"Model file {model_file} does not exist."
        self.logger = pb_utils.Logger
        self.instance_id = args.instance_name.split("_")[-1]

        # load the model
        self.session = InferenceSession(model_file, providers=['CPUExecutionProvider'])

        self.inputs = self.config.input
        self.outputs = self.config.output
        self.output_names = [output.name for output in self.outputs]

        if self.config.max_batch_size > 0:
            self.logger.log_info(f"[onnxrt] dynamic batch enabled with max batch size {self.config.max_batch_size}")

    def execute(self, requests: list) -> list:
        self.logger.log_info(f"[onnxrt] instance {self.instance_id} received {len(requests)} requests")

        # dynamic batching if enabled
        if self.config.max_batch_size > 0 and len(requests) > 1:
            return self._execute_batch(requests)

        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                onnxrt_inputs: dict[str, np.ndarray] = {}
                for input in self.inputs:
                    onnxrt_inputs[input.name] = pb_utils.get_input_tensor_by_name(request, input.name).as_numpy()

                onnxrt_outputs: list[np.ndarray] = self.session.run(self.output_names, onnxrt_inputs) # type: ignore
                output_tensors: list[pb_utils.Tensor] = []
                for output, output_array in zip(self.outputs, onnxrt_outputs):
                    output_tensor = pb_utils.Tensor(output.name, output_array)
                    output_tensors.append(output_tensor)
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)

            except Exception as e:
                self.logger.log_error(f"[onnxrt] inference failed: {str(e)}")
                response = pb_utils.InferenceResponse(error=pb_utils.TritonModelException(str(e)))

            responses.append(response)
        return responses

    def _execute_batch(self, requests: list) -> list:
        responses: list[pb_utils.InferenceResponse] = []
        try:
            onnxrt_inputs: dict[str, np.ndarray] = {}
            batch_sizes: list[int] = []
            # gather and concatenate inputs across all requests
            for input in self.inputs:
                inputs = [pb_utils.get_input_tensor_by_name(request, input.name).as_numpy() for request in requests]
                onnxrt_inputs[input.name] = np.concatenate(inputs, axis=0)
                if not batch_sizes:
                    batch_sizes = [inp.shape[0] for inp in inputs]

            # batch inference
            onnxrt_outputs: list[np.ndarray] = self.session.run(self.output_names, onnxrt_inputs) # type: ignore

            # scatter outputs for each request
            current_batch_index = 0
            for batch in batch_sizes:
                output_tensors: list[pb_utils.Tensor] = []
                for output, output_array in zip(self.outputs, onnxrt_outputs):
                    output_slice = output_array[current_batch_index:current_batch_index + batch]
                    output_tensor = pb_utils.Tensor(output.name, output_slice)
                    output_tensors.append(output_tensor)
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                responses.append(response)
                current_batch_index += batch

        except Exception as e:
            self.logger.log_error(f"[onnxrt] batch inference failed: {str(e)}")
            for _ in requests:
                response = pb_utils.InferenceResponse(error=pb_utils.TritonModelException(str(e)))
                responses.append(response)

        return responses