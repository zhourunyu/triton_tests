import numpy as np
import os
import gc

import triton_python_backend_utils as pb_utils

from acllite import AclLiteModel, AclLiteResource
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
        model_file = os.path.join(pb_utils.get_model_dir(), f"{args.name}.om")
        assert os.path.isfile(model_file), f"Model file {model_file} does not exist."
        self.logger = pb_utils.Logger
        self.instance_id = args.instance_name.split("_")[-1]

        # load the model
        device_id = int(args.instance_name.split("_")[-2])
        self.acl_resource = AclLiteResource(device_id=device_id)
        self.model = AclLiteModel(model_file)

        self.inputs = self.config.input
        self.outputs = self.config.output

        if self.config.max_batch_size > 0:
            self.logger.log_info(f"[ascendcl] dynamic batch enabled with max batch size {self.config.max_batch_size}")

    def execute(self, requests: list) -> list:
        self.logger.log_info(f"[ascendcl] instance {self.instance_id} received {len(requests)} requests")

        # dynamic batching if enabled
        if self.config.max_batch_size > 0 and len(requests) > 1:
            return self._execute_batch(requests)

        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                acl_inputs: list[np.ndarray] = []
                for input in self.inputs:
                    acl_inputs.append(pb_utils.get_input_tensor_by_name(request, input.name).as_numpy())
                acl_outputs = self.model.execute(acl_inputs)
                output_tensors: list[pb_utils.Tensor] = []
                for output, output_array in zip(self.outputs, acl_outputs):
                    output_tensor = pb_utils.Tensor(output.name, output_array)
                    output_tensors.append(output_tensor)
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)

            except Exception as e:
                self.logger.log_error(f"[ascendcl] inference failed: {str(e)}")
                response = pb_utils.InferenceResponse(error=pb_utils.TritonModelException(str(e)))

            responses.append(response)
        return responses

    def _execute_batch(self, requests: list) -> list:
        responses: list[pb_utils.InferenceResponse] = []
        try:
            acl_inputs: list[np.ndarray] = []
            batch_sizes: list[int] = []
            # gather and concatenate inputs across all requests
            for input in self.inputs:
                inputs = [pb_utils.get_input_tensor_by_name(request, input.name).as_numpy() for request in requests]
                acl_inputs.append(np.concatenate(inputs, axis=0))
                if not batch_sizes:
                    batch_sizes = [inp.shape[0] for inp in inputs]

            # batch inference
            acl_outputs = self.model.execute(acl_inputs)

            # scatter outputs for each request
            current_batch_index = 0
            for batch in batch_sizes:
                output_tensors: list[pb_utils.Tensor] = []
                for output, output_array in zip(self.outputs, acl_outputs):
                    output_slice = output_array[current_batch_index:current_batch_index + batch]
                    output_tensor = pb_utils.Tensor(output.name, output_slice)
                    output_tensors.append(output_tensor)
                response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                responses.append(response)
                current_batch_index += batch

        except Exception as e:
            self.logger.log_error(f"[ascendcl] batch inference failed: {str(e)}")
            for _ in requests:
                response = pb_utils.InferenceResponse(error=pb_utils.TritonModelException(str(e)))
                responses.append(response)

        return responses

    def finalize(self):
        self.model.destroy()
        self.acl_resource = None
        self.logger.log_info("[ascendcl] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[ascendcl] Garbage Collector on finalize... done")
