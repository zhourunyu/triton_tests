import triton_python_backend_utils as pb_utils
import numpy as np
import torch

input_size = 128
hidden_size = 256
num_layers = 2

class TritonPythonModel:
    def initialize(self, args):
        self.states = {}
        self.logger = pb_utils.Logger
        self.model = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.model.eval()
        self._init_state(0)  # corrid 0 for placeholder requests

    def _init_state(self, corrid):
        self.states[corrid] = (torch.zeros(num_layers, 1, hidden_size), torch.zeros(num_layers, 1, hidden_size))

    def execute(self, requests):
        self.logger.log_info(f"[lstm] received {len(requests)} requests")
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()

            start_tensor = pb_utils.get_input_tensor_by_name(request, "START").as_numpy().flat
            end_tensor = pb_utils.get_input_tensor_by_name(request, "END").as_numpy().flat
            corrid_tensor = pb_utils.get_input_tensor_by_name(request, "CORRID").as_numpy().flat

            outputs = []
            for input, is_start, is_end, corrid in zip(input_tensor, start_tensor, end_tensor, corrid_tensor):
                self.logger.log_info(f"[lstm] processing corrid {corrid}, start: {is_start}, end: {is_end}")
                if is_start:
                    self._init_state(corrid)

                if not corrid in self.states:
                    self.logger.log_warn(f"[lstm] received data for corrid {corrid} without START signal")
                    self._init_state(corrid)
                h, c = self.states[corrid]
                input_torch = torch.from_numpy(input).float().unsqueeze(0)  # Add batch dimension
                with torch.no_grad():
                    out, (h, c) = self.model(input_torch, (h, c))
                self.states[corrid] = (h, c)
                outputs.append(out.numpy().squeeze(0))  # Remove batch dimension

                if is_end:
                    self.states.pop(corrid, None)

            out_tensor = pb_utils.Tensor("OUTPUT", np.stack(outputs, dtype=np.float32))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)

        return responses

    def finalize(self):
        self.states.clear()