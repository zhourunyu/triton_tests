"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2020-6-04 20:12:13
MODIFIED: 2020-6-28 14:04:45
"""
import acl
import numpy as np
import os

import acllite.constants as const
import acllite.utils as utils
from acllite.logger import log_warning, log_info, log_debug

class AclLiteModel(object):
    """
    wrap acl model inference interface, include input dataset construction,
    execute, and output transform to numpy array
    Attributes:
        model_path: om offline mode file path
        load_type: load model type, 0 from file, 1 from memory
    """

    def __init__(self, model_path: str, load_type: int = 0):
        self._run_mode, ret = acl.rt.get_run_mode()
        utils.check_ret("acl.rt.get_run_mode", ret)

        self._model_id = None
        self._model_desc = None
        self._input_buffer: list[tuple[int, int]] = []
        self._output_buffer: list[tuple[int, int]] = []
        self._input_dataset = None
        self._output_dataset = None
        # dynamic batch related
        self._batch_idx: int | None = None
        self.batch_sizes: list[int] = []
        self._max_batch_size = 1
        self._batch_size = 1    # current batch size

        self._init(model_path, load_type)
        self._is_destroyed = False

    def _init(self, model_path: str, load_type: int):
        log_debug("acl model init")

        # load model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if load_type == 0:
            self._model_id, ret = acl.mdl.load_from_file(model_path)
            utils.check_ret("acl.mdl.load_from_file", ret)
        elif load_type == 1:
            with open(model_path, "rb") as f:
                om_bytes = f.read()
            if om_bytes:
                ptr = acl.util.bytes_to_ptr(om_bytes)
                self._model_id, ret = acl.mdl.load_from_mem(ptr, len(om_bytes))
                utils.check_ret("acl.mdl.load_from_mem", ret)
            else:
                raise RuntimeError(f"Failed to read model file: {model_path}")
        else:
            raise ValueError(f"load_type must be 0 or 1, got {load_type}")
        self._model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._model_desc, self._model_id)
        utils.check_ret("acl.mdl.get_desc", ret)

        # create dataset
        self._gen_input_dataset()
        self._gen_output_dataset()

        log_info("acl model init success")
        return const.SUCCESS

    def _gen_input_dataset(self):
        log_debug("acl model create model input dataset")
        batch_idx, ret = acl.mdl.get_input_index_by_name(self._model_desc, "ascend_mbatch_shape_data")
        if ret == const.ACL_SUCCESS:
            log_debug(f"Model has dynamic batch, batch index: {batch_idx}")
            batch_dict, ret = acl.mdl.get_dynamic_batch(self._model_desc)
            utils.check_ret("acl.mdl.get_dynamic_batch", ret)
            self.batch_sizes: list[int] = batch_dict["batch"]
            self._batch_idx = batch_idx
            self._max_batch_size: int = max(self.batch_sizes)
            log_debug(f"Supported dynamic batch sizes: {self.batch_sizes}")

        input_num = acl.mdl.get_num_inputs(self._model_desc)
        dataset = acl.mdl.create_dataset()
        for i in range(input_num):
            # malloc device memory for input
            size = acl.mdl.get_input_size_by_index(self._model_desc, i)
            addr, ret = acl.rt.malloc(size, const.ACL_MEM_MALLOC_NORMAL_ONLY)
            utils.check_ret("acl.rt.malloc", ret)
            # create input data buffer
            buffer = acl.create_data_buffer(addr, size)
            _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
            if ret:
                acl.rt.free(addr)
                acl.destroy_data_buffer(buffer)
                utils.check_ret("acl.destroy_data_buffer", ret)
            if i != self._batch_idx:
                self._input_buffer.append((addr, size))
        self._input_dataset = dataset
        log_debug("acl model create model input dataset success")

    def _gen_output_dataset(self):
        log_debug("acl model create model output dataset")
        output_num = acl.mdl.get_num_outputs(self._model_desc)
        dataset = acl.mdl.create_dataset()
        for i in range(output_num):
            # malloc device memory for output
            size = acl.mdl.get_output_size_by_index(self._model_desc, i)
            addr, ret = acl.rt.malloc(size, const.ACL_MEM_MALLOC_NORMAL_ONLY)
            utils.check_ret("acl.rt.malloc", ret)
            # create output data buffer
            buffer = acl.create_data_buffer(addr, size)
            _, ret = acl.mdl.add_dataset_buffer(dataset, buffer)
            if ret:
                acl.rt.free(addr)
                acl.destroy_data_buffer(buffer)
                utils.check_ret("acl.destroy_data_buffer", ret)
            self._output_buffer.append((addr, size))
        self._output_dataset = dataset
        log_debug("acl model create model output dataset success")

    def execute(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        """
        inference input data
        Args:
            inputs: input data list
        returns:
            inference result data, which is a numpy array list,
            each corresponse to a model output
        """
        if self._batch_idx is not None:
            batch_size = self._get_batch_size_from_inputs(inputs)
            self._set_dynamic_batch_size(batch_size)

        self._copy_inputs_to_device(inputs)
        ret = acl.mdl.execute(self._model_id,
                              self._input_dataset,
                              self._output_dataset)
        utils.check_ret("acl.mdl.execute", ret)

        return self._copy_outputs_from_device()

    def _copy_inputs_to_device(self, inputs: list[np.ndarray]):
        log_debug("Copy input data to device memory")
        if len(inputs) != len(self._input_buffer):
            raise ValueError(f"Input number mismatch, model requires {len(self._input_buffer)}, but got {len(inputs)}")
        if self._run_mode == const.ACL_HOST:
            copy_type = const.ACL_MEMCPY_HOST_TO_DEVICE
        else:
            copy_type = const.ACL_MEMCPY_DEVICE_TO_DEVICE

        for input, (buffer, buffer_size) in zip(inputs, self._input_buffer):
            if "bytes_to_ptr" in dir(acl.util):
                ptr = acl.util.bytes_to_ptr(input.tobytes())
            else:
                ptr = acl.util.numpy_to_ptr(input)
            size = input.nbytes

            if self._batch_idx is None:
                expected_size = buffer_size
            else:
                expected_size = buffer_size // self._max_batch_size * self._batch_size
            if size > buffer_size:
                raise ValueError(f"Input data size {size} exceeds model input size {buffer_size}")
            elif size != expected_size:
                log_warning(f"Input data size {size} does not match expected size {expected_size}")

            ret = acl.rt.memcpy(buffer, buffer_size, ptr, size, copy_type)
            utils.check_ret("acl.rt.memcpy", ret)

    def _get_batch_size_from_inputs(self, inputs: list[np.ndarray]) -> int:
        batch_sizes = set([input.shape[0] for input in inputs])
        if len(batch_sizes) != 1:
            raise ValueError(f"Inconsistent batch sizes in inputs: {batch_sizes}")
        batch_size = batch_sizes.pop()
        log_debug(f"Determined batch size from inputs: {batch_size}")
        return batch_size

    def _set_dynamic_batch_size(self, batch: int):
        if not batch in self.batch_sizes:
            raise ValueError(f"Batch size ({batch}) not supported, available sizes: {self.batch_sizes}")

        ret = acl.mdl.set_dynamic_batch_size(self._model_id, self._input_dataset, self._batch_idx, batch)
        utils.check_ret("acl.mdl.set_dynamic_batch_size", ret)
        self._batch_size = batch

    def _copy_outputs_from_device(self) -> list[np.ndarray]:
        log_debug("Copy output data from device to host memory")
        if self._run_mode == const.ACL_HOST:
            copy_type = const.ACL_MEMCPY_DEVICE_TO_HOST
        else:
            copy_type = const.ACL_MEMCPY_DEVICE_TO_DEVICE

        outputs: list[np.ndarray] = []
        for i, (buffer, buffer_size) in enumerate(self._output_buffer):
            dims_dict, ret = acl.mdl.get_cur_output_dims(self._model_desc, i)
            utils.check_ret("acl.mdl.get_cur_output_dims", ret)
            dims: list[int] = dims_dict["dims"]
            dtype = utils.to_numpy_type(acl.mdl.get_output_data_type(self._model_desc, i))

            size = int(np.prod(dims)) * np.dtype(dtype).itemsize
            if size > buffer_size:
                raise RuntimeError(f"Output data size {size} exceeds model output size {buffer_size}")

            if "bytes_to_ptr" in dir(acl.util):
                output_bytes = bytes(size)
                output = np.frombuffer(output_bytes, dtype=dtype).reshape(dims)
                ptr = acl.util.bytes_to_ptr(output_bytes)
            else:
                output = np.empty(dims, dtype=dtype)
                ptr = acl.util.numpy_to_ptr(output)

            ret = acl.rt.memcpy(ptr, size, buffer, size, copy_type)
            utils.check_ret("acl.rt.memcpy", ret)
            outputs.append(output)
        return outputs

    def destroy(self):
        if self._is_destroyed:
            return

        self._release_dataset(self._input_dataset)
        self._release_dataset(self._output_dataset)
        if self._model_id:
            acl.mdl.unload(self._model_id)
        if self._model_desc:
            acl.mdl.destroy_desc(self._model_desc)

        self._is_destroyed = True

    def _release_dataset(self, dataset):
        if not dataset:
            return

        num = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num):
            data_buffer = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buffer:
                self._release_databuffer(data_buffer)
        acl.mdl.destroy_dataset(dataset)

    def _release_databuffer(self, data_buffer):
        addr = acl.get_data_buffer_addr(data_buffer)
        if addr:
            acl.rt.free(addr)
        acl.destroy_data_buffer(data_buffer)

    def __del__(self):
        self.destroy()
