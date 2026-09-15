#!/bin/bash

docker run --rm \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests \
    -w /triton_tests/0 \
    --entrypoint /bin/sh \
    zhourunyu/triton_ascend_310p:260326 \
    -c "pip3 install onnxruntime && python3 onnx2om.py -m resnet50 -i resnet50.onnx -o resnet50.om"