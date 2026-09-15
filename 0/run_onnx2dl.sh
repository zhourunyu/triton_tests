#!/bin/bash

docker run --rm \
    --shm-size 1g \
    --runtime dlrt -e DENGLIN_DEVICES=0 \
    -v `pwd`/..:/triton_tests \
    -w /triton_tests/0 \
    --entrypoint /bin/sh \
    zhourunyu/triton_denglin:260326 \
    -c "pip3 install \"onnx<1.19\" onnxruntime && python3 onnx2dl.py -m resnet50 -i resnet50.onnx -o resnet50.dlnne"
