#!/bin/bash

docker run --rm \
    -v `pwd`/..:/triton_tests \
    -w /triton_tests/0 \
    --entrypoint /bin/sh \
    zhourunyu/triton_cpu:260921 \
    -c "python3 torch2onnx.py -m resnet50 -i resnet50-11ad3fa6.pth -o resnet50.onnx"
