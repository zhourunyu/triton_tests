#!/bin/bash

docker run --rm \
    -v `pwd`/..:/triton_tests \
    -w /triton_tests/0 \
    --entrypoint /bin/sh \
    zhourunyu/triton_cpu:260429 \
    -c "pip3 install --break-system-packages typer onnx && pip3 install --break-system-packages \"torch<2.9\" torchvision --index-url https://download.pytorch.org/whl/cpu && python3 torch2onnx.py -m resnet50 -i resnet50-11ad3fa6.pth -o resnet50.onnx"
