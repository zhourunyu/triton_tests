#!/bin/bash

docker run --rm \
    --shm-size 1g \
    --runtime iluvatar -e IX_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests \
    -w /triton_tests/0 \
    --entrypoint /bin/sh \
    zhourunyu/triton_corex:260921 \
    -c "python3 onnx2ix.py -m resnet50 -i resnet50.onnx -o resnet50.ixrt"
