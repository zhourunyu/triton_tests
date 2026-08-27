#!/bin/bash

#该脚本用于启动容器，容器中有pytorch模型转成成onnx模型、onnx模型转换成om格式模型所需的环境

echo "Running Triton Inference Server on Ascend 310P..."

docker run -d --name triton_ascend_310p \
    -p 8000:8000 \
    -p 8001:8001 \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests \
    -v /root/weight:/weight \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_ascend_310p:260326 --model-repository=/triton_tests/1/models/310p \
    --model-control-mode=explicit --load-model=resnet50 > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"