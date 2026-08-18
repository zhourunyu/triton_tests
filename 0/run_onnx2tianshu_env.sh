#!/bin/bash

#该脚本用于启动天数的docker容器，在容器内进行模型格式的转换，onnx->engine(tianshu)

echo "Running Triton Inference Server on Iluvatar MR-V100..."

docker run -d --name triton_corex \
    -p 8000:8000 \
    -p 8001:8001 \
    --shm-size 1g \
    --runtime iluvatar -e IX_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests \
    -v /root/weight:/weight:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_corex:260326 --model-repository=/triton_tests/1/models/zhikai \
    --model-control-mode=explicit --load-model=resnet50 > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"