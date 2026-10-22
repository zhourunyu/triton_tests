#!/bin/bash

echo "Running Triton Inference Server on Ascend 310P..."

docker network create triton &> /dev/null || true
docker run -d --name triton_ascend_310p \
    --network triton \
    -p 8000:8000 \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    zhourunyu/triton_ascend_310p:260921 --model-repository=/triton_tests/webpagedemo/models > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"


docker run -d --name triton_client \
    --network triton \
    -p 7860:7860 \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    -e SERVER_ADDR="triton_ascend_310p" \
    -e METRICS_DEVICE=NPU:0 \
    zhourunyu/triton_client:260921 > /dev/null

echo "Client is ready!"