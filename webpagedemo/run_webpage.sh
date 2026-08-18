#!/bin/bash

echo "Running Triton Inference Server on Ascend 310P..."

docker run -d --name triton_ascend_310p \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 9000:9000 \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v /root/310p/models:/models:ro \
    -v /root/weight:/weight:ro \
    zhourunyu/triton_ascend_310p:260611 --model-repository=/models

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"


docker run -d --name triton_client \
    -p 7860:7860 \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    -e SERVER_ADDR=192.168.10.98 \
    -e METRICS_DEVICE=NPU:0 \
    zhourunyu/triton_client:260611

echo "Client is ready!"