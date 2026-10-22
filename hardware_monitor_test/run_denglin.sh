#!/bin/bash

echo "Running Triton Inference Server on DLGPU..."

docker run -d --name triton_denglin \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 9000:9000 \
    --shm-size 1g \
    --runtime dlrt -e DENGLIN_DEVICES=0 \
    -v `pwd`/..:/triton_tests \
    -v /root/weight:/weight:ro \
    zhourunyu/triton_denglin:260921 --model-repository=/triton_tests/1/models/denglin \
    --model-control-mode=explicit --load-model=resnet50 > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

curl http://localhost:9000/metrics/dlgpu

docker stop triton_denglin > /dev/null
docker rm triton_denglin > /dev/null