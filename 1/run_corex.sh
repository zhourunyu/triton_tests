#!/bin/bash

echo "Running Triton Inference Server on Iluvatar MR-V100..."

docker run -d --name triton_corex \
    -p 8000:8000 \
    -p 8001:8001 \
    --shm-size 1g \
    --runtime iluvatar -e IX_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_corex:260326 --model-repository=/triton_tests/1/models/zhikai > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"


./test_triton_llm -m Qwen3-4B


docker stop triton_corex > /dev/null
docker rm triton_corex > /dev/null