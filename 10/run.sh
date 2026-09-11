#!/bin/bash

echo "Running Triton Inference Server on Ascend 310P..."

docker run -d --name triton_ascend_310p \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 9000:9000 \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    zhourunyu/triton_ascend_310p:260326 --model-repository=/triton_tests/1/models/310p > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

curl http://localhost:9000/metrics/npu
