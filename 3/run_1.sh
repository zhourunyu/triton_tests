#!/bin/bash

echo "Running Triton Inference Server on CPU..."

docker run -d --name triton_cpu \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    --shm-size 1g \
    -v `pwd`/..:/triton_tests:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_cpu:260429 --model-repository=/triton_tests/3/models --model-control-mode=explicit > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

python3 test_1.py

curl -s http://localhost:8002/metrics > metrics.txt
echo "Server metrics has been saved to metrics.txt"

docker logs triton_cpu &> triton.log
echo "Server logs has been saved to triton.log"
docker stop triton_cpu > /dev/null
docker rm triton_cpu > /dev/null