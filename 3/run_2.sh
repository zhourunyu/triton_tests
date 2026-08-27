#!/bin/bash

echo "Running Triton Inference Server on CPU..."

docker run -d --name triton_cpu \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    --shm-size 1g \
    -v `pwd`/..:/triton_tests:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_cpu:260429 --model-repository=/triton_tests/3/models --model-control-mode=poll --repository-poll-secs=10 > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

python3 test_2.py

docker logs triton_cpu &> triton_run_2.log
echo "Server logs has been saved to triton_run_2.log"
docker stop triton_cpu > /dev/null
docker rm triton_cpu > /dev/null