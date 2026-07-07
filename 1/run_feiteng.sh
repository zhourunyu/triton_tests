#!/bin/bash

echo "Running Triton Inference Server on Phytium..."

docker run -d --name triton_cpu \
    -p 8000:8000 \
    -p 8001:8001 \
    --shm-size 1g \
    -v `pwd`/..:/triton_tests:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_cpu:260429 --model-repository=/triton_tests/1/models/feiteng > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

./test_triton -m resnet50
./test_triton -m mobilenet_v2
./test_triton -m yolov5s
./test_triton -m bert-base-cased
./test_triton -m rnn
./test_triton -m gru
./test_triton -m lstm

docker stop triton_cpu > /dev/null
docker rm triton_cpu > /dev/null