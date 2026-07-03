#!/bin/bash

echo "Running Triton Inference Server on Ascend 310P..."

docker run -d --name triton_ascend_310p \
    -p 8000:8000 \
    -p 8001:8001 \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_ascend_310p:260326 --model-repository=/triton_tests/1/models/310p > /dev/null

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
./test_triton_llm -m Qwen3-4B

docker stop triton_ascend_310p > /dev/null
docker rm triton_ascend_310p > /dev/null