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
    zhourunyu/triton_ascend_310p:260326 --model-repository=/triton_tests/Quantization_demo/models --model-control-mode=explicit --load-model=Qwen3-4B-W8A8 > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

echo "Profiling Qwen3-4B-W8A8 model..."

genai-perf profile \
    -u localhost:8001 \
    -m Qwen3-4B-W8A8 \
    --tokenizer Qwen/Qwen3-4B \
    --backend vllm \
    --synthetic-input-tokens-mean 2048 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --concurrency 1 \
    --streaming &> genai_perf_w8a8.log

echo "Performance results has been saved to genai_perf_w8a8.log"

docker stop triton_ascend_310p > /dev/null

docker rm triton_ascend_310p > /dev/null


docker run -d --name triton_ascend_310p \
    -p 8000:8000 \
    -p 8001:8001 \
    --shm-size 1g \
    --runtime ascend -e ASCEND_VISIBLE_DEVICES=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    --entrypoint /opt/tritonserver/bin/tritonserver \
    zhourunyu/triton_ascend_310p:260326 --model-repository=/triton_tests/Quantization_demo/models --model-control-mode=explicit --load-model=Qwen3-4B > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

echo "Profiling Qwen3-4B model..."

genai-perf profile \
    -u localhost:8001 \
    -m Qwen3-4B \
    --tokenizer Qwen/Qwen3-4B \
    --backend vllm \
    --synthetic-input-tokens-mean 2048 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --concurrency 1 \
    --streaming &> genai_perf.log

echo "Performance results has been saved to genai_perf.log"

docker stop triton_ascend_310p > /dev/null
docker rm triton_ascend_310p > /dev/null
