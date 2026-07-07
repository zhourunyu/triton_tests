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

./test_triton -m resnet50
./test_triton -m mobilenet_v2
./test_triton -m yolov5s
./test_triton -m bert-base-cased
# ./test_triton -m rnn
# ./test_triton -m gru
# ./test_triton -m lstm
./test_triton_llm -m Qwen3-4B

perf_analyzer -i grpc -m resnet50 --concurrency-range 50 --shape input:1,3,224,224 &> perf.log
echo "Performance results has been saved to perf.log"
genai-perf profile \
    -u localhost:8001 \
    -m Qwen3-4B \
    --tokenizer Qwen/Qwen3-4B \
    --backend vllm \
    --synthetic-input-tokens-mean 2048 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --output-tokens-stddev 0 \
    --concurrency 50 \
    --streaming &> genai_perf.log
echo "GenAI performance results has been saved to genai_perf.log"

docker stop triton_corex > /dev/null
docker rm triton_corex > /dev/null