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
    zhourunyu/triton_ascend_310p:260326 --model-repository=/triton_tests/4/models --model-control-mode=explicit --load-model=resnet50 > /dev/null

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

./test_triton -m resnet50
# ./test_triton_llm -m Qwen3-4B

python3 ./test_1.py

docker logs triton_ascend_310p &> triton_run_resnet50.log
echo "Server logs has been saved to triton_run_resnet50.log"

docker stop triton_ascend_310p > /dev/null
docker rm triton_ascend_310p > /dev/null