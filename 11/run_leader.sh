#!/bin/bash

echo "Starting Leader Node of Triton Inference Server on MR-V100..."

docker run -d --name triton_corex \
    --network host \
    --shm-size 1g \
    --runtime iluvatar -e IX_VISIBLE_DEVICES=0 \
    -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    -e NCCL_SOCKET_IFNAME="eth1" \
    -e RAY_ADDRESS="auto" \
    -e VLLM_ENFORCE_CUDA_GRAPH=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    --entrypoint /usr/local/bin/ray \
    zhourunyu/triton_corex:260921 start --head --port 6379 --node-ip-address 192.168.10.218 --num-gpus 1 --block --disable-usage-stats > /dev/null

echo "Ray Head Started!"

while true; do
    NODE_COUNT=$(docker exec triton_corex python3 -c \
        'import ray; ray.init(address="auto"); print(sum(n["Alive"] for n in ray.nodes()))' 2>/dev/null || echo 0)
    if [ "$NODE_COUNT" -ge 2 ] 2>/dev/null; then
        break
    fi
    sleep 5
done
echo "All ray nodes are ready!"

docker exec -d triton_corex /opt/tritonserver/bin/tritonserver --model-repository=/triton_tests/11/models

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

./test_triton_llm -m Qwen3-4B
