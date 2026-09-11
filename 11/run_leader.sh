#!/bin/bash

echo "Starting Leader Node of Triton Inference Server on MR-V100..."

# -e NCCL_SOCKET_IFNAME=eth0 \
docker run -d --name triton_corex \
    --network host \
    --shm-size 1g \
    --runtime iluvatar -e IX_VISIBLE_DEVICES=0 \
    -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 \
    -v `pwd`/..:/triton_tests:ro \
    -v /root/weight:/weight:ro \
    --entrypoint /usr/local/bin/ray \
    zhourunyu/triton_corex:260326 start --head --port 6379 --num-gpus 1 --block --disable-usage-stats > /dev/null

while [ $(docker exec triton_corex python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))') -lt 2 ]; do
    sleep 1
done
echo "All ray nodes are ready!"

docker exec triton_corex /opt/tritonserver/bin/tritonserver --model-repository=/triton_tests/11/models

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

./test_triton_llm -m Qwen3-4B

# Run at the leader node
# docker exec triton_corex ray status
# docker exec triton_corex /opt/tritonserver/bin/tritonserver --model-repository=/triton_tests/11/models/zhikai
# ./test_triton_llm -m Qwen3-4B