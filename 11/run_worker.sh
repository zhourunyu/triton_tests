#!/bin/bash

echo "Starting Worker Node of Triton Inference Server on MR-V100..."

# -e NCCL_SOCKET_IFNAME=eth0 \
docker run -d --name triton_corex \
    --network host \
    --shm-size 1g \
    --runtime iluvatar -e IX_VISIBLE_DEVICES=0 \
    -v /root/weight:/weight:ro \
    --entrypoint /usr/local/bin/ray \
    zhourunyu/triton_corex:260326 start --address 192.168.10.218:6379 --num-gpus 1 --block --disable-usage-stats > /dev/null