#!/bin/bash

echo "Running Triton Inference Server with Kubernetes..."

helm install triton .

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

echo "Waiting for HPA metrics..."
while kubectl get hpa triton-triton-hpa-ascend 2>/dev/null | grep -q '<unknown>'; do
    sleep 5
done
echo "HPA metrics ready!"

kubectl describe hpa triton-triton-hpa-ascend > hpa_ascend_1.log
DESIRED=2
READY=$(kubectl get deployment triton-triton-ascend -o jsonpath='{.status.readyReplicas}')
echo "current: $READY/$DESIRED pods ready"

echo "run perf_analyzer -i grpc -m resnet50 --concurrency-range 50 --shape input:1,3,224,224"
perf_analyzer -i grpc -m resnet50 --concurrency-range 50 --shape input:1,3,224,224


# 等待扩容完成（desired 数量的 Pod 全部 Running）
echo "Waiting for scale-up to complete..."
DESIRED=2
while true; do
    READY=$(kubectl get deployment triton-triton-ascend -o jsonpath='{.status.readyReplicas}')
    if [ "$READY" = "$DESIRED" ] 2>/dev/null; then
        break
    fi
    sleep 5
done
echo "Scale-up complete: $READY/$DESIRED pods ready"


kubectl describe hpa triton-triton-hpa-ascend > hpa_ascend_2.log

echo "HPA details for ascend instance has been saved to hpa_ascend_1.log and hpa_ascend_2.log"

helm uninstall triton