#!/bin/bash

echo "Running Triton Inference Server with Kubernetes..."

helm install triton . --set instances.ascend.autoscaling.minReplicas=0 --set instances.iluvatar.autoscaling.enabled=true

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

perf_analyzer -i grpc -m resnet50 --concurrency-range 50 --shape input:1,3,224,224

kubectl describe hpa triton-ascend > hpa_ascend.log
echo "HPA details for Ascend instance has been saved to hpa_ascend.log"