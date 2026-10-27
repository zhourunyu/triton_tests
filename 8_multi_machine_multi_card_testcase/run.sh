#!/bin/bash

echo "Running Triton Inference Server with Kubernetes..."

helm install triton .

kubectl wait --for=condition=available deployment/triton-triton-ascend --timeout=300s
kubectl wait --for=condition=available deployment/triton-traefik --timeout=120s

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

python test_2.py

kubectl logs service/triton-traefik > traefik.log
echo "Traefik logs has been saved to traefik.log"

helm uninstall triton