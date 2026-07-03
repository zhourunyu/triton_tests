#!/bin/bash

echo "Running Triton Inference Server with Kubernetes..."

helm install triton .

while ! curl -s http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done
echo "Server is ready!"

python3 test_1.py
kubectl logs -l service=triton-traefik --tail=-1 > traefik.log
echo "Traefik logs has been saved to traefik.log"