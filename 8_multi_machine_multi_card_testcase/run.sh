#!/bin/bash

echo "Running Triton Inference Server with Kubernetes..."

helm install triton .

while ! curl -fsS http://localhost:8000/v2/health/ready > /dev/null; do
    sleep 1
done

sleep 10

echo "Server is ready!"

python test_2.py

kubectl logs service/triton-traefik > traefik.log
echo "Traefik logs has been saved to traefik.log"

helm uninstall triton