#!/bin/bash

echo "Running Triton Inference Server with Kubernetes..."

helm install triton .

# function wait_model() {
#     local ip=$(kubectl get endpointslices -l kubernetes.io/service-name=$1 -o jsonpath='{.items[0].endpoints[0].addresses[0]}')
#     while ! curl -fsS http://$ip:8000/v2/models/$2/ready > /dev/null; do
#         sleep 1
#     done
# }


# kubectl wait --for=condition=Ready pod -l app=triton --timeout=-1s
# wait_model triton-triton-ascend resnet50
# wait_model triton-triton-iluvatar resnet50
while ! curl -fsS http://localhost:8000/v2/health/ready &> /dev/null; do
    sleep 1
done
sleep 10
echo "Server is ready!"

python3 test_1.py
kubectl logs service/triton-traefik > traefik.log
echo "Traefik logs has been saved to traefik.log"

helm uninstall triton
