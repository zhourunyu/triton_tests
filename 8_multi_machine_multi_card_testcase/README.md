# Triton K8s部署

使用k8s部署Triton Inference Server，提供负载均衡、自动伸缩等功能。

## 环境准备

### 安装K8s和Helm

可以使用[kubeadm](https://kubernetes.io/zh-cn/docs/reference/setup-tools/kubeadm/)、[k3s](https://docs.k3s.io/zh/quick-start)等工具配置K8s集群。

在Server节点执行如下命令安装Helm:

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 配置Container Runtime

为k8s使用的Container Runtime（如containerd）配置自定义runtime，用于运行昇腾与登临的容器：

```toml
# /etc/containerd/config.toml
[plugins.'io.containerd.grpc.v1.cri'.containerd.runtimes.ascend]
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.grpc.v1.cri'.containerd.runtimes.ascend.options]
  BinaryName = "/usr/local/Ascend/Ascend-Docker-Runtime/ascend-docker-runtime"

[plugins.'io.containerd.grpc.v1.cri'.containerd.runtimes.denglin]
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.grpc.v1.cri'.containerd.runtimes.denglin.options]
  BinaryName = "/usr/local/bin/dlgpu_runtime"
```

重启containerd服务：

```bash
systemctl restart containerd
```

在k8s中注册新的RuntimeClass：

```bash
kubectl apply -f runtimeclass.yaml
```

### 安装Device Plugin

安装NFD（Node Feature Discovery），用于发现节点上的设备：

```bash
helm install --namespace nfd --create-namespace nfd oci://registry.k8s.io/nfd/charts/node-feature-discovery
```

手动导入登临与天数的Device Plugin镜像：
```
ctr images import denglin-device-plugin:v0.10.11-2-gbff68c5-202512261226.tar
ctr images import ix-device-plugin-4.4.0.tar
```

安装Device Plugin：

```bash
cd device-plugin
kubectl apply -f ascend-device-plugin-310P.yaml
kubectl apply -f denglin_device_plugin.yaml
kubectl apply -f ix-device-plugin.yaml
```

使用`kubectl get pods -n kube-system`查看Device Plugin是否正常运行。

正常运行后，`kubectl get nodes -o custom-columns=NAME:.metadata.name,CAPACITY:.status.capacity`可以看到新增的设备资源，如`denglin.com/gpu`、`huawei.com/Ascend310P`、`iluvatar.com/gpu`等。

## 部署Triton Inference Server

安装Prometheus：

```bash
helm install triton-metrics --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false prometheus-community/kube-prometheus-stack
```

部署Triton Inference Server：

```bash
helm install triton .
```

`values.yaml`中的实例配置：

```yaml
instances:
  iluvatar:
    weight: 10        # 负载均衡权重，越大分配的请求越多
    autoscaling:
      enabled: false  # 是否启用自动伸缩
      minReplicas: 1  # 最小副本数，autoscaling为false时为固定副本数
      maxReplicas: 1  # 最大副本数
    resources:        # 单个副本的资源请求和限制
      limits:
        iluvatar.com/gpu: 1
    image: zhourunyu/triton_corex:260326  # Triton镜像
    modelRepository: /root/zhikai/models  # 模型存储路径
    serverArgs:
    - '--model-repository=/models'        # Triton启动参数
    env:
    - name: VLLM_ENFORCE_CUDA_GRAPH       # 环境变量
      value: "1"
```