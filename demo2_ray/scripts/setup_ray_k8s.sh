#!/bin/bash
# 安装 KubeRay Operator 和部署 Ray 集群
set -e

echo "=== 安装 KubeRay Operator ==="

# 添加 Helm 仓库
echo "添加 KubeRay Helm 仓库..."
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# 安装 KubeRay Operator
echo "安装 KubeRay Operator..."
helm install kuberay-operator kuberay/kuberay-operator \
  --version 1.3.0 \
  --namespace kuberay-system \
  --create-namespace \
  --wait

echo "等待 Operator 就绪..."
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/name=kuberay-operator \
  -n kuberay-system \
  --timeout=120s

echo ""
echo "=== 部署 Ray 集群 ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

# 创建命名空间
echo "创建命名空间..."
kubectl apply -f "$K8S_DIR/namespace.yaml"

# 部署 Ray 集群
echo "部署 Ray 集群..."
kubectl apply -f "$K8S_DIR/ray-cluster.yaml"

echo "等待 Ray 集群就绪..."
kubectl wait --for=condition=ready pod \
  -l app=ray-demo \
  -n ray-demo \
  --timeout=300s

echo ""
echo "=== 部署完成 ==="
echo ""
echo "查看集群状态:"
echo "  kubectl get pods -n ray-demo"
echo ""
echo "访问 Dashboard:"
echo "  kubectl port-forward -n ray-demo svc/ray-demo-cluster-head-svc 8265:8265"
echo "  打开 http://localhost:8265"
echo ""
echo "连接 Ray Client:"
echo "  kubectl port-forward -n ray-demo svc/ray-demo-cluster-head-svc 10001:10001"
echo "  daft.set_runner_ray('ray://localhost:10001')"
