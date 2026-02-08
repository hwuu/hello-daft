#!/bin/bash
# 清理 Ray on Kubernetes 资源
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

echo "=== 清理 Ray on Kubernetes 资源 ==="

# 删除 Ray 集群
if kubectl get raycluster ray-demo-cluster -n ray-demo &>/dev/null; then
    echo "删除 Ray 集群..."
    kubectl delete -f "$K8S_DIR/ray-cluster.yaml" --ignore-not-found
    echo "等待 Pod 终止..."
    kubectl wait --for=delete pod -l app=ray-demo -n ray-demo --timeout=60s 2>/dev/null || true
fi

# 删除命名空间
if kubectl get namespace ray-demo &>/dev/null; then
    echo "删除命名空间..."
    kubectl delete -f "$K8S_DIR/namespace.yaml" --ignore-not-found
fi

# 卸载 KubeRay Operator
if helm status kuberay-operator -n kuberay-system &>/dev/null; then
    echo "卸载 KubeRay Operator..."
    helm uninstall kuberay-operator -n kuberay-system
    kubectl delete namespace kuberay-system --ignore-not-found
fi

echo ""
echo "=== 清理完成 ==="
