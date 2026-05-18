"""
deployment_config.py — Model Deployment Pipeline & Configuration

End-to-end configuration and orchestration for deploying fine-tuned
financial LLMs to production serving infrastructure.

Supports:
  - Model registry with versioning and lineage tracking
  - Serving backends (vLLM, TGI, Triton, TorchServe, ONNX Runtime)
  - Auto-scaling policies (CPU / GPU utilization, request-based)
  - Canary and blue-green deployment strategies
  - Health checks and readiness probes
  - Model quantization for efficient serving (GPTQ, AWQ, ONNX)
  - Monitoring, alerting, and SLA configuration
  - Infrastructure provisioning (Kubernetes, Docker, cloud endpoints)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ServingBackend(str, Enum):
    """Model serving framework."""

    VLLM = "vllm"
    TGI = "tgi"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    ONNX_RUNTIME = "onnx_runtime"
    CUSTOM = "custom"


class DeploymentStrategy(str, Enum):
    """Deployment rollout strategy."""

    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class ModelFormat(str, Enum):
    """Model artifact format for serving."""

    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    GPTQ = "gptq"
    AWQ = "awq"


class InfrastructureTarget(str, Enum):
    """Deployment infrastructure target."""

    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    AWS_SAGEMAKER = "aws_sagemaker"
    GCP_VERTEX = "gcp_vertex"
    AZURE_ML = "azure_ml"
    LOCAL = "local"


class ScalingMetric(str, Enum):
    """Metric for auto-scaling decisions."""

    CPU_UTILIZATION = "cpu_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    LATENCY_P99 = "latency_p99"
    QUEUE_DEPTH = "queue_depth"


class HealthCheckType(str, Enum):
    """Health check probe type."""

    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    COMMAND = "command"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ResourceRequirements:
    """Compute resource specification for a deployment.

    Attributes:
        cpu_cores: Number of CPU cores (request / limit).
        memory_gb: Memory in gigabytes.
        gpu_count: Number of GPUs.
        gpu_type: GPU type identifier (e.g. "nvidia-a100-80gb").
        storage_gb: Persistent storage in gigabytes.
        shared_memory_gb: Shared memory (/dev/shm) size.
    """

    cpu_cores: float = 4.0
    memory_gb: float = 16.0
    gpu_count: int = 1
    gpu_type: str = "nvidia-a100-80gb"
    storage_gb: float = 100.0
    shared_memory_gb: float = 8.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def to_k8s_resources(self) -> Dict[str, Dict[str, str]]:
        """Convert to Kubernetes resource spec."""
        resources: Dict[str, Dict[str, str]] = {
            "requests": {
                "cpu": str(self.cpu_cores),
                "memory": f"{self.memory_gb}Gi",
            },
            "limits": {
                "cpu": str(self.cpu_cores),
                "memory": f"{self.memory_gb}Gi",
            },
        }
        if self.gpu_count > 0:
            resources["limits"]["nvidia.com/gpu"] = str(self.gpu_count)
        return resources


@dataclass
class HealthCheckConfig:
    """Health check / readiness probe configuration.

    Attributes:
        check_type: Probe type (HTTP, TCP, gRPC, command).
        endpoint: Path or address to probe.
        port: Port to probe.
        initial_delay_seconds: Seconds to wait before first probe.
        period_seconds: Interval between probes.
        timeout_seconds: Probe timeout.
        failure_threshold: Failures before marking unhealthy.
        success_threshold: Successes before marking healthy.
    """

    check_type: HealthCheckType = HealthCheckType.HTTP
    endpoint: str = "/health"
    port: int = 8080
    initial_delay_seconds: int = 30
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["check_type"] = self.check_type.value
        return data


@dataclass
class AutoScalingConfig:
    """Auto-scaling policy configuration.

    Attributes:
        enabled: Whether auto-scaling is active.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        target_metric: Metric to scale on.
        target_value: Target metric value.
        scale_up_cooldown_seconds: Cooldown after scale-up.
        scale_down_cooldown_seconds: Cooldown after scale-down.
        scale_up_step: Max replicas to add per scaling event.
        scale_down_step: Max replicas to remove per scaling event.
    """

    enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 8
    target_metric: ScalingMetric = ScalingMetric.GPU_UTILIZATION
    target_value: float = 70.0
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    scale_up_step: int = 2
    scale_down_step: int = 1

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["target_metric"] = self.target_metric.value
        return data


@dataclass
class ServingConfig:
    """Model serving configuration.

    Attributes:
        backend: Serving framework to use.
        model_format: Model artifact format.
        host: Bind address for the serving endpoint.
        port: Port for the serving endpoint.
        max_batch_size: Maximum inference batch size.
        max_concurrent_requests: Maximum concurrent requests.
        max_sequence_length: Maximum input sequence length.
        max_new_tokens: Default maximum generation length.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        dtype: Model weight dtype for serving.
        quantization_method: Post-training quantization method.
        enable_streaming: Enable streaming responses.
        enable_prefix_caching: Enable KV-cache prefix caching.
        gpu_memory_utilization: Fraction of GPU memory to use.
        swap_space_gb: CPU swap space for KV-cache overflow.
        trust_remote_code: Trust remote model code.
    """

    backend: ServingBackend = ServingBackend.VLLM
    model_format: ModelFormat = ModelFormat.SAFETENSORS
    host: str = "0.0.0.0"
    port: int = 8080
    max_batch_size: int = 64
    max_concurrent_requests: int = 128
    max_sequence_length: int = 4096
    max_new_tokens: int = 512
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    quantization_method: Optional[str] = None
    enable_streaming: bool = True
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.90
    swap_space_gb: float = 4.0
    trust_remote_code: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["backend"] = self.backend.value
        data["model_format"] = self.model_format.value
        return data


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration.

    Attributes:
        enabled: Enable monitoring.
        metrics_port: Port for Prometheus metrics.
        metrics_path: Path for metrics endpoint.
        log_level: Logging level for the serving process.
        enable_request_logging: Log individual requests.
        latency_sla_p50_ms: Target p50 latency in ms.
        latency_sla_p99_ms: Target p99 latency in ms.
        error_rate_threshold: Max acceptable error rate.
        alert_webhook_url: Webhook for firing alerts.
        enable_tracing: Enable distributed tracing.
        tracing_endpoint: OpenTelemetry collector endpoint.
    """

    enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    log_level: str = "INFO"
    enable_request_logging: bool = False
    latency_sla_p50_ms: float = 200.0
    latency_sla_p99_ms: float = 1000.0
    error_rate_threshold: float = 0.01
    alert_webhook_url: Optional[str] = None
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CanaryConfig:
    """Canary deployment configuration.

    Attributes:
        enabled: Whether to use canary deployments.
        initial_weight_percent: Initial traffic percentage to canary.
        weight_increment: Percentage to increase per step.
        promotion_interval_seconds: Time between promotion steps.
        error_rate_rollback_threshold: Error rate that triggers rollback.
        latency_rollback_threshold_ms: p99 latency that triggers rollback.
        min_evaluation_requests: Minimum requests before evaluation.
    """

    enabled: bool = False
    initial_weight_percent: float = 10.0
    weight_increment: float = 20.0
    promotion_interval_seconds: int = 300
    error_rate_rollback_threshold: float = 0.05
    latency_rollback_threshold_ms: float = 2000.0
    min_evaluation_requests: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Top-level deployment configuration
# ---------------------------------------------------------------------------


@dataclass
class DeploymentConfig:
    """Complete deployment configuration for a financial LLM.

    Aggregates all sub-configurations and provides serialization,
    validation, and infrastructure generation utilities.

    Attributes:
        model_name: Display name for the deployed model.
        model_version: Semantic version string.
        model_path: Path to the model artifacts.
        deployment_name: Unique deployment identifier.
        namespace: Kubernetes namespace or cloud project.
        infrastructure: Target deployment platform.
        strategy: Deployment rollout strategy.
        replicas: Number of serving replicas.
        resources: Compute resource requirements.
        serving: Serving framework configuration.
        scaling: Auto-scaling configuration.
        health_check: Health/readiness probe configuration.
        monitoring: Monitoring and alerting configuration.
        canary: Canary deployment configuration.
        environment_variables: Extra env vars for the serving container.
        labels: Metadata labels.
        annotations: Metadata annotations.
    """

    model_name: str = "dhan-financial-llm"
    model_version: str = "1.0.0"
    model_path: str = "./output/dhan_training/final_model"
    deployment_name: str = "dhan-llm-serving"
    namespace: str = "dhan-ai"
    infrastructure: InfrastructureTarget = InfrastructureTarget.KUBERNETES
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    replicas: int = 2

    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    serving: ServingConfig = field(default_factory=ServingConfig)
    scaling: AutoScalingConfig = field(default_factory=AutoScalingConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    canary: CanaryConfig = field(default_factory=CanaryConfig)

    environment_variables: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(
        default_factory=lambda: {
            "app": "dhan-ai",
            "component": "llm-serving",
            "managed-by": "dhan-deployment",
        }
    )
    annotations: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full deployment config to a dict."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "deployment_name": self.deployment_name,
            "namespace": self.namespace,
            "infrastructure": self.infrastructure.value,
            "strategy": self.strategy.value,
            "replicas": self.replicas,
            "resources": self.resources.to_dict(),
            "serving": self.serving.to_dict(),
            "scaling": self.scaling.to_dict(),
            "health_check": self.health_check.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "canary": self.canary.to_dict(),
            "environment_variables": self.environment_variables,
            "labels": self.labels,
            "annotations": self.annotations,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save deployment config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Deployment config saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeploymentConfig":
        """Load deployment config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "DeploymentConfig":
        """Reconstruct a DeploymentConfig from a dict."""
        resources = ResourceRequirements(**data.pop("resources", {}))
        serving_data = data.pop("serving", {})
        if "backend" in serving_data:
            serving_data["backend"] = ServingBackend(serving_data["backend"])
        if "model_format" in serving_data:
            serving_data["model_format"] = ModelFormat(serving_data["model_format"])
        serving = ServingConfig(**serving_data)

        scaling_data = data.pop("scaling", {})
        if "target_metric" in scaling_data:
            scaling_data["target_metric"] = ScalingMetric(scaling_data["target_metric"])
        scaling = AutoScalingConfig(**scaling_data)

        hc_data = data.pop("health_check", {})
        if "check_type" in hc_data:
            hc_data["check_type"] = HealthCheckType(hc_data["check_type"])
        health_check = HealthCheckConfig(**hc_data)

        monitoring = MonitoringConfig(**data.pop("monitoring", {}))
        canary = CanaryConfig(**data.pop("canary", {}))

        if "infrastructure" in data:
            data["infrastructure"] = InfrastructureTarget(data["infrastructure"])
        if "strategy" in data:
            data["strategy"] = DeploymentStrategy(data["strategy"])

        return cls(
            resources=resources,
            serving=serving,
            scaling=scaling,
            health_check=health_check,
            monitoring=monitoring,
            canary=canary,
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Validate the deployment configuration. Returns a list of issues."""
        issues: List[str] = []

        if self.replicas < 1:
            issues.append("replicas must be >= 1")

        if self.scaling.enabled:
            if self.scaling.min_replicas > self.scaling.max_replicas:
                issues.append("scaling.min_replicas > scaling.max_replicas")
            if self.scaling.min_replicas < 1:
                issues.append("scaling.min_replicas must be >= 1")

        if self.serving.gpu_memory_utilization <= 0 or self.serving.gpu_memory_utilization > 1.0:
            issues.append("serving.gpu_memory_utilization must be in (0, 1.0]")

        if self.serving.tensor_parallel_size > self.resources.gpu_count:
            issues.append(
                f"tensor_parallel_size ({self.serving.tensor_parallel_size}) "
                f"> gpu_count ({self.resources.gpu_count})"
            )

        if self.resources.gpu_count > 0 and not self.resources.gpu_type:
            issues.append("gpu_type must be specified when gpu_count > 0")

        model_path = Path(self.model_path)
        if not model_path.exists():
            issues.append(f"model_path does not exist: {self.model_path}")

        if self.canary.enabled and self.strategy != DeploymentStrategy.CANARY:
            issues.append(
                "canary.enabled is True but strategy is not CANARY"
            )

        if self.monitoring.error_rate_threshold < 0 or self.monitoring.error_rate_threshold > 1:
            issues.append("monitoring.error_rate_threshold must be in [0, 1]")

        return issues

    # ------------------------------------------------------------------
    # Infrastructure generation
    # ------------------------------------------------------------------

    def generate_kubernetes_manifest(self) -> Dict[str, Any]:
        """Generate a Kubernetes Deployment manifest.

        Returns a dict representing a K8s Deployment resource
        that can be serialized to YAML.
        """
        container_env = [
            {"name": k, "value": v}
            for k, v in self.environment_variables.items()
        ]

        container_args = self._build_serving_args()

        liveness_probe = self._build_k8s_probe(self.health_check)
        readiness_probe = self._build_k8s_probe(self.health_check)
        readiness_probe["initialDelaySeconds"] = max(
            self.health_check.initial_delay_seconds // 2, 5
        )

        container = {
            "name": self.deployment_name,
            "image": self._resolve_serving_image(),
            "ports": [
                {"containerPort": self.serving.port, "name": "serving"},
                {"containerPort": self.monitoring.metrics_port, "name": "metrics"},
            ],
            "env": container_env,
            "args": container_args,
            "resources": self.resources.to_k8s_resources(),
            "livenessProbe": liveness_probe,
            "readinessProbe": readiness_probe,
            "volumeMounts": [
                {"name": "model-storage", "mountPath": "/models"},
                {"name": "dshm", "mountPath": "/dev/shm"},
            ],
        }

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_name,
                "namespace": self.namespace,
                "labels": self.labels,
                "annotations": self.annotations,
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"app": self.labels.get("app", self.deployment_name)}},
                "strategy": self._build_k8s_strategy(),
                "template": {
                    "metadata": {
                        "labels": {
                            **self.labels,
                            "model-version": self.model_version,
                        },
                    },
                    "spec": {
                        "containers": [container],
                        "volumes": [
                            {
                                "name": "model-storage",
                                "persistentVolumeClaim": {"claimName": f"{self.deployment_name}-pvc"},
                            },
                            {
                                "name": "dshm",
                                "emptyDir": {
                                    "medium": "Memory",
                                    "sizeLimit": f"{self.resources.shared_memory_gb}Gi",
                                },
                            },
                        ],
                        "nodeSelector": (
                            {"nvidia.com/gpu.product": self.resources.gpu_type}
                            if self.resources.gpu_count > 0
                            else {}
                        ),
                        "tolerations": (
                            [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
                            if self.resources.gpu_count > 0
                            else []
                        ),
                    },
                },
            },
        }

        return manifest

    def generate_service_manifest(self) -> Dict[str, Any]:
        """Generate a Kubernetes Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.deployment_name}-svc",
                "namespace": self.namespace,
                "labels": self.labels,
            },
            "spec": {
                "type": "ClusterIP",
                "selector": {"app": self.labels.get("app", self.deployment_name)},
                "ports": [
                    {
                        "name": "serving",
                        "port": self.serving.port,
                        "targetPort": self.serving.port,
                        "protocol": "TCP",
                    },
                    {
                        "name": "metrics",
                        "port": self.monitoring.metrics_port,
                        "targetPort": self.monitoring.metrics_port,
                        "protocol": "TCP",
                    },
                ],
            },
        }

    def generate_hpa_manifest(self) -> Optional[Dict[str, Any]]:
        """Generate a Kubernetes HorizontalPodAutoscaler manifest."""
        if not self.scaling.enabled:
            return None

        metric_map = {
            ScalingMetric.CPU_UTILIZATION: {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": int(self.scaling.target_value),
                    },
                },
            },
            ScalingMetric.GPU_UTILIZATION: {
                "type": "Pods",
                "pods": {
                    "metric": {"name": "gpu_utilization"},
                    "target": {
                        "type": "AverageValue",
                        "averageValue": str(int(self.scaling.target_value)),
                    },
                },
            },
            ScalingMetric.REQUEST_RATE: {
                "type": "Pods",
                "pods": {
                    "metric": {"name": "http_requests_per_second"},
                    "target": {
                        "type": "AverageValue",
                        "averageValue": str(int(self.scaling.target_value)),
                    },
                },
            },
        }

        metrics_spec = metric_map.get(
            self.scaling.target_metric,
            metric_map[ScalingMetric.CPU_UTILIZATION],
        )

        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.deployment_name}-hpa",
                "namespace": self.namespace,
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name,
                },
                "minReplicas": self.scaling.min_replicas,
                "maxReplicas": self.scaling.max_replicas,
                "metrics": [metrics_spec],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": self.scaling.scale_up_cooldown_seconds,
                        "policies": [
                            {
                                "type": "Pods",
                                "value": self.scaling.scale_up_step,
                                "periodSeconds": self.scaling.scale_up_cooldown_seconds,
                            }
                        ],
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": self.scaling.scale_down_cooldown_seconds,
                        "policies": [
                            {
                                "type": "Pods",
                                "value": self.scaling.scale_down_step,
                                "periodSeconds": self.scaling.scale_down_cooldown_seconds,
                            }
                        ],
                    },
                },
            },
        }

    def generate_docker_compose(self) -> Dict[str, Any]:
        """Generate a Docker Compose service definition."""
        service: Dict[str, Any] = {
            "image": self._resolve_serving_image(),
            "ports": [
                f"{self.serving.port}:{self.serving.port}",
                f"{self.monitoring.metrics_port}:{self.monitoring.metrics_port}",
            ],
            "environment": {
                **self.environment_variables,
                "MODEL_PATH": "/models",
            },
            "command": self._build_serving_args(),
            "volumes": [
                f"{self.model_path}:/models:ro",
            ],
            "healthcheck": {
                "test": [
                    "CMD",
                    "curl",
                    "-f",
                    f"http://localhost:{self.serving.port}{self.health_check.endpoint}",
                ],
                "interval": f"{self.health_check.period_seconds}s",
                "timeout": f"{self.health_check.timeout_seconds}s",
                "retries": self.health_check.failure_threshold,
                "start_period": f"{self.health_check.initial_delay_seconds}s",
            },
            "deploy": {
                "replicas": self.replicas,
            },
            "shm_size": f"{self.resources.shared_memory_gb}g",
            "restart": "unless-stopped",
        }

        if self.resources.gpu_count > 0:
            service["deploy"]["resources"] = {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "count": self.resources.gpu_count,
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }

        return {
            "version": "3.8",
            "services": {self.deployment_name: service},
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _resolve_serving_image(self) -> str:
        """Resolve the Docker image for the serving backend."""
        image_map = {
            ServingBackend.VLLM: "vllm/vllm-openai:latest",
            ServingBackend.TGI: "ghcr.io/huggingface/text-generation-inference:latest",
            ServingBackend.TRITON: "nvcr.io/nvidia/tritonserver:24.01-py3",
            ServingBackend.TORCHSERVE: "pytorch/torchserve:latest-gpu",
            ServingBackend.ONNX_RUNTIME: "mcr.microsoft.com/onnxruntime/server:latest",
        }
        return image_map.get(self.serving.backend, "python:3.11-slim")

    def _build_serving_args(self) -> List[str]:
        """Build command-line arguments for the serving backend."""
        if self.serving.backend == ServingBackend.VLLM:
            args = [
                "--model", "/models",
                "--host", self.serving.host,
                "--port", str(self.serving.port),
                "--tensor-parallel-size", str(self.serving.tensor_parallel_size),
                "--max-model-len", str(self.serving.max_sequence_length),
                "--dtype", self.serving.dtype,
                "--gpu-memory-utilization", str(self.serving.gpu_memory_utilization),
                "--swap-space", str(int(self.serving.swap_space_gb)),
                "--max-num-seqs", str(self.serving.max_concurrent_requests),
            ]
            if self.serving.enable_prefix_caching:
                args.append("--enable-prefix-caching")
            if self.serving.quantization_method:
                args.extend(["--quantization", self.serving.quantization_method])
            if self.serving.trust_remote_code:
                args.append("--trust-remote-code")
            return args

        elif self.serving.backend == ServingBackend.TGI:
            args = [
                "--model-id", "/models",
                "--hostname", self.serving.host,
                "--port", str(self.serving.port),
                "--num-shard", str(self.serving.tensor_parallel_size),
                "--max-input-length", str(self.serving.max_sequence_length),
                "--max-total-tokens", str(
                    self.serving.max_sequence_length + self.serving.max_new_tokens
                ),
                "--max-batch-total-tokens", str(
                    self.serving.max_batch_size
                    * (self.serving.max_sequence_length + self.serving.max_new_tokens)
                ),
                "--dtype", self.serving.dtype,
            ]
            if self.serving.quantization_method:
                args.extend(["--quantize", self.serving.quantization_method])
            return args

        return []

    def _build_k8s_probe(self, hc: HealthCheckConfig) -> Dict[str, Any]:
        """Convert a HealthCheckConfig to a Kubernetes probe spec."""
        probe: Dict[str, Any] = {
            "initialDelaySeconds": hc.initial_delay_seconds,
            "periodSeconds": hc.period_seconds,
            "timeoutSeconds": hc.timeout_seconds,
            "failureThreshold": hc.failure_threshold,
            "successThreshold": hc.success_threshold,
        }
        if hc.check_type == HealthCheckType.HTTP:
            probe["httpGet"] = {"path": hc.endpoint, "port": hc.port}
        elif hc.check_type == HealthCheckType.TCP:
            probe["tcpSocket"] = {"port": hc.port}
        elif hc.check_type == HealthCheckType.GRPC:
            probe["grpc"] = {"port": hc.port}
        elif hc.check_type == HealthCheckType.COMMAND:
            probe["exec"] = {"command": [hc.endpoint]}
        return probe

    def _build_k8s_strategy(self) -> Dict[str, Any]:
        """Build the Kubernetes deployment strategy spec."""
        if self.strategy == DeploymentStrategy.RECREATE:
            return {"type": "Recreate"}

        return {
            "type": "RollingUpdate",
            "rollingUpdate": {
                "maxSurge": "25%",
                "maxUnavailable": 0,
            },
        }


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Local model registry for tracking model versions and lineage.

    Stores model metadata, metrics, and deployment history in a
    JSON-based registry on the local filesystem.

    Example::

        registry = ModelRegistry("./model_registry")
        registry.register(
            model_name="dhan-financial-llm",
            version="1.0.0",
            model_path="./output/final_model",
            metrics={"eval_loss": 0.42, "eval_perplexity": 1.52},
            training_config={"epochs": 3, "lr": 2e-5},
        )
        latest = registry.get_latest("dhan-financial-llm")
    """

    def __init__(self, registry_dir: Union[str, Path]) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "registry_index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            with open(self._index_path) as f:
                return json.load(f)
        return {"models": {}}

    def _save_index(self) -> None:
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    def register(
        self,
        model_name: str,
        version: str,
        model_path: Union[str, Path],
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Register a new model version.

        Args:
            model_name: Name of the model.
            version: Version string (e.g. "1.0.0").
            model_path: Path to model artifacts.
            metrics: Evaluation metrics.
            training_config: Training configuration used.
            tags: Key-value tags for filtering.
            description: Human-readable description.

        Returns:
            The registry entry for the registered model.
        """
        model_path = Path(model_path)
        checksum = self._compute_checksum(model_path)

        entry = {
            "model_name": model_name,
            "version": version,
            "model_path": str(model_path.resolve()),
            "checksum": checksum,
            "metrics": metrics or {},
            "training_config": training_config or {},
            "tags": tags or {},
            "description": description,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "registered",
        }

        if model_name not in self._index["models"]:
            self._index["models"][model_name] = {"versions": {}}

        self._index["models"][model_name]["versions"][version] = entry
        self._index["models"][model_name]["latest_version"] = version
        self._save_index()

        version_dir = self.registry_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(entry, f, indent=2, default=str)

        logger.info("Registered model %s v%s (checksum=%s)", model_name, version, checksum[:12])
        return entry

    def get_latest(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest registered version of a model."""
        model_data = self._index["models"].get(model_name)
        if not model_data:
            return None
        latest = model_data.get("latest_version")
        if not latest:
            return None
        return model_data["versions"].get(latest)

    def get_version(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get a specific model version."""
        model_data = self._index["models"].get(model_name)
        if not model_data:
            return None
        return model_data["versions"].get(version)

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._index["models"].keys())

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        model_data = self._index["models"].get(model_name, {})
        return list(model_data.get("versions", {}).keys())

    def compare_versions(
        self, model_name: str, version_a: str, version_b: str
    ) -> Dict[str, Any]:
        """Compare metrics between two model versions."""
        a = self.get_version(model_name, version_a)
        b = self.get_version(model_name, version_b)
        if not a or not b:
            raise ValueError("One or both versions not found")

        comparison: Dict[str, Any] = {"version_a": version_a, "version_b": version_b, "metrics": {}}
        all_metric_keys = set(a["metrics"].keys()) | set(b["metrics"].keys())
        for key in sorted(all_metric_keys):
            val_a = a["metrics"].get(key)
            val_b = b["metrics"].get(key)
            diff = None
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
            comparison["metrics"][key] = {
                "version_a": val_a,
                "version_b": val_b,
                "difference": diff,
            }
        return comparison

    def promote(self, model_name: str, version: str, stage: str = "production") -> None:
        """Promote a model version to a deployment stage."""
        model_data = self._index["models"].get(model_name)
        if not model_data or version not in model_data["versions"]:
            raise ValueError(f"Model {model_name} v{version} not found")
        model_data["versions"][version]["status"] = stage
        model_data[f"{stage}_version"] = version
        self._save_index()
        logger.info("Promoted %s v%s to %s", model_name, version, stage)

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA-256 checksum of model artifacts."""
        sha = hashlib.sha256()
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
        elif path.is_dir():
            for fpath in sorted(path.rglob("*")):
                if fpath.is_file():
                    sha.update(str(fpath.relative_to(path)).encode())
                    with open(fpath, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha.update(chunk)
        return sha.hexdigest()


# ---------------------------------------------------------------------------
# Deployment Pipeline
# ---------------------------------------------------------------------------


class DeploymentPipeline:
    """Orchestrates the end-to-end model deployment workflow.

    Coordinates model validation, artifact preparation, infrastructure
    provisioning, and deployment execution.

    Example::

        config = DeploymentConfig(
            model_name="dhan-financial-llm",
            model_version="1.0.0",
            model_path="./output/final_model",
            infrastructure=InfrastructureTarget.KUBERNETES,
        )
        registry = ModelRegistry("./model_registry")

        pipeline = DeploymentPipeline(config, registry)
        pipeline.deploy()
    """

    def __init__(
        self,
        config: DeploymentConfig,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self._deployment_id = f"{config.deployment_name}-{int(time.time())}"

    def deploy(self) -> Dict[str, Any]:
        """Execute the full deployment pipeline.

        Steps:
            1. Validate configuration
            2. Register model in registry
            3. Prepare model artifacts
            4. Generate infrastructure manifests
            5. Apply deployment
            6. Verify health

        Returns:
            Deployment result with status and details.
        """
        result: Dict[str, Any] = {
            "deployment_id": self._deployment_id,
            "status": "pending",
            "steps": [],
        }

        try:
            self._step(result, "validate", self._validate)
            self._step(result, "register", self._register_model)
            self._step(result, "prepare_artifacts", self._prepare_artifacts)
            self._step(result, "generate_manifests", self._generate_manifests)
            self._step(result, "apply", self._apply_deployment)
            self._step(result, "verify", self._verify_health)

            result["status"] = "deployed"
            logger.info("Deployment %s completed successfully", self._deployment_id)

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error("Deployment %s failed: %s", self._deployment_id, e)

        return result

    def rollback(self, previous_version: str) -> Dict[str, Any]:
        """Roll back to a previous model version.

        Args:
            previous_version: Version string to roll back to.

        Returns:
            Rollback result.
        """
        logger.info(
            "Rolling back %s from v%s to v%s",
            self.config.model_name,
            self.config.model_version,
            previous_version,
        )

        if self.registry:
            prev = self.registry.get_version(self.config.model_name, previous_version)
            if not prev:
                return {"status": "failed", "error": f"Version {previous_version} not found"}
            self.config.model_version = previous_version
            self.config.model_path = prev["model_path"]

        return self.deploy()

    def generate_all_manifests(self) -> Dict[str, Any]:
        """Generate all infrastructure manifests without deploying.

        Returns:
            Dict of manifest type -> manifest content.
        """
        manifests: Dict[str, Any] = {}

        if self.config.infrastructure in (
            InfrastructureTarget.KUBERNETES,
            InfrastructureTarget.LOCAL,
        ):
            manifests["deployment"] = self.config.generate_kubernetes_manifest()
            manifests["service"] = self.config.generate_service_manifest()
            hpa = self.config.generate_hpa_manifest()
            if hpa:
                manifests["hpa"] = hpa

        if self.config.infrastructure in (
            InfrastructureTarget.DOCKER,
            InfrastructureTarget.LOCAL,
        ):
            manifests["docker_compose"] = self.config.generate_docker_compose()

        return manifests

    def save_manifests(self, output_dir: Union[str, Path]) -> List[Path]:
        """Generate and save all manifests to disk.

        Returns:
            List of file paths written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests = self.generate_all_manifests()
        written: List[Path] = []

        for name, content in manifests.items():
            file_path = output_dir / f"{name}.json"
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2)
            written.append(file_path)
            logger.info("Saved manifest: %s", file_path)

        config_path = output_dir / "deployment_config.json"
        self.config.save(config_path)
        written.append(config_path)

        return written

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _step(
        self,
        result: Dict[str, Any],
        name: str,
        fn: Callable[[], Dict[str, Any]],
    ) -> None:
        """Execute a pipeline step and record the result."""
        start = time.time()
        step_result = fn()
        step_result["duration_seconds"] = round(time.time() - start, 2)
        step_result["name"] = name
        result["steps"].append(step_result)

        if step_result.get("status") == "failed":
            raise RuntimeError(
                f"Step '{name}' failed: {step_result.get('error', 'unknown')}"
            )

    def _validate(self) -> Dict[str, Any]:
        """Validate the deployment configuration."""
        issues = self.config.validate()
        if issues:
            return {"status": "failed", "error": f"Validation errors: {issues}"}
        return {"status": "passed", "message": "Configuration is valid"}

    def _register_model(self) -> Dict[str, Any]:
        """Register the model in the registry."""
        if self.registry is None:
            return {"status": "skipped", "message": "No registry configured"}

        entry = self.registry.register(
            model_name=self.config.model_name,
            version=self.config.model_version,
            model_path=self.config.model_path,
            tags=self.config.labels,
        )
        return {"status": "passed", "registry_entry": entry}

    def _prepare_artifacts(self) -> Dict[str, Any]:
        """Prepare model artifacts for serving."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            return {"status": "failed", "error": f"Model path not found: {model_path}"}

        artifact_files = list(model_path.rglob("*"))
        total_size = sum(f.stat().st_size for f in artifact_files if f.is_file())

        return {
            "status": "passed",
            "artifact_count": len(artifact_files),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
        }

    def _generate_manifests(self) -> Dict[str, Any]:
        """Generate infrastructure manifests."""
        manifests = self.generate_all_manifests()
        return {
            "status": "passed",
            "manifest_types": list(manifests.keys()),
        }

    def _apply_deployment(self) -> Dict[str, Any]:
        """Apply the deployment to the target infrastructure."""
        if self.config.infrastructure == InfrastructureTarget.LOCAL:
            return self._apply_local()
        elif self.config.infrastructure == InfrastructureTarget.DOCKER:
            return self._apply_docker()
        elif self.config.infrastructure == InfrastructureTarget.KUBERNETES:
            return self._apply_kubernetes()
        else:
            return {
                "status": "passed",
                "message": (
                    f"Manifests generated for {self.config.infrastructure.value}. "
                    "Manual deployment required."
                ),
            }

    def _apply_local(self) -> Dict[str, Any]:
        """Start a local serving process."""
        logger.info("Local deployment: manifests ready for manual startup")
        return {
            "status": "passed",
            "message": "Local manifests generated. Use Docker Compose or direct CLI to start.",
        }

    def _apply_docker(self) -> Dict[str, Any]:
        """Deploy via Docker Compose."""
        compose = self.config.generate_docker_compose()
        compose_path = Path(self.config.model_path).parent / "docker-compose.yml"
        with open(compose_path, "w") as f:
            json.dump(compose, f, indent=2)

        return {
            "status": "passed",
            "message": f"Docker Compose file written to {compose_path}",
            "compose_path": str(compose_path),
        }

    def _apply_kubernetes(self) -> Dict[str, Any]:
        """Deploy to Kubernetes cluster."""
        manifest_dir = Path(self.config.model_path).parent / "k8s_manifests"
        written = self.save_manifests(manifest_dir)

        return {
            "status": "passed",
            "message": f"Kubernetes manifests written to {manifest_dir}",
            "manifest_files": [str(p) for p in written],
            "apply_command": f"kubectl apply -f {manifest_dir}/ -n {self.config.namespace}",
        }

    def _verify_health(self) -> Dict[str, Any]:
        """Verify that the deployment is healthy (placeholder for live checks)."""
        return {
            "status": "passed",
            "message": (
                "Health verification is pending. "
                f"Check endpoint: http://localhost:{self.config.serving.port}"
                f"{self.config.health_check.endpoint}"
            ),
        }
