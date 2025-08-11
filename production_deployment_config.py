#!/usr/bin/env python3
"""
本番環境デプロイメント設定
Phase G: 本番運用最適化フェーズ

本番環境への安全なデプロイメントのための設定管理システム
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class DeploymentEnvironment(Enum):
    """デプロイメント環境"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ServiceTier(Enum):
    """サービスレベル"""

    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class DatabaseConfig:
    """データベース設定"""

    host: str
    port: int
    database: str
    username: str
    password_env_var: str
    ssl_enabled: bool = True
    connection_pool_size: int = 20
    connection_timeout: int = 30
    query_timeout: int = 60


@dataclass
class CacheConfig:
    """キャッシュ設定"""

    backend: str  # redis, memory, memcached
    host: str
    port: int
    password_env_var: Optional[str] = None
    ttl_seconds: int = 3600
    max_memory_mb: int = 512
    eviction_policy: str = "allkeys-lru"


@dataclass
class ApiConfig:
    """API設定"""

    host: str
    port: int
    workers: int
    threads_per_worker: int
    max_requests_per_worker: int
    timeout_seconds: int
    rate_limit_per_minute: int
    cors_origins: List[str]
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


@dataclass
class MonitoringConfig:
    """監視設定"""

    enabled: bool
    metrics_endpoint: str
    log_level: str
    log_format: str
    prometheus_port: int
    jaeger_endpoint: Optional[str] = None
    error_reporting_dsn: Optional[str] = None


@dataclass
class SecurityConfig:
    """セキュリティ設定"""

    jwt_secret_env_var: str
    api_key_env_var: str
    encryption_key_env_var: str
    allowed_hosts: List[str]
    csrf_protection: bool
    session_timeout_minutes: int
    password_policy: Dict[str, Any]


@dataclass
class PerformanceConfig:
    """パフォーマンス設定"""

    optimization_level: str
    gpu_enabled: bool
    pytorch_enabled: bool
    batch_size: int
    cache_enabled: bool
    async_processing: bool
    connection_pooling: bool
    query_caching: bool


@dataclass
class DeploymentConfig:
    """デプロイメント設定"""

    environment: DeploymentEnvironment
    service_tier: ServiceTier
    version: str
    deployment_id: str
    timestamp: datetime

    # コンポーネント設定
    database: DatabaseConfig
    cache: CacheConfig
    api: ApiConfig
    monitoring: MonitoringConfig
    security: SecurityConfig
    performance: PerformanceConfig

    # 環境変数
    required_env_vars: List[str]
    optional_env_vars: List[str]


class ProductionDeploymentManager:
    """本番デプロイメント管理"""

    def __init__(self):
        self.config_templates = self._initialize_config_templates()

        print("=" * 80)
        print("[DEPLOYMENT] Day Trade 本番環境デプロイメント設定")
        print("Phase G: 本番運用最適化フェーズ")
        print("=" * 80)

    def _initialize_config_templates(self) -> Dict[str, Dict[str, Any]]:
        """設定テンプレート初期化"""
        return {
            "database": {
                "development": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "daytrade_dev",
                    "username": "dev_user",
                    "connection_pool_size": 5,
                },
                "staging": {
                    "host": "staging-db.internal",
                    "port": 5432,
                    "database": "daytrade_staging",
                    "username": "staging_user",
                    "connection_pool_size": 10,
                },
                "production": {
                    "host": "prod-db-cluster.internal",
                    "port": 5432,
                    "database": "daytrade_prod",
                    "username": "prod_user",
                    "connection_pool_size": 50,
                },
            },
            "api": {
                "development": {
                    "host": "127.0.0.1",
                    "port": 8000,
                    "workers": 1,
                    "threads_per_worker": 4,
                    "rate_limit_per_minute": 1000,
                },
                "staging": {
                    "host": "0.0.0.0",
                    "port": 8080,
                    "workers": 2,
                    "threads_per_worker": 8,
                    "rate_limit_per_minute": 500,
                },
                "production": {
                    "host": "0.0.0.0",
                    "port": 443,
                    "workers": 8,
                    "threads_per_worker": 16,
                    "rate_limit_per_minute": 10000,
                },
            },
        }

    def create_deployment_config(
        self,
        environment: DeploymentEnvironment,
        service_tier: ServiceTier = ServiceTier.STANDARD,
    ) -> DeploymentConfig:
        """デプロイメント設定作成"""

        print(f"[CONFIG] {environment.value} 環境設定作成中...")
        print(f"サービスレベル: {service_tier.value}")

        # データベース設定
        db_template = self.config_templates["database"][environment.value]
        database_config = DatabaseConfig(
            host=db_template["host"],
            port=db_template["port"],
            database=db_template["database"],
            username=db_template["username"],
            password_env_var=f"DB_PASSWORD_{environment.value.upper()}",
            ssl_enabled=environment != DeploymentEnvironment.DEVELOPMENT,
            connection_pool_size=db_template["connection_pool_size"],
        )

        # キャッシュ設定
        cache_config = CacheConfig(
            backend="redis"
            if environment == DeploymentEnvironment.PRODUCTION
            else "memory",
            host="redis-cluster.internal"
            if environment == DeploymentEnvironment.PRODUCTION
            else "localhost",
            port=6379,
            password_env_var=f"REDIS_PASSWORD_{environment.value.upper()}"
            if environment == DeploymentEnvironment.PRODUCTION
            else None,
            ttl_seconds=3600
            if environment == DeploymentEnvironment.PRODUCTION
            else 1800,
            max_memory_mb=2048
            if environment == DeploymentEnvironment.PRODUCTION
            else 512,
        )

        # API設定
        api_template = self.config_templates["api"][environment.value]
        api_config = ApiConfig(
            host=api_template["host"],
            port=api_template["port"],
            workers=api_template["workers"],
            threads_per_worker=api_template["threads_per_worker"],
            max_requests_per_worker=1000
            if environment == DeploymentEnvironment.PRODUCTION
            else 100,
            timeout_seconds=30,
            rate_limit_per_minute=api_template["rate_limit_per_minute"],
            cors_origins=self._get_cors_origins(environment),
            ssl_cert_path="/etc/ssl/certs/daytrade.crt"
            if environment == DeploymentEnvironment.PRODUCTION
            else None,
            ssl_key_path="/etc/ssl/private/daytrade.key"
            if environment == DeploymentEnvironment.PRODUCTION
            else None,
        )

        # 監視設定
        monitoring_config = MonitoringConfig(
            enabled=True,
            metrics_endpoint="/metrics",
            log_level="INFO"
            if environment == DeploymentEnvironment.PRODUCTION
            else "DEBUG",
            log_format="json"
            if environment == DeploymentEnvironment.PRODUCTION
            else "text",
            prometheus_port=9090,
            jaeger_endpoint="http://jaeger:14268/api/traces"
            if environment == DeploymentEnvironment.PRODUCTION
            else None,
            error_reporting_dsn=f"SENTRY_DSN_{environment.value.upper()}"
            if environment != DeploymentEnvironment.DEVELOPMENT
            else None,
        )

        # セキュリティ設定
        security_config = SecurityConfig(
            jwt_secret_env_var=f"JWT_SECRET_{environment.value.upper()}",
            api_key_env_var=f"API_KEY_{environment.value.upper()}",
            encryption_key_env_var=f"ENCRYPTION_KEY_{environment.value.upper()}",
            allowed_hosts=self._get_allowed_hosts(environment),
            csrf_protection=environment == DeploymentEnvironment.PRODUCTION,
            session_timeout_minutes=60
            if environment == DeploymentEnvironment.PRODUCTION
            else 120,
            password_policy=self._get_password_policy(environment),
        )

        # パフォーマンス設定
        performance_config = PerformanceConfig(
            optimization_level="gpu_accelerated"
            if environment == DeploymentEnvironment.PRODUCTION
            else "optimized",
            gpu_enabled=environment == DeploymentEnvironment.PRODUCTION,
            pytorch_enabled=True,
            batch_size=64 if environment == DeploymentEnvironment.PRODUCTION else 32,
            cache_enabled=True,
            async_processing=environment == DeploymentEnvironment.PRODUCTION,
            connection_pooling=True,
            query_caching=environment != DeploymentEnvironment.DEVELOPMENT,
        )

        # 統合設定
        config = DeploymentConfig(
            environment=environment,
            service_tier=service_tier,
            version=self._get_version(),
            deployment_id=self._generate_deployment_id(),
            timestamp=datetime.now(),
            database=database_config,
            cache=cache_config,
            api=api_config,
            monitoring=monitoring_config,
            security=security_config,
            performance=performance_config,
            required_env_vars=self._get_required_env_vars(environment),
            optional_env_vars=self._get_optional_env_vars(environment),
        )

        return config

    def _get_cors_origins(self, environment: DeploymentEnvironment) -> List[str]:
        """CORS オリジン設定"""
        cors_map = {
            DeploymentEnvironment.DEVELOPMENT: [
                "http://localhost:3000",
                "http://localhost:8080",
            ],
            DeploymentEnvironment.STAGING: ["https://staging.daytrade.com"],
            DeploymentEnvironment.PRODUCTION: [
                "https://daytrade.com",
                "https://www.daytrade.com",
            ],
            DeploymentEnvironment.TEST: ["http://test.local"],
        }
        return cors_map.get(environment, ["*"])

    def _get_allowed_hosts(self, environment: DeploymentEnvironment) -> List[str]:
        """許可ホスト設定"""
        hosts_map = {
            DeploymentEnvironment.DEVELOPMENT: ["localhost", "127.0.0.1"],
            DeploymentEnvironment.STAGING: [
                "staging.daytrade.com",
                "staging-api.daytrade.com",
            ],
            DeploymentEnvironment.PRODUCTION: [
                "daytrade.com",
                "api.daytrade.com",
                "www.daytrade.com",
            ],
            DeploymentEnvironment.TEST: ["test.local", "test-api.local"],
        }
        return hosts_map.get(environment, ["localhost"])

    def _get_password_policy(
        self, environment: DeploymentEnvironment
    ) -> Dict[str, Any]:
        """パスワードポリシー"""
        if environment == DeploymentEnvironment.PRODUCTION:
            return {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_digits": True,
                "require_special_chars": True,
                "max_age_days": 90,
            }
        else:
            return {
                "min_length": 8,
                "require_uppercase": False,
                "require_lowercase": True,
                "require_digits": True,
                "require_special_chars": False,
                "max_age_days": 365,
            }

    def _get_required_env_vars(self, environment: DeploymentEnvironment) -> List[str]:
        """必須環境変数"""
        base_vars = [
            f"DB_PASSWORD_{environment.value.upper()}",
            f"JWT_SECRET_{environment.value.upper()}",
            f"API_KEY_{environment.value.upper()}",
            f"ENCRYPTION_KEY_{environment.value.upper()}",
        ]

        if environment == DeploymentEnvironment.PRODUCTION:
            base_vars.extend(
                [
                    f"REDIS_PASSWORD_{environment.value.upper()}",
                    f"SENTRY_DSN_{environment.value.upper()}",
                    "SSL_CERT_PATH",
                    "SSL_KEY_PATH",
                ]
            )

        return base_vars

    def _get_optional_env_vars(self, environment: DeploymentEnvironment) -> List[str]:
        """オプション環境変数"""
        return [
            "ALPHA_VANTAGE_API_KEY",
            "QUANDL_API_KEY",
            "YAHOO_FINANCE_API_KEY",
            "SMTP_SERVER",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SLACK_WEBHOOK_URL",
            "DISCORD_WEBHOOK_URL",
        ]

    def _get_version(self) -> str:
        """バージョン取得"""
        try:
            # Git からバージョン情報を取得
            import subprocess

            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "1.0.0"

    def _generate_deployment_id(self) -> str:
        """デプロイメントID生成"""
        import uuid

        return str(uuid.uuid4())[:8]

    def save_config_files(
        self, config: DeploymentConfig, output_dir: str = "deployment"
    ):
        """設定ファイル保存"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        env_name = config.environment.value

        print(f"[SAVE] {env_name} 環境設定ファイル作成中...")

        # JSON設定ファイル
        json_config = asdict(config)
        json_config["timestamp"] = config.timestamp.isoformat()
        json_config["environment"] = config.environment.value
        json_config["service_tier"] = config.service_tier.value

        json_file = output_path / f"config_{env_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)

        # YAML設定ファイル
        yaml_file = output_path / f"config_{env_name}.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(json_config, f, default_flow_style=False, allow_unicode=True)

        # 環境変数テンプレート
        env_file = output_path / f".env.{env_name}"
        self._create_env_template(config, env_file)

        # Docker Compose
        docker_file = output_path / f"docker-compose.{env_name}.yml"
        self._create_docker_compose(config, docker_file)

        # Kubernetes マニフェスト
        if config.environment == DeploymentEnvironment.PRODUCTION:
            k8s_dir = output_path / "k8s"
            k8s_dir.mkdir(exist_ok=True)
            self._create_kubernetes_manifests(config, k8s_dir)

        # Nginx設定
        nginx_file = output_path / f"nginx_{env_name}.conf"
        self._create_nginx_config(config, nginx_file)

        print("[OK] 設定ファイル保存完了:")
        print(f"  - {json_file}")
        print(f"  - {yaml_file}")
        print(f"  - {env_file}")
        print(f"  - {docker_file}")
        print(f"  - {nginx_file}")

        if config.environment == DeploymentEnvironment.PRODUCTION:
            print(f"  - {k8s_dir}/ (Kubernetes manifests)")

    def _create_env_template(self, config: DeploymentConfig, output_file: Path):
        """環境変数テンプレート作成"""
        env_content = f"""# Day Trade {config.environment.value.upper()} Environment Variables
# Generated on {config.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

# Database Configuration
DB_HOST={config.database.host}
DB_PORT={config.database.port}
DB_NAME={config.database.database}
DB_USER={config.database.username}
{config.database.password_env_var}=YOUR_DB_PASSWORD_HERE

# Cache Configuration
CACHE_BACKEND={config.cache.backend}
CACHE_HOST={config.cache.host}
CACHE_PORT={config.cache.port}
"""

        if config.cache.password_env_var:
            env_content += f"{config.cache.password_env_var}=YOUR_CACHE_PASSWORD_HERE\n"

        env_content += f"""
# API Configuration
API_HOST={config.api.host}
API_PORT={config.api.port}
API_WORKERS={config.api.workers}

# Security Configuration
{config.security.jwt_secret_env_var}=YOUR_JWT_SECRET_HERE
{config.security.api_key_env_var}=YOUR_API_KEY_HERE
{config.security.encryption_key_env_var}=YOUR_ENCRYPTION_KEY_HERE

# Performance Configuration
OPTIMIZATION_LEVEL={config.performance.optimization_level}
GPU_ENABLED={config.performance.gpu_enabled}
PYTORCH_ENABLED={config.performance.pytorch_enabled}

# Monitoring Configuration
LOG_LEVEL={config.monitoring.log_level}
PROMETHEUS_PORT={config.monitoring.prometheus_port}

# Optional External API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
QUANDL_API_KEY=your_quandl_key_here
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(env_content)

    def _create_docker_compose(self, config: DeploymentConfig, output_file: Path):
        """Docker Compose設定作成"""
        compose_content = f"""version: '3.8'

services:
  daytrade-app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{config.api.port}:{config.api.port}"
    environment:
      - ENVIRONMENT={config.environment.value}
      - API_HOST={config.api.host}
      - API_PORT={config.api.port}
      - DB_HOST={config.database.host}
      - DB_PORT={config.database.port}
      - CACHE_HOST={config.cache.host}
      - CACHE_PORT={config.cache.port}
    env_file:
      - .env.{config.environment.value}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    depends_on:
      - database
      - cache

  database:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB={config.database.database}
      - POSTGRES_USER={config.database.username}
      - POSTGRES_PASSWORD=${{DB_PASSWORD_{config.environment.value.upper()}}}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql:/docker-entrypoint-initdb.d
    restart: unless-stopped
    ports:
      - "{config.database.port}:{config.database.port}"

  cache:
    image: {"redis:7-alpine" if config.cache.backend == "redis" else "memcached:alpine"}
    {"command: redis-server --requirepass ${REDIS_PASSWORD_" + config.environment.value.upper() + "}" if config.cache.backend == "redis" and config.cache.password_env_var else ""}
    ports:
      - "{config.cache.port}:{config.cache.port}"
    restart: unless-stopped

"""

        if config.environment == DeploymentEnvironment.PRODUCTION:
            compose_content += """  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - daytrade-app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  grafana_data:
"""
        else:
            compose_content += """
volumes:
  postgres_data:
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(compose_content)

    def _create_nginx_config(self, config: DeploymentConfig, output_file: Path):
        """Nginx設定作成"""
        nginx_content = f"""events {{
    worker_connections 1024;
}}

http {{
    upstream daytrade_backend {{
        server daytrade-app:{config.api.port};
    }}

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate={config.api.rate_limit_per_minute}r/m;

    server {{
        listen 80;
        server_name {' '.join(config.security.allowed_hosts)};

        # Redirect HTTP to HTTPS in production
        {'return 301 https://$server_name$request_uri;' if config.environment == DeploymentEnvironment.PRODUCTION else ''}

        location / {{
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://daytrade_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout {config.api.timeout_seconds}s;
        }}

        location /health {{
            proxy_pass http://daytrade_backend/health;
            access_log off;
        }}

        location /metrics {{
            proxy_pass http://daytrade_backend/metrics;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            deny all;
        }}
    }}
"""

        if config.environment == DeploymentEnvironment.PRODUCTION:
            nginx_content += f"""
    server {{
        listen 443 ssl http2;
        server_name {' '.join(config.security.allowed_hosts)};

        ssl_certificate {config.api.ssl_cert_path};
        ssl_certificate_key {config.api.ssl_key_path};
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS;
        ssl_prefer_server_ciphers off;

        location / {{
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://daytrade_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout {config.api.timeout_seconds}s;
        }}
    }}
"""

        nginx_content += "\n}\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(nginx_content)

    def _create_kubernetes_manifests(self, config: DeploymentConfig, output_dir: Path):
        """Kubernetes マニフェスト作成"""
        # Deployment
        deployment_manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: daytrade-app
  labels:
    app: daytrade
spec:
  replicas: {config.api.workers}
  selector:
    matchLabels:
      app: daytrade
  template:
    metadata:
      labels:
        app: daytrade
    spec:
      containers:
      - name: daytrade
        image: daytrade:latest
        ports:
        - containerPort: {config.api.port}
        env:
        - name: ENVIRONMENT
          value: "{config.environment.value}"
        - name: API_PORT
          value: "{config.api.port}"
        envFrom:
        - secretRef:
            name: daytrade-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: {config.api.port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: {config.api.port}
          initialDelaySeconds: 5
          periodSeconds: 5
"""

        # Service
        service_manifest = f"""apiVersion: v1
kind: Service
metadata:
  name: daytrade-service
spec:
  selector:
    app: daytrade
  ports:
  - port: 80
    targetPort: {config.api.port}
  type: LoadBalancer
"""

        # Secrets template
        secrets_manifest = f"""apiVersion: v1
kind: Secret
metadata:
  name: daytrade-secrets
type: Opaque
data:
  # Base64 encoded values - replace with actual values
  DB_PASSWORD_{config.environment.value.upper()}: <BASE64_ENCODED_DB_PASSWORD>
  JWT_SECRET_{config.environment.value.upper()}: <BASE64_ENCODED_JWT_SECRET>
  API_KEY_{config.environment.value.upper()}: <BASE64_ENCODED_API_KEY>
  ENCRYPTION_KEY_{config.environment.value.upper()}: <BASE64_ENCODED_ENCRYPTION_KEY>
"""

        # Save manifests
        with open(output_dir / "deployment.yaml", "w") as f:
            f.write(deployment_manifest)

        with open(output_dir / "service.yaml", "w") as f:
            f.write(service_manifest)

        with open(output_dir / "secrets.yaml", "w") as f:
            f.write(secrets_manifest)

    def validate_config(self, config: DeploymentConfig) -> Dict[str, List[str]]:
        """設定検証"""
        issues = {"errors": [], "warnings": [], "info": []}

        # 必須環境変数チェック
        for env_var in config.required_env_vars:
            if not os.getenv(env_var):
                issues["errors"].append(f"必須環境変数が未設定: {env_var}")

        # セキュリティチェック
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.api.ssl_cert_path:
                issues["errors"].append("本番環境ではSSL証明書が必要です")

            if not config.security.csrf_protection:
                issues["warnings"].append("CSRF保護が無効になっています")

            if config.security.session_timeout_minutes > 120:
                issues["warnings"].append("セッションタイムアウトが長すぎます")

        # パフォーマンス設定チェック
        if not config.performance.cache_enabled:
            issues["warnings"].append("キャッシュが無効になっています")

        if config.api.workers > 16:
            issues["warnings"].append("ワーカー数が多すぎる可能性があります")

        # データベース設定チェック
        if config.database.connection_pool_size < 5:
            issues["warnings"].append("データベース接続プールサイズが小さすぎます")

        issues["info"].append(f"設定環境: {config.environment.value}")
        issues["info"].append(f"サービスレベル: {config.service_tier.value}")
        issues["info"].append(f"API ワーカー数: {config.api.workers}")

        return issues


def main():
    """メイン実行"""
    manager = ProductionDeploymentManager()

    print("\n[CREATE] 全環境のデプロイメント設定作成中...")

    environments = [
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentEnvironment.STAGING,
        DeploymentEnvironment.PRODUCTION,
    ]

    for env in environments:
        try:
            service_tier = (
                ServiceTier.ENTERPRISE
                if env == DeploymentEnvironment.PRODUCTION
                else ServiceTier.STANDARD
            )

            config = manager.create_deployment_config(env, service_tier)
            manager.save_config_files(config)

            # 設定検証
            issues = manager.validate_config(config)

            if issues["errors"]:
                print(f"\n[ERROR] {env.value} 環境の設定エラー:")
                for error in issues["errors"]:
                    print(f"  - {error}")

            if issues["warnings"]:
                print(f"\n[WARNING] {env.value} 環境の設定警告:")
                for warning in issues["warnings"]:
                    print(f"  - {warning}")

            print(f"\n[OK] {env.value} 環境設定完了")

        except Exception as e:
            print(f"\n[ERROR] {env.value} 環境設定エラー: {e}")

    print("\n" + "=" * 80)
    print("[COMPLETE] 本番環境デプロイメント設定完了")
    print("=" * 80)
    print("生成されたファイル:")
    print("  - deployment/ ディレクトリ内のすべての設定ファイル")
    print("  - 各環境用の Docker Compose ファイル")
    print("  - Nginx 設定ファイル")
    print("  - Kubernetes マニフェスト (本番環境)")
    print("  - 環境変数テンプレート")


if __name__ == "__main__":
    main()
