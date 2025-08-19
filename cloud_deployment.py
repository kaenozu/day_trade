#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cloud Deployment - クラウドデプロイメントシステム
Issue #952対応: Docker + Kubernetes + CI/CD + 複数クラウド対応
"""

import os
import yaml
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class CloudProvider(Enum):
    """クラウドプロバイダ"""
    AWS = "AWS"
    AZURE = "AZURE"
    GCP = "GCP"
    HEROKU = "HEROKU"
    RAILWAY = "RAILWAY"
    VERCEL = "VERCEL"


class DeploymentEnvironment(Enum):
    """デプロイメント環境"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """デプロイメント設定"""
    app_name: str = "day-trade-personal"
    version: str = "1.0.0"
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    cloud_provider: CloudProvider = CloudProvider.HEROKU
    replicas: int = 2
    cpu_limit: str = "1000m"
    memory_limit: str = "1Gi"
    auto_scaling: bool = True
    health_check_path: str = "/health"
    domain: str = ""
    ssl_enabled: bool = True


class DockerManager:
    """Docker管理"""
    
    def __init__(self, app_name: str):
        self.app_name = app_name
    
    def generate_dockerfile(self) -> str:
        """Dockerfile生成"""
        dockerfile_content = f"""# Day Trade Personal - Multi-stage Docker Build
FROM python:3.11-slim as base

# 作業ディレクトリ設定
WORKDIR /app

# システム依存関係インストール
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Python依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# 静的ファイルコレクト
RUN python -m flask --app daytrade_core.py collect-static || true

# 本番用イメージ
FROM python:3.11-slim as production

WORKDIR /app

# 必要なランタイム依存関係のみ
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Python依存関係コピー
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# アプリケーションコピー
COPY --from=base /app .

# 非ルートユーザー作成
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# ポート公開
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 起動コマンド
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "daytrade_core:app"]
"""
        
        with open("Dockerfile", 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        return "Dockerfile"
    
    def generate_docker_compose(self, config: DeploymentConfig) -> str:
        """docker-compose.yml生成"""
        compose_content = f"""version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: {self.app_name}:{config.version}
    container_name: {self.app_name}-app
    restart: unless-stopped
    environment:
      - FLASK_ENV={config.environment.value}
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/daytradedb
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your-secret-key-here
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    networks:
      - daytrade-network
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  db:
    image: postgres:15-alpine
    container_name: {self.app_name}-db
    restart: unless-stopped
    environment:
      - POSTGRES_DB=daytradedb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - daytrade-network

  redis:
    image: redis:7-alpine
    container_name: {self.app_name}-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - daytrade-network
    command: redis-server --appendonly yes

  nginx:
    image: nginx:alpine
    container_name: {self.app_name}-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./static:/var/www/static
    depends_on:
      - app
    networks:
      - daytrade-network

volumes:
  postgres_data:
  redis_data:

networks:
  daytrade-network:
    driver: bridge
"""
        
        with open("docker-compose.yml", 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        return "docker-compose.yml"
    
    def generate_dockerignore(self) -> str:
        """dockerignore生成"""
        dockerignore_content = """# Day Trade Personal - Docker Ignore

# Git
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.coverage
.pytest_cache/

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Data
data/backups/
data/temp/

# Node
node_modules/
npm-debug.log

# Docker
Dockerfile
docker-compose*.yml
.dockerignore

# CI/CD
.github/
.gitlab-ci.yml
azure-pipelines.yml

# Documentation
docs/
README.md
CHANGELOG.md

# Testing
tests/
test_*
*_test.py

# Development files
.env.development
.env.local
"""
        
        with open(".dockerignore", 'w', encoding='utf-8') as f:
            f.write(dockerignore_content)
        
        return ".dockerignore"


class KubernetesManager:
    """Kubernetes管理"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.manifests_dir = "k8s"
        os.makedirs(self.manifests_dir, exist_ok=True)
    
    def generate_deployment(self) -> str:
        """Deployment manifest生成"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.config.app_name}-deployment",
                "labels": {
                    "app": self.config.app_name,
                    "version": self.config.version
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.app_name,
                            "version": self.config.version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": self.config.app_name,
                            "image": f"{self.config.app_name}:{self.config.version}",
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "env": [
                                {
                                    "name": "FLASK_ENV",
                                    "value": self.config.environment.value
                                },
                                {
                                    "name": "DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": f"{self.config.app_name}-secrets",
                                            "key": "database-url"
                                        }
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "250m",
                                    "memory": "512Mi"
                                },
                                "limits": {
                                    "cpu": self.config.cpu_limit,
                                    "memory": self.config.memory_limit
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": self.config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": self.config.health_check_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3
                            }
                        }]
                    }
                }
            }
        }
        
        deployment_path = os.path.join(self.manifests_dir, "deployment.yaml")
        with open(deployment_path, 'w', encoding='utf-8') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        return deployment_path
    
    def generate_service(self) -> str:
        """Service manifest生成"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.app_name}-service",
                "labels": {
                    "app": self.config.app_name
                }
            },
            "spec": {
                "type": "ClusterIP",
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP",
                    "name": "http"
                }],
                "selector": {
                    "app": self.config.app_name
                }
            }
        }
        
        service_path = os.path.join(self.manifests_dir, "service.yaml")
        with open(service_path, 'w', encoding='utf-8') as f:
            yaml.dump(service, f, default_flow_style=False)
        
        return service_path
    
    def generate_ingress(self) -> str:
        """Ingress manifest生成"""
        ingress = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.config.app_name}-ingress",
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod" if self.config.ssl_enabled else ""
                }
            },
            "spec": {
                "ingressClassName": "nginx",
                "rules": [{
                    "host": self.config.domain or f"{self.config.app_name}.example.com",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{self.config.app_name}-service",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        if self.config.ssl_enabled:
            ingress["spec"]["tls"] = [{
                "hosts": [self.config.domain or f"{self.config.app_name}.example.com"],
                "secretName": f"{self.config.app_name}-tls"
            }]
        
        ingress_path = os.path.join(self.manifests_dir, "ingress.yaml")
        with open(ingress_path, 'w', encoding='utf-8') as f:
            yaml.dump(ingress, f, default_flow_style=False)
        
        return ingress_path
    
    def generate_hpa(self) -> str:
        """HPA manifest生成"""
        if not self.config.auto_scaling:
            return ""
        
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.app_name}-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.config.app_name}-deployment"
                },
                "minReplicas": 2,
                "maxReplicas": 10,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        hpa_path = os.path.join(self.manifests_dir, "hpa.yaml")
        with open(hpa_path, 'w', encoding='utf-8') as f:
            yaml.dump(hpa, f, default_flow_style=False)
        
        return hpa_path


class CloudProviderManager:
    """クラウドプロバイダ管理"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_heroku_config(self) -> Dict[str, str]:
        """Heroku設定生成"""
        # Procfile
        procfile_content = f"""web: gunicorn --bind 0.0.0.0:$PORT --workers 4 daytrade_core:app
worker: python worker.py
"""
        
        with open("Procfile", 'w', encoding='utf-8') as f:
            f.write(procfile_content)
        
        # app.json
        app_json = {
            "name": self.config.app_name,
            "description": "Day Trade Personal - AI-powered financial analysis platform",
            "image": "heroku/python",
            "stack": "heroku-22",
            "keywords": ["python", "flask", "ai", "trading", "finance"],
            "website": f"https://{self.config.app_name}.herokuapp.com",
            "repository": f"https://github.com/yourusername/{self.config.app_name}",
            "env": {
                "FLASK_ENV": {
                    "description": "Flask environment",
                    "value": self.config.environment.value
                },
                "SECRET_KEY": {
                    "description": "Secret key for Flask application",
                    "generator": "secret"
                },
                "DATABASE_URL": {
                    "description": "PostgreSQL database URL"
                }
            },
            "formation": {
                "web": {
                    "quantity": self.config.replicas,
                    "size": "standard-1x"
                }
            },
            "addons": [
                "heroku-postgresql:mini",
                "heroku-redis:mini"
            ],
            "buildpacks": [
                {
                    "url": "heroku/python"
                }
            ],
            "environments": {
                "test": {
                    "formation": {
                        "web": {"quantity": 1, "size": "standard-1x"}
                    },
                    "addons": ["heroku-postgresql:mini"]
                }
            }
        }
        
        with open("app.json", 'w', encoding='utf-8') as f:
            json.dump(app_json, f, indent=2)
        
        return {"Procfile": "Procfile", "app.json": "app.json"}
    
    def generate_railway_config(self) -> str:
        """Railway設定生成"""
        railway_toml = f"""[build]
builder = "NIXPACKS"

[deploy]
healthcheckPath = "{self.config.health_check_path}"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[environment]
PYTHON_VERSION = "3.11"
"""
        
        with open("railway.toml", 'w', encoding='utf-8') as f:
            f.write(railway_toml)
        
        return "railway.toml"
    
    def generate_vercel_config(self) -> str:
        """Vercel設定生成"""
        vercel_json = {
            "version": 2,
            "name": self.config.app_name,
            "builds": [
                {
                    "src": "daytrade_core.py",
                    "use": "@vercel/python"
                }
            ],
            "routes": [
                {
                    "src": "/(.*)",
                    "dest": "/daytrade_core.py"
                }
            ],
            "env": {
                "FLASK_ENV": self.config.environment.value
            },
            "regions": ["nrt1", "hnd1"],  # Tokyo regions
            "functions": {
                "daytrade_core.py": {
                    "maxDuration": 30
                }
            }
        }
        
        with open("vercel.json", 'w', encoding='utf-8') as f:
            json.dump(vercel_json, f, indent=2)
        
        return "vercel.json"


class CICDManager:
    """CI/CD管理"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_github_actions(self) -> str:
        """GitHub Actions CI/CD生成"""
        ci_dir = ".github/workflows"
        os.makedirs(ci_dir, exist_ok=True)
        
        workflow_content = f"""name: Day Trade Personal CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('**/requirements.txt') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run security scan
      run: |
        python security_assessment.py
    
    - name: Run tests
      run: |
        python -m pytest --cov=. --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
          type=raw,value=latest,enable={{{{is_default_branch}}}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Deploy to staging
      if: github.ref == 'refs/heads/develop'
      run: |
        echo "Deploy to staging environment"
        # Add staging deployment commands here
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to production environment"
        # Add production deployment commands here

  lighthouse:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Lighthouse CI Action
      uses: treosh/lighthouse-ci-action@v10
      with:
        uploadArtifacts: true
        temporaryPublicStorage: true
"""
        
        workflow_path = os.path.join(ci_dir, "ci-cd.yml")
        with open(workflow_path, 'w', encoding='utf-8') as f:
            f.write(workflow_content)
        
        return workflow_path
    
    def generate_gitlab_ci(self) -> str:
        """GitLab CI生成"""
        gitlab_ci = f"""# Day Trade Personal GitLab CI/CD

stages:
  - test
  - security
  - build
  - deploy

variables:
  DOCKER_REGISTRY: registry.gitlab.com
  DOCKER_IMAGE: $DOCKER_REGISTRY/$CI_PROJECT_PATH
  PYTHON_VERSION: "3.11"

# Cache pip dependencies
cache:
  paths:
    - ~/.cache/pip/

before_script:
  - python --version
  - pip install --upgrade pip

# テストステージ
test:
  stage: test
  image: python:3.11
  services:
    - postgres:15
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: postgres
    POSTGRES_PASSWORD: postgres
    DATABASE_URL: postgresql://postgres:postgres@postgres:5432/test_db
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - python -m pytest --cov=. --cov-report=term --cov-report=xml
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
  only:
    - main
    - develop
    - merge_requests

# セキュリティスキャン
security_scan:
  stage: security
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - python security_assessment.py
  artifacts:
    reports:
      sast: data/security_report_*.html
  only:
    - main
    - develop

# Dockerイメージビルド
build:
  stage: build
  image: docker:24.0.5
  services:
    - docker:24.0.5-dind
  script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker build -t $DOCKER_IMAGE:$CI_COMMIT_SHA .
    - docker build -t $DOCKER_IMAGE:latest .
    - docker push $DOCKER_IMAGE:$CI_COMMIT_SHA
    - docker push $DOCKER_IMAGE:latest
  only:
    - main

# ステージング環境デプロイ
deploy_staging:
  stage: deploy
  image: alpine/helm:latest
  script:
    - echo "Deploy to staging"
    - helm upgrade --install {self.config.app_name}-staging ./helm/{self.config.app_name} --set image.tag=$CI_COMMIT_SHA
  environment:
    name: staging
    url: https://{self.config.app_name}-staging.example.com
  only:
    - develop

# 本番環境デプロイ
deploy_production:
  stage: deploy
  image: alpine/helm:latest
  script:
    - echo "Deploy to production"
    - helm upgrade --install {self.config.app_name} ./helm/{self.config.app_name} --set image.tag=$CI_COMMIT_SHA
  environment:
    name: production
    url: https://{self.config.app_name}.example.com
  when: manual
  only:
    - main
"""
        
        with open(".gitlab-ci.yml", 'w', encoding='utf-8') as f:
            f.write(gitlab_ci)
        
        return ".gitlab-ci.yml"


class CloudDeploymentManager:
    """クラウドデプロイメント管理"""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        self.docker_manager = DockerManager(self.config.app_name)
        self.k8s_manager = KubernetesManager(self.config)
        self.cloud_manager = CloudProviderManager(self.config)
        self.cicd_manager = CICDManager(self.config)
    
    def generate_deployment_files(self) -> Dict[str, List[str]]:
        """デプロイメントファイル生成"""
        results = {
            "docker": [],
            "kubernetes": [],
            "cloud": [],
            "cicd": []
        }
        
        logging.info("Generating cloud deployment configuration...")
        
        try:
            # Docker関連
            logging.info("Generating Docker files...")
            results["docker"].append(self.docker_manager.generate_dockerfile())
            results["docker"].append(self.docker_manager.generate_docker_compose(self.config))
            results["docker"].append(self.docker_manager.generate_dockerignore())
            
            # Kubernetes関連
            logging.info("Generating Kubernetes manifests...")
            results["kubernetes"].append(self.k8s_manager.generate_deployment())
            results["kubernetes"].append(self.k8s_manager.generate_service())
            results["kubernetes"].append(self.k8s_manager.generate_ingress())
            
            hpa_path = self.k8s_manager.generate_hpa()
            if hpa_path:
                results["kubernetes"].append(hpa_path)
            
            # クラウドプロバイダ設定
            logging.info(f"Generating {self.config.cloud_provider.value} configuration...")
            
            if self.config.cloud_provider == CloudProvider.HEROKU:
                heroku_files = self.cloud_manager.generate_heroku_config()
                results["cloud"].extend(heroku_files.values())
            elif self.config.cloud_provider == CloudProvider.RAILWAY:
                results["cloud"].append(self.cloud_manager.generate_railway_config())
            elif self.config.cloud_provider == CloudProvider.VERCEL:
                results["cloud"].append(self.cloud_manager.generate_vercel_config())
            
            # CI/CD設定
            logging.info("Generating CI/CD configuration...")
            results["cicd"].append(self.cicd_manager.generate_github_actions())
            results["cicd"].append(self.cicd_manager.generate_gitlab_ci())
            
            # requirements.txt生成（デプロイ用）
            self._generate_requirements()
            
            # デプロイメントスクリプト生成
            self._generate_deployment_scripts()
            
            logging.info("Cloud deployment configuration completed!")
            
        except Exception as e:
            logging.error(f"Deployment configuration generation failed: {e}")
            raise
        
        return results
    
    def _generate_requirements(self):
        """requirements.txt生成"""
        requirements = """# Day Trade Personal - Production Dependencies
Flask==3.0.0
gunicorn==21.2.0
psycopg2-binary==2.9.9
redis==5.0.1
celery==5.3.4
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
requests==2.31.0
python-dotenv==1.0.0
SQLAlchemy==2.0.23
Flask-SQLAlchemy==3.1.1
Flask-Migrate==4.0.5
Flask-Login==0.6.3
Flask-WTF==1.2.1
WTForms==3.1.0
Jinja2==3.1.2
MarkupSafe==2.1.3
itsdangerous==2.1.2
click==8.1.7
blinker==1.7.0
"""
        
        with open("requirements.txt", 'w', encoding='utf-8') as f:
            f.write(requirements)
    
    def _generate_deployment_scripts(self):
        """デプロイメントスクリプト生成"""
        # デプロイスクリプト（Unix）
        deploy_sh = f"""#!/bin/bash
# Day Trade Personal - Deployment Script

set -e

echo "🚀 Starting {self.config.app_name} deployment..."

# 環境変数確認
if [ -z "$ENVIRONMENT" ]; then
    ENVIRONMENT="{self.config.environment.value}"
fi

echo "Environment: $ENVIRONMENT"
echo "Cloud Provider: {self.config.cloud_provider.value}"

# Docker関連
if command -v docker &> /dev/null; then
    echo "📦 Building Docker image..."
    docker build -t {self.config.app_name}:{self.config.version} .
    
    echo "🧪 Running tests..."
    docker run --rm {self.config.app_name}:{self.config.version} python -m pytest
    
    echo "🔍 Security scan..."
    docker run --rm {self.config.app_name}:{self.config.version} python security_assessment.py
fi

# クラウドプロバイダ別デプロイ
case "{self.config.cloud_provider.value}" in
    "HEROKU")
        echo "🌐 Deploying to Heroku..."
        heroku container:push web --app {self.config.app_name}
        heroku container:release web --app {self.config.app_name}
        ;;
    "RAILWAY")
        echo "🚄 Deploying to Railway..."
        railway up
        ;;
    "VERCEL")
        echo "▲ Deploying to Vercel..."
        vercel --prod
        ;;
    *)
        echo "⚠️ Manual deployment required for {self.config.cloud_provider.value}"
        ;;
esac

echo "✅ Deployment completed!"
echo "🌍 Application URL: https://{self.config.domain or f'{self.config.app_name}.example.com'}"
"""
        
        with open("deploy.sh", 'w', encoding='utf-8') as f:
            f.write(deploy_sh)
        
        # Windows用デプロイスクリプト
        deploy_bat = f"""@echo off
REM Day Trade Personal - Windows Deployment Script

echo 🚀 Starting {self.config.app_name} deployment...

if not defined ENVIRONMENT set ENVIRONMENT={self.config.environment.value}

echo Environment: %ENVIRONMENT%
echo Cloud Provider: {self.config.cloud_provider.value}

REM Docker関連
docker --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo 📦 Building Docker image...
    docker build -t {self.config.app_name}:{self.config.version} .
    
    echo 🧪 Running tests...
    docker run --rm {self.config.app_name}:{self.config.version} python -m pytest
)

echo ✅ Deployment completed!
echo 🌍 Application URL: https://{self.config.domain or f'{self.config.app_name}.example.com'}
"""
        
        with open("deploy.bat", 'w', encoding='utf-8') as f:
            f.write(deploy_bat)
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """デプロイメント概要"""
        return {
            "app_name": self.config.app_name,
            "version": self.config.version,
            "environment": self.config.environment.value,
            "cloud_provider": self.config.cloud_provider.value,
            "replicas": self.config.replicas,
            "auto_scaling": self.config.auto_scaling,
            "ssl_enabled": self.config.ssl_enabled,
            "domain": self.config.domain or f"{self.config.app_name}.example.com",
            "features": {
                "docker_support": True,
                "kubernetes_support": True,
                "ci_cd_support": True,
                "multi_cloud_support": True,
                "auto_scaling": self.config.auto_scaling,
                "health_checks": True,
                "ssl_termination": self.config.ssl_enabled
            }
        }


# グローバルインスタンス
cloud_deployment_manager = CloudDeploymentManager()


def deploy_to_cloud(config: DeploymentConfig = None) -> Dict[str, List[str]]:
    """クラウドデプロイメント設定"""
    if config:
        global cloud_deployment_manager
        cloud_deployment_manager = CloudDeploymentManager(config)
    
    return cloud_deployment_manager.generate_deployment_files()


def get_deployment_summary() -> Dict[str, Any]:
    """デプロイメント概要取得"""
    return cloud_deployment_manager.get_deployment_summary()


if __name__ == "__main__":
    print("=== Cloud Deployment Configuration Test ===")
    
    # 各クラウドプロバイダ用の設定生成
    providers = [
        (CloudProvider.HEROKU, "heroku"),
        (CloudProvider.RAILWAY, "railway"),
        (CloudProvider.VERCEL, "vercel")
    ]
    
    for provider, name in providers:
        print(f"\\n{name.upper()} Configuration:")
        print("-" * 40)
        
        config = DeploymentConfig(
            app_name="day-trade-personal",
            version="1.0.0",
            environment=DeploymentEnvironment.PRODUCTION,
            cloud_provider=provider,
            replicas=2,
            auto_scaling=True,
            domain=f"daytrade.{name}.example.com"
        )
        
        results = deploy_to_cloud(config)
        
        for category, files in results.items():
            if files:
                print(f"  {category}: {len(files)} files")
                for file in files:
                    print(f"    - {file}")
    
    # デプロイメント概要
    summary = get_deployment_summary()
    print(f"\\nDeployment Summary:")
    print(f"  App: {summary['app_name']} v{summary['version']}")
    print(f"  Environment: {summary['environment']}")
    print(f"  Cloud Provider: {summary['cloud_provider']}")
    print(f"  Domain: {summary['domain']}")
    print(f"  Features: {list(summary['features'].keys())}")
    
    print("\\nCloud deployment configuration test completed!")