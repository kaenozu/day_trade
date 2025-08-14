# Issue #800 Phase 3: クラウド環境デプロイ自動化

## 🌐 Infrastructure as Code 概要

Issue #487 EnsembleSystem (93%精度) の本番運用に向けたクラウド環境自動デプロイ基盤です。

### 🏗️ アーキテクチャ構成

```
┌─────────────────────────────────────────────────────────────┐
│                 Cloud Infrastructure                        │
├─────────────────────────────────────────────────────────────┤
│  🌐 AWS (推奨)          │  ☁️ Azure (代替)                  │
│  ├─ ECS Fargate         │  ├─ Container Instances           │
│  ├─ RDS PostgreSQL      │  ├─ Azure SQL Database            │
│  ├─ ElastiCache Redis   │  ├─ Azure Cache for Redis         │
│  ├─ S3 Storage          │  ├─ Storage Account               │
│  ├─ Application LB      │  ├─ Application Gateway           │
│  └─ CloudWatch          │  └─ Application Insights          │
├─────────────────────────────────────────────────────────────┤
│                   Container Orchestration                   │
│  🚢 Kubernetes                                              │
│  ├─ ML Service (EnsembleSystem 93%精度)                     │
│  ├─ Data Service (DataFetcher + SmartSymbolSelector)        │
│  ├─ Scheduler Service (ExecutionScheduler)                  │
│  └─ Auto Scaling + Load Balancing                          │
├─────────────────────────────────────────────────────────────┤
│                     Deployment Tools                        │
│  📦 Helm Charts + 🔧 Terraform                             │
│  ├─ Infrastructure Provisioning                            │
│  ├─ Configuration Management                               │
│  └─ Environment-specific Overlays                          │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 デプロイメント方法

### 1. AWS環境デプロイ (推奨)

#### 前提条件
```bash
# 必要ツール
- Terraform >= 1.0
- AWS CLI >= 2.0
- kubectl >= 1.25
- Helm >= 3.8

# AWS認証設定
aws configure
export AWS_REGION=ap-northeast-1
```

#### インフラストラクチャ構築
```bash
# Terraform初期化
cd infrastructure/terraform/aws
terraform init

# プランニング
terraform plan -var-file="production.tfvars"

# 実行
terraform apply -var-file="production.tfvars"
```

#### アプリケーションデプロイ
```bash
# EKSクラスター設定
aws eks update-kubeconfig --region ap-northeast-1 --name day-trade-production

# Helmチャートデプロイ
cd infrastructure/helm
helm upgrade --install day-trade-ml ./day-trade-ml \
  --namespace day-trade \
  --create-namespace \
  --values values.production.yaml
```

### 2. Azure環境デプロイ (代替)

#### 前提条件
```bash
# Azure CLI認証
az login
az account set --subscription "your-subscription-id"
```

#### インフラストラクチャ構築
```bash
# Terraform初期化
cd infrastructure/terraform/azure
terraform init

# 実行
terraform apply -var-file="production.tfvars"
```

#### AKSクラスターデプロイ
```bash
# AKSクラスター設定
az aks get-credentials --resource-group day-trade-production --name day-trade-aks

# Helmデプロイ
helm upgrade --install day-trade-ml ./infrastructure/helm/day-trade-ml \
  --namespace day-trade \
  --create-namespace \
  --values values.azure.yaml
```

### 3. Kubernetesマニフェスト直接デプロイ

```bash
# ネームスペース作成
kubectl apply -f infrastructure/kubernetes/base/namespace.yaml

# サービスデプロイ
kubectl apply -f infrastructure/kubernetes/base/
```

## 📊 環境別設定

### 🛠️ 開発環境 (Development)
- **リソース**: 最小構成
- **レプリカ数**: 1
- **ストレージ**: 一時的
- **監視**: 基本レベル

```bash
# 開発環境デプロイ
helm upgrade --install day-trade-ml ./day-trade-ml \
  --namespace day-trade-dev \
  --values values.development.yaml
```

### 🎭 ステージング環境 (Staging)
- **リソース**: 本番類似
- **レプリカ数**: 本番と同等
- **ストレージ**: 永続化
- **監視**: 詳細レベル

```bash
# ステージング環境デプロイ
helm upgrade --install day-trade-ml ./day-trade-ml \
  --namespace day-trade-staging \
  --values values.staging.yaml
```

### 🏭 本番環境 (Production)
- **リソース**: 高性能
- **レプリカ数**: 高可用性
- **ストレージ**: 冗長化
- **監視**: 全面監視

```bash
# 本番環境デプロイ
helm upgrade --install day-trade-ml ./day-trade-ml \
  --namespace day-trade \
  --values values.production.yaml
```

## 🔧 設定管理

### Terraform Variables

#### AWS設定 (`terraform/aws/terraform.tfvars`)
```hcl
# 基本設定
aws_region = "ap-northeast-1"
environment = "production"

# ネットワーク設定
vpc_cidr = "10.0.0.0/16"
public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
private_subnet_cidrs = ["10.0.10.0/24", "10.0.20.0/24"]

# ECS設定
ml_service_cpu = 2048
ml_service_memory = 4096
data_service_cpu = 1024
data_service_memory = 2048

# データベース設定
db_instance_class = "db.t3.medium"
db_allocated_storage = 100
db_password = "your-secure-password"

# Redis設定
redis_node_type = "cache.t3.micro"
redis_num_cache_nodes = 1
```

#### Azure設定 (`terraform/azure/terraform.tfvars`)
```hcl
# 基本設定
azure_region = "Japan East"
environment = "production"

# ネットワーク設定
vnet_cidr = "10.1.0.0/16"
container_subnet_cidr = "10.1.1.0/24"
database_subnet_cidr = "10.1.2.0/24"

# SQL Database設定
sql_database_sku = "S2"
sql_database_max_size_gb = 100
sql_admin_username = "sqladmin"
sql_admin_password = "your-secure-password"

# Redis設定
redis_capacity = 1
redis_family = "C"
redis_sku_name = "Standard"
```

### Helm Values

#### 本番環境設定 (`values.production.yaml`)
```yaml
environment: production

mlService:
  replicaCount: 3
  resources:
    requests:
      memory: "4Gi"
      cpu: "2000m"
    limits:
      memory: "8Gi"
      cpu: "4000m"

postgresql:
  enabled: true
  primary:
    persistence:
      size: 200Gi

redis:
  enabled: true
  master:
    persistence:
      size: 20Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

## 📊 監視・ロギング

### Prometheus メトリクス
- システムリソース使用率
- アプリケーション性能指標
- 93%精度達成率
- API レスポンス時間

### Grafana ダッシュボード
- リアルタイム監視
- パフォーマンス分析
- アラート設定
- SLA監視

### CloudWatch / Application Insights
- ログ集約・分析
- カスタムメトリクス
- 異常検知
- 自動スケーリング

## 🔐 セキュリティ

### ネットワークセキュリティ
- VPC/VNet分離
- セキュリティグループ/NSG
- プライベートサブネット
- SSL/TLS暗号化

### 認証・認可
- IAM ロール・ポリシー
- Service Account
- RBAC設定
- シークレット管理

### データ暗号化
- 保存時暗号化 (at rest)
- 転送時暗号化 (in transit)
- Key Vault / SSM Parameter Store
- 証明書管理

## 🔄 CI/CD統合

### GitHub Actions連携
```yaml
# .github/workflows/deploy-cloud.yml
name: 🌐 Cloud Deploy

on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
        - development
        - staging
        - production

jobs:
  deploy-infrastructure:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v2

    - name: Deploy Infrastructure
      run: |
        cd infrastructure/terraform/aws
        terraform init
        terraform apply -auto-approve \
          -var-file="${{ github.event.inputs.environment }}.tfvars"

  deploy-application:
    needs: deploy-infrastructure
    runs-on: ubuntu-latest
    steps:
    - name: Deploy with Helm
      run: |
        helm upgrade --install day-trade-ml ./infrastructure/helm/day-trade-ml \
          --namespace day-trade-${{ github.event.inputs.environment }} \
          --values values.${{ github.event.inputs.environment }}.yaml
```

## 🛠️ トラブルシューティング

### よくある問題

#### 1. Terraform実行エラー
```bash
# 状態ファイル確認
terraform state list
terraform state show <resource>

# 強制的なリソース削除
terraform destroy -target=<resource>
```

#### 2. Helm デプロイエラー
```bash
# デプロイ状況確認
helm status day-trade-ml -n day-trade

# ロールバック
helm rollback day-trade-ml 1 -n day-trade

# デバッグモード
helm upgrade --install day-trade-ml ./day-trade-ml --debug --dry-run
```

#### 3. Pod起動失敗
```bash
# Pod状況確認
kubectl get pods -n day-trade
kubectl describe pod <pod-name> -n day-trade
kubectl logs <pod-name> -n day-trade

# リソース確認
kubectl top nodes
kubectl top pods -n day-trade
```

### パフォーマンス最適化

#### リソース調整
```yaml
# CPU集約的ワークロード用
resources:
  requests:
    cpu: "2000m"
    memory: "4Gi"
  limits:
    cpu: "4000m"
    memory: "8Gi"

# メモリ集約的ワークロード用
resources:
  requests:
    cpu: "1000m"
    memory: "8Gi"
  limits:
    cpu: "2000m"
    memory: "16Gi"
```

#### 自動スケーリング設定
```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## 📈 運用・保守

### 定期メンテナンス
- 週次: セキュリティアップデート
- 月次: パフォーマンスレビュー  
- 四半期: 災害復旧テスト

### バックアップ・復旧
- 自動データベースバックアップ
- スナップショット管理
- クロスリージョンレプリケーション

### スケーリング戦略
- 水平スケーリング (Pod レプリカ)
- 垂直スケーリング (リソース増強)
- クラスタースケーリング (ノード追加)

---

## 📞 サポート・問い合わせ

### 運用サポート
- **Level 1**: DevOps Team
- **Level 2**: Cloud Architecture Team
- **Level 3**: SRE Team

### ドキュメント
- [AWS設定ガイド](terraform/aws/README.md)
- [Azure設定ガイド](terraform/azure/README.md)
- [Kubernetes運用ガイド](kubernetes/README.md)
- [Helm使用法](helm/README.md)

**🎯 Issue #800 Phase 3: クラウド環境デプロイ自動化完了**