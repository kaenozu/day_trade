# Issue #800 Phase 2: CI/CDデプロイメントガイド

## 🚀 CI/CDパイプライン概要

Issue #487 EnsembleSystem (93%精度) の本番運用に向けた完全自動化CI/CDパイプラインです。

### 🔄 ワークフロー構成

```
┌─────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline Flow                      │
├─────────────────────────────────────────────────────────────┤
│  📝 Code Push  →  🧪 Test  →  🏗️ Build  →  🚀 Deploy      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🧪 test.yml        🏗️ build.yml       🚀 deploy-*.yml    │
│  ├─ Code Quality    ├─ Docker Build    ├─ Development      │
│  ├─ Unit Tests      ├─ Multi-arch      ├─ Staging          │
│  ├─ Integration     ├─ Security Scan   └─ Production       │
│  ├─ Performance     └─ Registry Push                       │
│  └─ 93% Accuracy                                           │
└─────────────────────────────────────────────────────────────┘
```

## 📋 GitHub Secrets 設定

### 必須シークレット

以下のシークレットをGitHubリポジトリ設定で追加してください：

#### 🔐 データベース・キャッシュ
```
STAGING_DB_PASSWORD=staging_secure_password
STAGING_REDIS_PASSWORD=staging_redis_password
PROD_POSTGRES_DB=day_trade_production
PROD_POSTGRES_USER=day_trade_prod_user
PROD_POSTGRES_PASSWORD=production_secure_password
PROD_REDIS_PASSWORD=production_redis_password
```

#### 🌐 API・外部サービス
```
STAGING_API_KEY=staging_market_data_api_key
PROD_API_KEY=production_market_data_api_key
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
```

#### 📊 監視・ダッシュボード
```
STAGING_GRAFANA_PASSWORD=staging_grafana_password
PROD_GRAFANA_PASSWORD=production_grafana_password
```

## 🚀 デプロイメント手順

### 1. 開発環境デプロイ (自動)

**トリガー**: `develop` ブランチへのプッシュ

```bash
# 自動実行されるワークフロー
- 🧪 重要テスト実行
- 🏗️ イメージビルド・プル
- 🚀 開発環境デプロイ
- ✅ ヘルスチェック・統合テスト
```

**確認URL**:
- ML Service: `http://dev-ml.company.com:8000/health`
- Data Service: `http://dev-data.company.com:8001/health`
- Scheduler: `http://dev-scheduler.company.com:8002/health`
- Grafana: `http://dev-monitoring.company.com:3000`

### 2. ステージング環境デプロイ (自動)

**トリガー**: `main` ブランチへのプッシュ

```bash
# 自動実行されるワークフロー
- 🎯 93%精度検証
- 🧪 包括的テストスイート
- 📊 品質ゲート評価
- 🎭 ステージング環境デプロイ
- 🏭 本番準備テスト（負荷・セキュリティ・パフォーマンス）
- 🔐 本番デプロイ承認待機
```

**確認URL**:
- ML Service: `http://staging-ml.company.com:8000/health`
- Data Service: `http://staging-data.company.com:8001/health`
- Scheduler: `http://staging-scheduler.company.com:8002/health`
- Monitoring: `http://staging-monitoring.company.com:3000`

### 3. 本番環境デプロイ (手動承認)

**トリガー**: Manual workflow dispatch

```bash
# GitHub Actions UI から実行
# Settings → Actions → deploy-prod.yml → Run workflow

# 入力パラメータ:
# - deploy_version: v1.0.0 (必須)
# - deployment_strategy: rolling/blue-green/canary
# - maintenance_window: true/false
```

**実行手順**:
1. GitHub Actions ページでワークフロー選択
2. パラメータ入力
3. 承認者による手動承認
4. 自動バックアップ実行
5. ローリングデプロイ実行
6. 本番動作検証

**本番URL**:
- ML Service: `https://ml.day-trade.company.com`
- Data Service: `https://data.day-trade.company.com`
- Scheduler: `https://scheduler.day-trade.company.com`
- Monitoring: `https://monitoring.day-trade.company.com`

## 🧪 テスト戦略

### 自動テスト階層

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Pyramid                           │
├─────────────────────────────────────────────────────────────┤
│                    🏭 Production Tests                      │
│                 ├─ End-to-End Validation                    │
│                 ├─ Load Testing (50+ concurrent)            │
│                 └─ Security & Performance                   │
├─────────────────────────────────────────────────────────────┤
│                   🎭 Integration Tests                      │
│               ├─ Service-to-Service API                     │
│               ├─ Database & Redis Integration               │
│               └─ 93% Accuracy Cross-Validation              │
├─────────────────────────────────────────────────────────────┤
│                     🧪 Unit Tests                          │
│           ├─ EnsembleSystem Components                      │
│           ├─ DataFetcher & SmartSymbolSelector              │
│           ├─ ExecutionScheduler Logic                       │
│           └─ Code Coverage >90% (ML), >85% (Automation)     │
└─────────────────────────────────────────────────────────────┘
```

### テスト実行コマンド

```bash
# ローカル開発環境でのテスト
pytest tests/ml/ -v --cov=src.day_trade.ml --cov-fail-under=90
pytest tests/automation/ -v --cov=src.day_trade.automation --cov-fail-under=85
pytest tests/performance/ -v --benchmark-only

# 93%精度検証
pytest tests/ml/test_ensemble_system_advanced.py::TestEnsembleSystemAdvanced::test_93_percent_accuracy_target -v
```

## 🔧 環境設定

### GitHub Environments

以下の環境を GitHub Settings → Environments で設定：

#### 🛠️ development
- **Protection Rules**: なし
- **Environment Secrets**: 開発用API キー
- **Reviewers**: 不要

#### 🎭 staging  
- **Protection Rules**: mainブランチのみ
- **Environment Secrets**: ステージング用設定
- **Reviewers**: 不要（自動実行）

#### 🔐 production-approval
- **Protection Rules**: 手動承認必須
- **Reviewers**: DevOps Team, Tech Lead
- **Wait Timer**: 5分

#### 🏭 production
- **Protection Rules**: 手動承認必須
- **Environment Secrets**: 本番用設定
- **Reviewers**: DevOps Team, Tech Lead, Product Owner

## 📊 監視・アラート

### CI/CDパイプライン監視

```
┌─────────────────────────────────────────────────────────────┐
│                  Pipeline Monitoring                        │
├─────────────────────────────────────────────────────────────┤
│  📈 Success Rate    │  ⏱️ Build Time     │  🔍 Test Coverage  │
│  ├─ >95% target     │  ├─ <10min target  │  ├─ >90% ML        │
│  └─ Weekly Report   │  └─ Trend Analysis │  └─ >85% Auto      │
├─────────────────────────────────────────────────────────────┤
│  🚨 Alert Conditions                                        │
│  ├─ Build Failure → Slack Notification                     │
│  ├─ Test Failure → Email + Slack                           │
│  ├─ Security Vulnerability → Immediate Alert               │
│  └─ Production Deploy → Success/Failure Notification       │
└─────────────────────────────────────────────────────────────┘
```

### パフォーマンス目標

| 指標 | 目標 | アラート閾値 |
|------|------|-------------|
| Build Time | <10分 | >15分 |
| Test Execution | <5分 | >8分 |
| Deploy Time | <5分 | >10分 |
| ML Service Response | <500ms | >1秒 |
| 93% Accuracy | >93% | <90% |

## 🔄 ロールバック手順

### 自動ロールバック

```bash
# ヘルスチェック失敗時の自動ロールバック
if curl -f http://localhost:8000/health --max-time 30; then
  echo "✅ Deployment Successful"
else
  echo "❌ Deployment Failed - Initiating Rollback"
  docker compose -f docker-compose.production.yml down
  docker compose -f docker-compose.production.backup.yml up -d
fi
```

### 手動ロールバック

```bash
# 1. 前バージョンのイメージタグ確認
docker image ls | grep day-trade

# 2. 前バージョンでの再デプロイ実行
gh workflow run deploy-prod.yml \
  -f deploy_version=v0.9.0 \
  -f deployment_strategy=rolling

# 3. データベース復旧（必要時）
psql day_trade_production < backup_YYYYMMDD_HHMMSS.sql
```

## 🎯 成功指標

### 技術指標
- ✅ デプロイ成功率: >95%
- ✅ 平均デプロイ時間: <10分
- ✅ テストカバレッジ: ML >90%, Automation >85%
- ✅ 93%精度維持: 継続達成

### 運用指標
- ✅ ダウンタイム: <1分/月
- ✅ 障害検出時間: <5分
- ✅ 復旧時間: <15分
- ✅ セキュリティ脆弱性: ゼロ

---

## 📞 サポート・問い合わせ

### CI/CD トラブルシューティング
- **GitHub Actions ログ**: Actions タブで詳細確認
- **Slack Channel**: `#ci-cd-alerts`
- **Documentation**: この DEPLOYMENT_GUIDE.md

### エスカレーション
1. **Level 1**: Development Team
2. **Level 2**: DevOps Team  
3. **Level 3**: Tech Lead + Product Owner

**🎯 Issue #800 Phase 2: CI/CDパイプライン構築完了**