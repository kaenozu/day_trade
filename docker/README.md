# Issue #800 Phase 1: Docker環境構築ガイド

## 📊 Docker環境概要

Issue #487完全自動化システムの本番運用基盤として、マルチサービスDocker環境を構築しました。

### 🏗️ アーキテクチャ構成

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Environment                      │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ ML Service  │ │Data Service │ │ Scheduler Service       │ │
│ │ :8000       │ │ :8001       │ │ :8002                   │ │
│ │ EnsembleS.. │ │ DataFetcher │ │ ExecutionScheduler      │ │
│ │ 93%精度     │ │ SymbolSel.. │ │ 自動化ワークフロー      │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│ │ PostgreSQL  │ │   Redis     │ │ Prometheus + Grafana    │ │
│ │ :5432       │ │ :6379       │ │ :9090 + :3000           │ │
│ │ メインDB    │ │ キャッシュ  │ │ 監視・ダッシュボード    │ │
│ └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 クイックスタート

### 1. 開発環境起動

```bash
# Docker環境起動
docker compose -f docker/docker-compose.yml up -d

# ログ確認
docker compose -f docker/docker-compose.yml logs -f
```

### 2. サービス確認

```bash
# ヘルスチェック
curl http://localhost:8000/health  # ML Service
curl http://localhost:8001/health  # Data Service
curl http://localhost:8002/health  # Scheduler Service

# 監視ダッシュボード
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

### 3. 本番環境デプロイ

```bash
# 環境変数設定
cp .env.example .env
# .envファイルを編集して本番設定を記入

# 本番環境起動
docker compose -f docker/docker-compose.prod.yml up -d
```

## 📋 サービス詳細

### MLサービス (Port: 8000)
- **コンポーネント**: EnsembleSystem (93%精度)
- **機能**: XGBoost + CatBoost + RandomForest
- **API**: `/api/v1/predict`, `/health`, `/metrics`
- **リソース**: CPU 2.0, Memory 2G (本番)

### データサービス (Port: 8001)
- **コンポーネント**: DataFetcher + SmartSymbolSelector
- **機能**: リアルタイムデータ取得、銘柄選択
- **API**: `/api/v1/data`, `/api/v1/symbols`, `/health`
- **リソース**: CPU 1.0, Memory 1G (本番)

### スケジューラサービス (Port: 8002)
- **コンポーネント**: ExecutionScheduler
- **機能**: 自動化ワークフロー、タスク管理
- **API**: `/api/v1/scheduler`, `/api/v1/tasks`, `/health`
- **リソース**: CPU 0.5, Memory 512M (本番)

## 🔧 設定・カスタマイズ

### 環境変数設定

```bash
# .env ファイル例
ENVIRONMENT=production
POSTGRES_DB=day_trade
POSTGRES_USER=day_trade_user
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password
MARKET_DATA_API_KEY=your_api_key
GRAFANA_ADMIN_PASSWORD=your_grafana_password
```

### ボリュームマウント

- **models/**: MLモデルファイル
- **data/**: 市場データキャッシュ
- **logs/**: アプリケーションログ
- **schedules/**: スケジューラ設定

## 📊 監視・ログ

### Prometheus メトリクス
- システムリソース使用率
- アプリケーション性能指標
- 予測精度・レスポンス時間

### Grafanaダッシュボード
- リアルタイム監視
- パフォーマンス分析
- アラート設定

### ログ管理
```bash
# サービス別ログ確認
docker compose logs ml-service
docker compose logs data-service
docker compose logs scheduler-service

# リアルタイムログ
docker compose logs -f --tail=100
```

## 🔐 セキュリティ

### 本番環境セキュリティ
- 非rootユーザー実行
- ファイアウォール設定
- シークレット管理
- SSL/TLS暗号化

### ヘルスチェック
- 30秒間隔での自動チェック
- 異常時の自動再起動
- 依存サービス確認

## 🛠️ トラブルシューティング

### よくある問題

1. **ポート競合**
   ```bash
   # ポート使用状況確認
   netstat -tulpn | grep :8000
   ```

2. **メモリ不足**
   ```bash
   # リソース使用量確認
   docker stats
   ```

3. **依存関係エラー**
   ```bash
   # サービス依存順序確認
   docker compose up --remove-orphans
   ```

### ログ分析
```bash
# エラーログ抽出
docker compose logs | grep ERROR

# パフォーマンス分析
docker compose logs ml-service | grep "prediction_time"
```

## 📈 パフォーマンステスト

### 負荷テスト
```bash
# 予測APIテスト
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,2,3,4,5]}'

# 並行リクエストテスト
for i in {1..100}; do
  curl http://localhost:8000/health &
done
wait
```

### メモリ・CPU監視
```bash
# リソース使用量リアルタイム監視
watch docker stats
```

## 🔄 CI/CD統合準備

### GitHub Actions設定
- Docker イメージ自動ビルド
- テスト自動実行
- 本番環境自動デプロイ

### 品質ゲート
- テストカバレッジ > 90%
- セキュリティスキャン
- パフォーマンスベンチマーク

---

## 📝 次のステップ

### Issue #800 Phase 2: CI/CDパイプライン構築
- GitHub Actions設定
- 自動テスト統合
- デプロイ自動化

### Issue #800 Phase 3: クラウド環境デプロイ
- AWS/Azure環境構築
- Infrastructure as Code
- 本番運用体制

**🎯 Docker環境構築完了 - 本番運用基盤準備完了**