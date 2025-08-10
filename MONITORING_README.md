# リアルタイムメトリクス・アラートシステム

Day Trade システム用の包括的なリアルタイム監視・メトリクス収集・アラート生成システム

## 🎯 システム概要

### 主要機能
- **リアルタイムメトリクス収集**: システム・ビジネス指標を秒単位で収集
- **Prometheus統合**: 業界標準メトリクス形式でデータ保存・クエリ
- **Grafana可視化**: 美しいダッシュボードでリアルタイム監視
- **インテリジェントアラート**: 多段階アラートシステム
- **パフォーマンス分析**: システムボトルネック特定・最適化提案

### 技術スタック
- **Prometheus**: メトリクス収集・保存エンジン
- **Grafana**: 可視化ダッシュボード
- **AlertManager**: アラート管理・通知
- **FastAPI**: メトリクスエクスポーターAPI
- **Docker**: コンテナ化デプロイメント

## 🚀 クイックスタート

### 1. システムテスト実行
```bash
python test_monitoring_simple.py
```

### 2. 監視環境起動
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Web UI アクセス
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093
- **メトリクス**: http://localhost:8000/metrics

## 📊 メトリクス収集項目

### システムメトリクス
- CPU使用率
- メモリ使用量
- ディスク使用量
- ネットワーク統計
- データベース接続プール

### リスク管理メトリクス
- リスク分析実行回数・処理時間
- 現在のリスクスコア（銘柄別）
- 不正検知結果・精度
- リスクアラート発生回数
- リスクコンポーネント状態

### 取引メトリクス
- 取引実行回数・結果
- 取引執行時間
- ポートフォリオ価値
- 未実現・実現損益
- 勝率・最大ドローダウン

### AI エンジンメトリクス
- AI予測実行回数・処理時間
- 予測精度
- 生成AI API呼び出し統計
- GPU使用率
- モデル訓練状況

## 🔔 アラート設定

### アラートレベル
- **Critical**: 即座通知（リスクスコア >0.9）
- **High**: 5分以内通知（リスクスコア >0.7）
- **Medium**: 通常通知（リスクスコア >0.5）
- **Low**: 情報レベル

### 通知チャネル
- Email通知
- Slack統合（設定により）
- Webhook連携

## 🛠️ 設定とカスタマイゼーション

### 1. Prometheus設定
- ファイル: `monitoring/prometheus/prometheus.yml`
- スクレイプ間隔、アラートルール等を調整

### 2. Grafana ダッシュボード
- ディレクトリ: `monitoring/grafana/dashboard_configs/`
- JSON形式でダッシュボード設定

### 3. AlertManager設定
- ファイル: `monitoring/alertmanager/alertmanager.yml`
- 通知ルーティング、受信者設定

### 4. カスタムメトリクス追加
```python
from src.day_trade.monitoring.metrics import get_metrics_collector

collector = get_metrics_collector()
# カスタムメトリクス実装
```

## 🔧 開発者向け

### メトリクス収集デコレーター使用
```python
from src.day_trade.monitoring.metrics.decorators import measure_execution_time

@measure_execution_time(component="my_component")
async def my_function():
    # 自動的に実行時間測定・記録
    pass
```

### リスク分析メトリクス統合
```python
from src.day_trade.monitoring.metrics.decorators import measure_risk_analysis_performance

@measure_risk_analysis_performance()
async def my_risk_analysis():
    # リスク分析パフォーマンス自動測定
    return risk_result
```

## 📁 ディレクトリ構造

```
monitoring/
├── prometheus/                 # Prometheus設定
│   ├── prometheus.yml         # メイン設定
│   └── rules/                 # アラートルール
├── grafana/                   # Grafana設定
│   ├── datasources/          # データソース設定
│   ├── dashboards/           # ダッシュボードプロビジョニング
│   └── dashboard_configs/    # ダッシュボードJSON
├── alertmanager/             # AlertManager設定
│   └── alertmanager.yml      # 通知設定
└── Dockerfile.exporter       # メトリクスエクスポーター

src/day_trade/monitoring/
├── metrics/                  # メトリクス収集システム
│   ├── prometheus_metrics.py # Prometheusメトリクス定義
│   ├── decorators.py         # 測定デコレーター
│   └── metrics_exporter.py   # HTTPエクスポーター
```

## 🎯 パフォーマンス指標

### 目標値
- **メトリクス収集間隔**: 1-15秒
- **アラート応答時間**: 5秒以内
- **ダッシュボード更新**: リアルタイム
- **システム負荷**: CPU <5%

### 実績値（テスト環境）
- ✅ メトリクス収集: 100%成功
- ✅ テスト実行時間: 0.001秒
- ✅ サーバー起動: 即座
- ✅ エンドポイント: 全て正常

## 🔍 トラブルシューティング

### よくある問題

1. **メトリクス収集エラー**
   ```bash
   # 依存関係確認
   pip install prometheus-client fastapi uvicorn psutil
   ```

2. **ポート競合**
   ```bash
   # ポート使用状況確認
   netstat -an | findstr :8000
   ```

3. **Docker起動エラー**
   ```bash
   # ログ確認
   docker-compose -f docker-compose.monitoring.yml logs
   ```

## 📝 ライセンス

このプロジェクトは Day Trade システムの一部として MIT ライセンスで提供されます。

## 🤝 貢献

バグレポート、機能要望、プルリクエストをお待ちしています。

---

**Day Trade Team** - 2025年8月10日
