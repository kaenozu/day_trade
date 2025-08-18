# API耐障害性機能ガイド

## 概要

Day Tradeシステムの外部API通信の耐障害性を強化するための包括的な機能群です。ネットワーク障害やAPI障害時にもシステムが継続稼働できるよう、高度なリトライ機構、サーキットブレーカー、フェイルオーバー機能を提供します。

## 主な機能

### 1. 高度なリトライ機構

#### 指数バックオフ + ジッター
```python
from src.day_trade.utils.api_resilience import RetryConfig

retry_config = RetryConfig(
    max_attempts=3,        # 最大リトライ回数
    base_delay=1.0,        # 基本遅延時間（秒）
    max_delay=60.0,        # 最大遅延時間（秒）
    exponential_base=2.0,  # 指数倍数
    jitter=True           # ランダム化有効
)
```

**特徴**:
- 指数バックオフによる段階的遅延増加
- ジッター機能で同時リクエストの分散
- HTTPステータスコード別リトライ判定
- 設定可能な最大遅延時間

#### 利点
- **負荷分散**: 同時リクエストによるサーバー負荷を軽減
- **成功率向上**: 一時的な障害からの自動復旧
- **効率的**: 無駄な短間隔リトライを避ける

### 2. サーキットブレーカーパターン

#### 3段階状態管理
```python
from src.day_trade.utils.api_resilience import CircuitState

# CLOSED: 正常状態 - 全リクエスト通過
# OPEN: 遮断状態 - 全リクエスト拒否
# HALF_OPEN: 半開状態 - 試行的にリクエスト通過
```

#### 設定例
```python
from src.day_trade.utils.api_resilience import CircuitBreakerConfig

circuit_config = CircuitBreakerConfig(
    failure_threshold=5,     # 失敗閾値
    success_threshold=3,     # 成功閾値（半開状態で）
    timeout=60.0,           # 開状態のタイムアウト（秒）
    monitor_window=300.0    # 監視ウィンドウ（秒）
)
```

#### 動作フロー
1. **CLOSED → OPEN**: 失敗回数が閾値に達すると遮断
2. **OPEN → HALF_OPEN**: タイムアウト後に試行状態に移行
3. **HALF_OPEN → CLOSED**: 成功回数が閾値に達すると復旧
4. **HALF_OPEN → OPEN**: 失敗時は再び遮断

### 3. フェイルオーバー機能

#### 複数エンドポイント管理
```python
from src.day_trade.utils.api_resilience import APIEndpoint

endpoints = [
    APIEndpoint(
        name="primary",
        base_url="https://query1.finance.yahoo.com",
        priority=1,
        timeout=30.0,
        health_check_path="/v8/finance/chart/AAPL"
    ),
    APIEndpoint(
        name="secondary",
        base_url="https://query2.finance.yahoo.com",
        priority=2,
        timeout=30.0,
        health_check_path="/v8/finance/chart/AAPL"
    )
]
```

#### 自動ヘルスチェック
- 定期的なエンドポイント監視（デフォルト5分間隔）
- 失敗時の自動切り離し
- 復旧時の自動組み込み

### 4. 段階的フォールバック戦略

```python
# 1. 標準実行: 通常のAPI呼び出し
# 2. キャッシュフォールバック: 期限切れデータの利用
# 3. 劣化モード: デフォルトデータでサービス継続
```

#### キャッシュフォールバック
```python
# TTL期限切れでも一時的にキャッシュデータを利用
original_ttl = func._cache_ttl
func._cache_ttl = 3600 * 24  # 24時間に延長
result = func(*args, **kwargs)
```

#### 劣化モード
```python
# 最低限のデフォルトデータを返す
{
    "symbol": "7203.T",
    "current_price": 0.0,
    "data_quality": "degraded",
    "source": "fallback"
}
```

## 使用方法

### 基本的な使用

#### 拡張版StockFetcher
```python
from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher

# 全機能有効で初期化
fetcher = EnhancedStockFetcher(
    enable_fallback=True,           # フォールバック有効
    enable_circuit_breaker=True,    # サーキットブレーカー有効
    enable_health_monitoring=True,  # ヘルスモニタリング有効
    retry_count=3,                  # リトライ回数
    retry_delay=1.0                 # リトライ遅延
)

# 耐障害性機能付きで価格取得
price_data = fetcher.get_current_price("7203")
print(f"価格: {price_data['current_price']}")
print(f"データ品質: {price_data['data_quality']}")
```

#### システム状態監視
```python
# システム全体の状態を取得
status = fetcher.get_system_status()
print(f"サービス: {status['service']}")
print(f"全体ヘルス: {status['resilience_status']['overall_health']}")

# ヘルスチェック実行
health = fetcher.health_check()
print(f"ステータス: {health['status']}")
print(f"応答時間: {health['response_time_ms']:.2f}ms")
```

### 高度な使用

#### カスタム耐障害性クライアント
```python
from src.day_trade.utils.api_resilience import ResilientAPIClient

# カスタム設定
client = ResilientAPIClient(
    endpoints=endpoints,
    retry_config=retry_config,
    circuit_config=circuit_config,
    enable_health_check=True,
    health_check_interval=300.0
)

# HTTPリクエスト実行
response = client.get("/api/data")
```

#### デコレータ使用
```python
from src.day_trade.utils.api_resilience import resilient_api_call

@resilient_api_call(endpoints, retry_config, circuit_config)
def fetch_data(client):
    return client.get("/api/stock_data")

# 自動的に耐障害性機能が適用される
data = fetch_data()
```

## 設定とカスタマイズ

### 環境変数

```bash
# リトライ設定
API_RETRY_MAX_ATTEMPTS=3
API_RETRY_BASE_DELAY=1.0
API_RETRY_MAX_DELAY=60.0

# サーキットブレーカー設定
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_SUCCESS_THRESHOLD=3
CIRCUIT_TIMEOUT=60.0

# ヘルスチェック設定
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=300
```

### プログラムによる設定

```python
# 本番環境用設定
production_config = {
    "retry": RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=120.0,
        jitter=True
    ),
    "circuit": CircuitBreakerConfig(
        failure_threshold=10,
        success_threshold=5,
        timeout=300.0
    )
}

# 開発環境用設定
development_config = {
    "retry": RetryConfig(
        max_attempts=2,
        base_delay=0.5,
        max_delay=10.0,
        jitter=False
    ),
    "circuit": CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0
    )
}
```

## 監視とログ

### 構造化ログ出力

```python
# API呼び出しログ
log_api_call("yfinance_enhanced", "GET", "ticker.info for 7203.T",
           context="enhanced_stock_fetcher")

# パフォーマンスメトリクス
log_performance_metric("enhanced_price_fetch_time", 588.2, "ms",
                     stock_code="7203", data_source="enhanced")

# ビジネスイベント
log_business_event("api_execution_success",
                 strategy="standard", function="get_current_price")
```

### セキュリティイベント

```python
# サーキットブレーカー開放
log_security_event("circuit_breaker_opened", "warning",
                 failure_count=5, threshold=5)

# 劣化モード移行
log_security_event("degraded_mode_activated", "warning",
                 function="get_current_price")
```

### メトリクス例

```json
{
  "timestamp": "2025-08-02T20:04:59.362841Z",
  "level": "info",
  "message": "Performance metric",
  "metric_name": "enhanced_price_fetch_time",
  "value": 588.238000869751,
  "unit": "ms",
  "stock_code": "7203",
  "data_source": "enhanced"
}
```

## トラブルシューティング

### よくある問題

#### 1. 全エンドポイントが利用不可
```bash
# 症状: "利用可能なAPIエンドポイントがありません"エラー
# 対処: ネットワーク接続確認、エンドポイント設定見直し

# ネットワーク確認
ping query1.finance.yahoo.com
ping query2.finance.yahoo.com

# エンドポイント状態確認
python -c "
from src.day_trade.data.enhanced_stock_fetcher import EnhancedStockFetcher
fetcher = EnhancedStockFetcher()
print(fetcher.get_system_status())
"
```

#### 2. サーキットブレーカーが頻繁に開く
```python
# 設定の調整
circuit_config = CircuitBreakerConfig(
    failure_threshold=10,  # 閾値を上げる
    timeout=300.0         # タイムアウトを延長
)
```

#### 3. パフォーマンスの劣化
```python
# キャッシュ設定の最適化
fetcher = EnhancedStockFetcher(
    price_cache_ttl=60,      # TTLを延長
    historical_cache_ttl=600, # TTLを延長
    enable_health_monitoring=False  # 必要に応じて無効化
)
```

### デバッグ方法

#### ログレベル設定
```bash
export LOG_LEVEL=DEBUG
python your_script.py
```

#### ヘルスチェック詳細
```python
health = fetcher.health_check()
for check_name, result in health['checks'].items():
    print(f"{check_name}: {result['status']} - {result['details']}")
```

#### サーキットブレーカー状態確認
```python
status = fetcher.get_system_status()
for endpoint in status['resilience_status']['endpoints']:
    print(f"{endpoint['name']}: {endpoint['circuit_state']}")
```

## パフォーマンス最適化

### ベストプラクティス

1. **適切なTTL設定**
   ```python
   # 頻繁に変更されるデータは短いTTL
   price_cache_ttl=30

   # 変更頻度の低いデータは長いTTL
   company_info_cache_ttl=3600
   ```

2. **エンドポイント優先度の設定**
   ```python
   # レスポンスの速いエンドポイントを優先
   APIEndpoint("fast_endpoint", "https://fast.api.com", priority=1)
   APIEndpoint("slow_endpoint", "https://slow.api.com", priority=2)
   ```

3. **ヘルスチェックの最適化**
   ```python
   # 負荷を考慮してチェック間隔を調整
   health_check_interval=600  # 10分間隔
   ```

### パフォーマンス指標

| メトリクス | 目標値 | 実測値 |
|-----------|--------|--------|
| API応答時間 | < 1000ms | 588ms |
| ヘルスチェック | < 1000ms | 590ms |
| キャッシュヒット率 | > 80% | 測定中 |
| 成功率 | > 99% | 測定中 |

## 今後の拡張予定

### 短期（1-2ヶ月）
- [ ] 予測的フェイルオーバー（予兆検知）
- [ ] 動的負荷分散
- [ ] メトリクス自動化

### 中期（3-6ヶ月）
- [ ] 機械学習による障害予測
- [ ] 自動チューニング機能
- [ ] マルチリージョン対応

### 長期（6ヶ月以上）
- [ ] カスタムプロトコル対応
- [ ] 分散トレーシング統合
- [ ] リアルタイム監視ダッシュボード

---

この機能により、Day Tradeシステムは高い可用性と信頼性を実現し、様々な障害状況下でも継続的なサービス提供が可能になります。
