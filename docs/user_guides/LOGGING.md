# 構造化ロギングガイド

Day Tradeアプリケーションの構造化ロギングシステムの使用方法について説明します。

## 概要

このアプリケーションでは、`structlog`ライブラリを使用した構造化ロギングを実装しています。これにより、ログの検索・分析・監視が容易になり、本番環境での問題追跡が効率的に行えます。

## 基本的な使用方法

### ロガーの取得

```python
from day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
```

### 基本的なログ出力

```python
# 情報ログ
logger.info("処理開始", function="calculate_rsi", symbol="7203")

# 警告ログ
logger.warning("データ不足", symbol="7203", missing_days=5)

# エラーログ
logger.error("API接続エラー", api="yfinance", status_code=500)
```

## 構造化データの活用

### ビジネスイベントのログ

```python
from day_trade.utils.logging_config import log_business_event

# 取引実行
log_business_event(
    "trade_executed",
    symbol="7203",
    trade_type="BUY",
    quantity=100,
    price=2500.0,
    commission=250.0
)

# シグナル生成
log_business_event(
    "signal_generated",
    symbol="7203",
    signal_type="BUY",
    strength="STRONG",
    confidence=85.6,
    reasons=["golden_cross", "volume_spike"]
)
```

### パフォーマンスメトリクス

```python
from day_trade.utils.logging_config import log_performance_metric

# 処理時間の記録
log_performance_metric(
    "data_fetch_time",
    value=1250.0,
    unit="ms",
    symbol="7203",
    data_points=100
)

# 精度メトリクス
log_performance_metric(
    "prediction_accuracy",
    value=87.5,
    unit="percent",
    model="ensemble",
    period="30days"
)
```

### データベース操作

```python
from day_trade.utils.logging_config import log_database_operation

# データベース操作の記録
log_database_operation(
    "INSERT",
    "trades",
    trade_id="T123456",
    symbol="7203",
    rows_affected=1
)

log_database_operation(
    "UPDATE",
    "positions",
    symbol="7203",
    rows_affected=1,
    execution_time=45.2
)
```

### API呼び出し

```python
from day_trade.utils.logging_config import log_api_call

# 外部API呼び出しの記録
log_api_call(
    "yfinance",
    "GET",
    "https://query1.finance.yahoo.com/v8/finance/chart/7203.T",
    status_code=200,
    response_time=856.0,
    data_points=100
)
```

### エラーハンドリング

```python
from day_trade.utils.logging_config import log_error_with_context

try:
    # 何らかの処理
    result = risky_operation()
except Exception as e:
    log_error_with_context(
        e,
        context={
            "symbol": "7203",
            "operation": "price_calculation",
            "input_data": input_params
        }
    )
```

## ログレベルと出力形式

### 環境変数での設定

```bash
# ログレベル設定
export LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ログ形式設定
export LOG_FORMAT=json  # json, console

# 環境設定
export ENVIRONMENT=production  # development, production
```

### 開発環境と本番環境

**開発環境（ENVIRONMENT=development）:**
- カラー付きコンソール出力
- 人間が読みやすい形式
- デバッグ情報の表示

**本番環境（ENVIRONMENT=production）:**
- JSON形式での出力
- 構造化データ
- ログ集約システムとの連携に適した形式

## ログ出力例

### 開発環境での出力例

```
2024-08-02T10:30:45.123456Z [info     ] 売買シグナル生成 [day_trade.analysis.signals] confidence=85.6 market_context={'volatility': 0.025, 'trend_direction': 'upward'} price=2500.0 reasons=['golden_cross', 'volume_spike'] section=signal_generation signal_type=BUY strength=STRONG timestamp=2024-08-02T10:30:45.123456 validity_score=87.5
```

### 本番環境での出力例

```json
{
  "timestamp": "2024-08-02T10:30:45.123456Z",
  "level": "info",
  "logger": "day_trade.analysis.signals",
  "event": "売買シグナル生成",
  "section": "signal_generation",
  "signal_type": "BUY",
  "strength": "STRONG",
  "confidence": 85.6,
  "price": 2500.0,
  "timestamp": "2024-08-02T10:30:45.123456",
  "reasons": ["golden_cross", "volume_spike"],
  "market_context": {
    "volatility": 0.025,
    "trend_direction": "upward"
  },
  "validity_score": 87.5
}
```

## ログの分析と監視

### ログクエリの例

```bash
# エラーログの検索
grep '"level": "error"' app.log | jq .

# 特定シンボルの取引ログ
grep '"symbol": "7203"' app.log | grep '"event": "trade_executed"' | jq .

# パフォーマンスメトリクスの抽出
grep '"event": "Performance metric"' app.log | jq '.metric_name, .value, .unit'

# シグナル生成の分析
grep '"section": "signal_generation"' app.log | jq '.signal_type, .confidence, .validity_score'
```

### ログ集約システムとの連携

JSON形式のログは以下のシステムと連携できます：

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Fluentd**
- **Grafana Loki**
- **AWS CloudWatch Logs**
- **Google Cloud Logging**

## セクション分類

ログでは`section`フィールドを使用して処理を分類しています：

- `signal_generation`: シグナル生成
- `trade_execution`: 取引実行
- `portfolio_analysis`: ポートフォリオ分析
- `data_fetch`: データ取得
- `error_handling`: エラー処理
- `performance_measurement`: パフォーマンス計測
- `database_operation`: データベース操作
- `api_call`: API呼び出し

## ベストプラクティス

### 1. 一貫性のあるフィールド名

```python
# Good: 一貫したフィールド名
logger.info("取引実行", symbol="7203", quantity=100, price=2500.0)

# Bad: 不一致なフィールド名
logger.info("取引実行", stock_code="7203", shares=100, cost=2500.0)
```

### 2. 適切なログレベル

```python
# DEBUG: 詳細な処理情報
logger.debug("計算中", step="rsi", period=14, current_value=65.4)

# INFO: 重要なビジネスイベント
logger.info("取引実行", symbol="7203", result="success")

# WARNING: 注意すべき状況
logger.warning("データ遅延", delay_minutes=5, symbol="7203")

# ERROR: エラー状況
logger.error("API接続失敗", api="yfinance", retry_count=3)
```

### 3. コンテキスト情報の提供

```python
# 処理のコンテキストを含める
logger.info(
    "ポートフォリオ更新",
    section="portfolio_management",
    user_id="user123",
    total_value=1000000.0,
    positions_count=5,
    last_updated="2024-08-02T10:30:45Z"
)
```

### 4. 機密情報の除外

```python
# Good: 機密情報を除外
logger.info("ユーザー認証", user_id="user123", auth_method="api_key")

# Bad: 機密情報を含める
logger.info("ユーザー認証", user_id="user123", password="secret123")
```

## トラブルシューティング

### ログが出力されない場合

1. **ログレベルの確認**
   ```python
   import os
   print(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'INFO')}")
   ```

2. **ロガー設定の確認**
   ```python
   from day_trade.utils.logging_config import setup_logging
   setup_logging()  # 明示的に初期化
   ```

3. **出力先の確認**
   ```python
   import logging
   print(f"Root logger handlers: {logging.root.handlers}")
   ```

### パフォーマンスの最適化

```python
# 重い処理のログは条件付きで
if logger.isEnabledFor(logging.DEBUG):
    expensive_data = calculate_expensive_metrics()
    logger.debug("詳細メトリクス", **expensive_data)
```

## 参考リンク

- [structlog Documentation](https://www.structlog.org/)
- [Python Logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Twelve-Factor App: Logs](https://12factor.net/logs)
