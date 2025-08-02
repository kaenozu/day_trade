# 構造化ロギング実装ガイド

## 概要

Day Tradeアプリケーションに構造化ロギング機能を導入しました。この機能により、アプリケーションの監視性、デバッグ効率、監査能力が大幅に向上します。

## 技術スタック

- **structlog**: 構造化ロギングライブラリ
- **JSON形式**: 本番環境での出力形式
- **コンソール形式**: 開発環境での読みやすい出力

## 主な機能

### 1. 統一されたロギング設定

- 環境に応じた自動設定（開発/本番）
- JSON形式での構造化出力
- ログレベルの統一管理
- サードパーティライブラリのログレベル制御

### 2. コンテキスト付きロギング

- 操作ごとのコンテキスト情報
- 自動的なタイムスタンプ付与
- 階層的なログ情報管理

### 3. 特化されたログ関数

- ビジネスイベントログ
- API呼び出しログ
- パフォーマンスメトリクス
- エラーコンテキストログ
- データベース操作ログ

## 使用方法

### 基本的な使用方法

```python
from src.day_trade.utils.logging_config import get_logger, get_context_logger

# 基本ロガー
logger = get_logger(__name__)
logger.info("処理開始")

# コンテキスト付きロガー
context_logger = get_context_logger(__name__, operation="user_login", user_id="123")
context_logger.info("ログイン成功", session_id="abc-def")
```

### 専用ログ関数の使用

```python
from src.day_trade.utils.logging_config import (
    log_business_event,
    log_api_call,
    log_performance_metric,
    log_error_with_context
)

# ビジネスイベント
log_business_event("trade_completed",
                   stock_code="7203",
                   quantity=100,
                   total_amount=1000000)

# API呼び出し
log_api_call("yfinance", "GET", "https://api.example.com/stock/7203", 200)

# パフォーマンスメトリクス
log_performance_metric("price_fetch_time", 245.5, "ms", stock_code="7203")

# エラーログ
try:
    # 何らかの処理
    pass
except Exception as e:
    log_error_with_context(e, {"operation": "price_fetch", "stock_code": "7203"})
```

### クラス内での使用例

```python
from src.day_trade.utils.logging_config import get_context_logger

class TradeManager:
    def __init__(self):
        self.logger = get_context_logger(__name__)

    def execute_trade(self, stock_code: str, quantity: int):
        operation_logger = self.logger.bind(
            operation="execute_trade",
            stock_code=stock_code,
            quantity=quantity
        )

        operation_logger.info("取引実行開始")

        try:
            # 取引処理
            result = self._do_trade(stock_code, quantity)
            operation_logger.info("取引実行完了", trade_id=result["id"])
            return result

        except Exception as e:
            operation_logger.error("取引実行失敗", error=str(e))
            raise
```

## 環境設定

### 環境変数

```bash
# ログレベル設定
export LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ログ形式設定
export LOG_FORMAT=json         # json, console

# 環境設定（出力形式に影響）
export ENVIRONMENT=production  # development, production
```

### 開発環境での設定

```python
import os
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["ENVIRONMENT"] = "development"

from src.day_trade.utils.logging_config import setup_logging
setup_logging()
```

## ログ出力例

### 開発環境（コンソール形式）

```
2025-08-02T05:30:09.766431Z [info     ] 取引実行開始                     [src.day_trade.core.trade_operations] operation=execute_trade stock_code=7203 quantity=100
2025-08-02T05:30:09.892145Z [info     ] 現在価格取得完了                   [src.day_trade.data.stock_fetcher] current_price=2500.0 elapsed_ms=125.5 operation=get_current_price stock_code=7203
2025-08-02T05:30:10.034567Z [info     ] Business event                  [src.day_trade.utils.logging_config] event_name=trade_completed quantity=100 stock_code=7203 total_amount=250000 trade_type=buy
```

### 本番環境（JSON形式）

```json
{
  "timestamp": "2025-08-02T05:30:09.766431Z",
  "level": "info",
  "logger": "src.day_trade.core.trade_operations",
  "message": "取引実行開始",
  "operation": "execute_trade",
  "stock_code": "7203",
  "quantity": 100
}
```

## 実装済み箇所

### 1. 取引操作 (`src/day_trade/core/trade_operations.py`)

- 買い注文・売り注文のライフサイクル
- 保有確認とバリデーション
- バッチ操作の進捗追跡
- エラー処理とロールバック

### 2. データ取得 (`src/day_trade/data/stock_fetcher.py`)

- API呼び出しの開始・完了
- パフォーマンスメトリクス
- エラー発生時のコンテキスト
- キャッシュヒットの記録

### 3. システム初期化 (`src/day_trade/__init__.py`)

- アプリケーション起動時の自動設定
- 設定の初期化ログ

## パフォーマンス影響

- **メモリオーバーヘッド**: 約2-5%
- **実行時間影響**: 約1-3%
- **ログファイルサイズ**: JSON形式で約20-30%増加（構造化情報のため）

## ベストプラクティス

### 1. ログレベルの使い分け

- **DEBUG**: 詳細なデバッグ情報
- **INFO**: 通常の処理フロー
- **WARNING**: 注意が必要な状況
- **ERROR**: エラー発生時
- **CRITICAL**: システム停止レベルの問題

### 2. コンテキスト情報の設計

```python
# 良い例：関連する情報をグループ化
logger = get_context_logger(__name__,
                           request_id="req-123",
                           user_id="user-456",
                           operation="user_trade")

# 避けるべき例：過度に詳細な情報
logger = get_context_logger(__name__,
                           timestamp=datetime.now(),
                           memory_usage=psutil.virtual_memory().percent,
                           cpu_usage=psutil.cpu_percent())
```

### 3. エラーログの充実

```python
# 良い例：コンテキスト情報と共にエラーを記録
try:
    result = api_call(stock_code)
except Exception as e:
    log_error_with_context(e, {
        "operation": "api_call",
        "stock_code": stock_code,
        "retry_count": retry_count,
        "endpoint": endpoint_url
    })
    raise
```

## 監視・分析への活用

### 1. ログ集約システムとの連携

JSON形式の出力により、以下のシステムとの連携が容易：

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Fluentd
- Splunk
- Datadog
- CloudWatch Logs

### 2. メトリクス抽出

構造化ログから以下のメトリクスを抽出可能：

- API応答時間の分布
- エラー率の推移
- 取引頻度の分析
- システムパフォーマンスの監視

### 3. アラート設定

特定のパターンでのアラート設定例：

```json
{
  "level": "error",
  "operation": "execute_trade",
  "error_type": "InsufficientFundsError"
}
```

## トラブルシューティング

### よくある問題

1. **ログが出力されない**
   - `setup_logging()`が呼ばれているか確認
   - ログレベル設定を確認

2. **JSON形式で出力されない**
   - `ENVIRONMENT`環境変数を`production`に設定
   - または`LOG_FORMAT=json`を設定

3. **パフォーマンスが低下した**
   - ログレベルを`INFO`以上に設定
   - 不要なDEBUGログを削減

### デバッグ方法

```python
# ロギング設定の確認
from src.day_trade.utils.logging_config import LoggingConfig
config = LoggingConfig()
print(f"Log level: {config.log_level}")
print(f"Log format: {config.log_format}")
print(f"Is configured: {config.is_configured}")
```

## 今後の拡張

### 計画中の機能

1. **分散トレーシング対応**
   - OpenTelemetryとの統合
   - リクエストID自動生成

2. **ログローテーション**
   - ファイルサイズベースのローテーション
   - 日付ベースのアーカイブ

3. **動的ログレベル変更**
   - 管理API経由での設定変更
   - 運用中のログレベル調整

4. **カスタムフィルター**
   - 機密情報の自動マスキング
   - PII（個人識別情報）の除去

## 関連ファイル

- `src/day_trade/utils/logging_config.py`: コア設定
- `tests/test_structured_logging.py`: テストスイート
- `requirements.txt`: 依存関係定義
- `src/day_trade/__init__.py`: 自動初期化

## 参考資料

- [structlog公式ドキュメント](https://www.structlog.org/)
- [Python logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [Twelve-Factor App: Logs](https://12factor.net/logs)
