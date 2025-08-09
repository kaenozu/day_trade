# 例外処理体系化改善計画

## 📋 現状分析

### 問題点
- **汎用例外の多用**: 1,146箇所で`except Exception`を使用
- **詳細情報不足**: エラーの種類や原因が特定困難
- **不適切なエラーハンドリング**: 例外を隠蔽する処理が存在
- **統一性の欠如**: モジュール間でのエラー処理方式が不統一

### 良い点
- **例外クラス体系**: `src/day_trade/utils/exceptions.py`に包括的な例外クラスが定義済み ✅
- **エラーハンドラ**: `enhanced_error_handler.py`で統一的エラー処理基盤が存在 ✅

## 🎯 改善目標

### 短期目標（1-2週間）
1. **主要コンポーネントでの例外処理改善**
   - AnalysisOnlyEngine の例外処理体系化
   - データ取得系モジュールの例外分類
   - 重要な20ファイルでの汎用例外を具体例外に変更

### 中期目標（1ヶ月）
2. **例外処理ベストプラクティス確立**
   - 統一例外処理パターンの定義
   - ログ出力標準化
   - エラー回復戦略の実装

### 長期目標（2ヶ月）
3. **全システム統一**
   - 全144ファイルの例外処理見直し
   - 監視・アラート機能強化

## 🔧 具体的改善項目

### 1. 分析エンジンの例外処理改善

**現在の問題例**:
```python
# analysis_only_engine.py:192
except Exception as e:
    logger.error(f"分析処理中にエラーが発生: {e}")
```

**改善後**:
```python
from src.day_trade.utils.exceptions import (
    AnalysisError, DataError, NetworkError,
    handle_database_exception, handle_network_exception
)

try:
    # 分析処理
    await self._perform_market_analysis()
except (DataError, NetworkError) as e:
    logger.error("予期されたエラーが発生", extra={
        "error_type": e.__class__.__name__,
        "error_code": e.error_code,
        "details": e.details
    })
    self._handle_recoverable_error(e)
except AnalysisError as e:
    logger.error("分析処理エラー", extra=e.to_dict())
    self._update_statistics(analysis_time, success=False)
except Exception as e:
    logger.critical("予期しないエラー", extra={
        "error": str(e),
        "traceback": traceback.format_exc()
    })
    raise AnalysisError("重大な分析エラー", "ANALYSIS_CRITICAL", {"original": str(e)})
```

### 2. データ取得系の例外処理改善

**対象ファイル**:
- `src/day_trade/data/stock_fetcher.py`
- `src/day_trade/data/advanced_ml_engine.py`
- `src/day_trade/data/batch_data_fetcher.py`

**改善パターン**:
```python
try:
    data = await self.fetch_data(symbol)
except requests.exceptions.ConnectionError as e:
    raise handle_network_exception(e)
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        raise RateLimitError("API呼び出し上限", "RATE_LIMIT", {
            "retry_after": e.response.headers.get("Retry-After")
        })
    raise handle_network_exception(e)
except ValueError as e:
    raise DataError("データ形式エラー", "DATA_FORMAT", {"symbol": symbol})
```

### 3. データベース操作の例外処理改善

**対象ファイル**:
- `src/day_trade/models/database.py`
- `src/day_trade/models/optimized_database.py`

**改善パターン**:
```python
try:
    result = session.execute(query)
except sqlalchemy.exc.IntegrityError as e:
    raise handle_database_exception(e)
except sqlalchemy.exc.OperationalError as e:
    raise handle_database_exception(e)
```

## 🏗️ 実装計画

### Phase 1: 基盤強化（1週間）

#### A. 例外処理ユーティリティ拡張
```python
# src/day_trade/utils/exception_handler.py (新規作成)
class ExceptionContext:
    """例外コンテキスト管理"""

    def __init__(self, component: str, operation: str):
        self.component = component
        self.operation = operation

    def handle(self, exc: Exception) -> DayTradeError:
        """統一例外変換"""
        if isinstance(exc, requests.exceptions.RequestException):
            return handle_network_exception(exc)
        elif isinstance(exc, sqlalchemy.exc.SQLAlchemyError):
            return handle_database_exception(exc)
        # ... その他の変換ロジック

def with_exception_handling(component: str, operation: str):
    """例外処理デコレータ"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context = ExceptionContext(component, operation)
            try:
                return await func(*args, **kwargs)
            except DayTradeError:
                raise  # すでに変換済みは再スロー
            except Exception as e:
                logger.error(f"{component}.{operation} failed",
                           extra={"error": str(e)})
                raise context.handle(e)
        return wrapper
    return decorator
```

#### B. ログ構造統一
```python
# 統一ログ出力関数
def log_exception(logger, exc: Exception, context: dict = None):
    """統一例外ログ出力"""
    if isinstance(exc, DayTradeError):
        logger.error("ビジネスエラー", extra={
            **exc.to_dict(),
            **(context or {})
        })
    else:
        logger.critical("システムエラー", extra={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
            **(context or {})
        })
```

### Phase 2: 主要コンポーネント改善（1週間）

#### 優先対象ファイル（20ファイル）
1. `automation/analysis_only_engine.py` ⚡ 最優先
2. `data/stock_fetcher.py` ⚡ 最優先
3. `automation/trading_engine.py`
4. `data/advanced_ml_engine.py`
5. `models/database.py`
6. `analysis/enhanced_report_manager.py`
7. `core/integrated_analysis_system.py`
8. `data/batch_data_fetcher.py`
9. `dashboard/analysis_dashboard_server.py`
10. `api/websocket_streaming_client.py`

### Phase 3: 全体適用（2-3週間）
- 残り124ファイルの例外処理改善
- 自動検出ツールによる品質確認
- テストケース追加

## ✅ 成功指標

### 量的指標
| 項目 | 現在 | 目標 | 期限 |
|------|------|------|------|
| 汎用`except Exception`数 | 1,146 | 200以下 | Phase 3完了時 |
| 具体例外処理率 | 20% | 80% | Phase 3完了時 |
| エラーログの構造化率 | 30% | 90% | Phase 2完了時 |

### 質的指標
- **エラー特定速度**: トラブル発生時の原因特定時間を50%短縮
- **回復性**: 一時的エラーからの自動回復率を80%に向上
- **監視性**: 構造化ログによる異常検知精度向上

## 🔍 実装例

### AnalysisOnlyEngine改善例

**改善前**:
```python
try:
    await self._perform_market_analysis()
except Exception as e:
    logger.error(f"分析処理中にエラーが発生: {e}")
```

**改善後**:
```python
from src.day_trade.utils.exceptions import AnalysisError, DataError
from src.day_trade.utils.exception_handler import log_exception

try:
    await self._perform_market_analysis()
except DataError as e:
    log_exception(logger, e, {"component": "AnalysisEngine", "operation": "market_analysis"})
    self._update_statistics(analysis_time, success=False)
    # データエラーは処理を継続
except AnalysisError as e:
    log_exception(logger, e, {"component": "AnalysisEngine"})
    self.status = AnalysisStatus.ERROR
    self._notify_error(e)
except Exception as e:
    # 予期しないエラーは適切な例外に変換
    analysis_error = AnalysisError(
        message="分析処理で予期しないエラーが発生",
        error_code="ANALYSIS_UNEXPECTED_ERROR",
        details={"original_error": str(e), "operation": "market_analysis"}
    )
    log_exception(logger, analysis_error, {"component": "AnalysisEngine"})
    raise analysis_error
```

## 📊 期待効果

### 短期効果
- **デバッグ効率**: エラー原因の特定速度向上
- **システム安定性**: 適切なエラー回復による稼働率向上
- **コード品質**: 統一的な例外処理による保守性向上

### 長期効果
- **運用効率**: 自動化されたエラー処理と回復
- **ユーザー体験**: より分かりやすいエラーメッセージ
- **システム監視**: 構造化ログによる高度な監視・分析

---
*改善計画策定日: 2025-08-09*
*対象ブランチ: feature/advanced-ml-tuning-system*
