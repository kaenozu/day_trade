# エラーハンドリングガイドライン

## 概要

Day Tradeシステムでは統一的なエラーハンドリングアプローチを採用しています。このガイドラインでは、適切なエラーハンドリングの実装方法について説明します。

## 基本原則

1. **統一的な例外クラスの使用**: `src/day_trade/utils/exceptions.py`で定義されたカスタム例外クラスを使用する
2. **構造化ロギング**: `src/day_trade/utils/logging_config.py`の統一的ロギング機能を使用する
3. **ユーザーフレンドリーエラー表示**: `EnhancedErrorHandler`を使用してユーザーに分かりやすいエラーメッセージを表示する
4. **機密情報の保護**: エラー情報から機密データを自動的にサニタイズする

## 実装パターン

### 1. 基本的なimport文

```python
from ..utils.enhanced_error_handler import (
    get_default_error_handler,
    handle_cli_error,
)
from ..utils.exceptions import (
    # 適切な例外クラスをimport
    DatabaseError,
    APIError,
    ValidationError,
    # etc.
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()
```

### 2. 例外の分類と使用

#### データベース関連
- `DatabaseError`: 一般的なDB関連エラー
- `DatabaseConnectionError`: DB接続エラー
- `DatabaseIntegrityError`: データ整合性エラー
- `DatabaseOperationalError`: DB操作エラー

#### API・ネットワーク関連
- `APIError`: 一般的なAPI関連エラー
- `NetworkError`: ネットワーク接続エラー
- `RateLimitError`: レート制限エラー
- `AuthenticationError`: 認証エラー
- `ResourceNotFoundError`: リソース未検出エラー

#### データ・ビジネスロジック関連
- `DataError`: データ関連エラー
- `ValidationError`: データ検証エラー
- `TradingError`: 取引関連エラー
- `AnalysisError`: 分析関連エラー
- `BacktestError`: バックテスト関連エラー

### 3. エラーハンドリングの実装例

#### 良い例: 統一的なエラーハンドリング

```python
def process_trading_data(symbol: str, data: dict) -> dict:
    try:
        # ビジネスロジックの実装
        validated_data = validate_trading_data(data)
        result = execute_trading_logic(symbol, validated_data)
        return result

    except ValidationError as e:
        # 既にカスタム例外の場合はそのまま再発生
        raise e

    except ValueError as e:
        # 標準例外をカスタム例外に変換
        validation_error = ValidationError(
            message="取引データの形式が正しくありません",
            error_code="TRADING_DATA_VALIDATION_ERROR",
            details={
                "symbol": symbol,
                "original_error": str(e)
            }
        )
        error_handler.log_error(
            validation_error,
            context={
                "user_action": "取引データ処理",
                "symbol": symbol
            }
        )
        raise validation_error

    except Exception as e:
        # 予期しない例外の処理
        trading_error = TradingError(
            message="取引データ処理中に予期しないエラーが発生しました",
            error_code="TRADING_UNEXPECTED_ERROR",
            details={"original_error": str(e)}
        )
        error_handler.log_error(
            trading_error,
            context={
                "user_action": "取引データ処理",
                "symbol": symbol
            }
        )
        raise trading_error
```

#### 悪い例: 統一されていないエラーハンドリング

```python
# 非推奨: 統一的でないエラーハンドリング
def process_trading_data(symbol: str, data: dict) -> dict:
    try:
        # ビジネスロジック
        return execute_trading_logic(symbol, data)
    except Exception as e:
        # 問題点:
        # 1. 一般的なloggingを使用
        # 2. 標準的なValueError/Exception を直接raise
        # 3. ユーザーフレンドリーなメッセージなし
        # 4. 機密情報のサニタイズなし
        logging.error(f"エラー: {e}")  # NG
        raise ValueError(f"処理エラー: {e}")  # NG
```

### 4. CLI向けエラーハンドリング

CLIアプリケーションでのエラー処理：

```python
def main_command(symbol: str):
    try:
        result = process_trading_command(symbol)
        print(f"処理成功: {result}")

    except (TradingError, ValidationError, DatabaseError) as e:
        # 統一的なCLIエラーハンドリング
        handle_cli_error(
            error=e,
            context={"user_input": symbol},
            user_action="取引コマンド実行",
            show_technical=False  # 一般ユーザーには技術的詳細を隠す
        )
        return 1

    except Exception as e:
        # 予期しない例外
        unexpected_error = TradingError(
            message="予期しないエラーが発生しました",
            error_code="UNEXPECTED_CLI_ERROR",
            details={"original_error": str(e)}
        )
        handle_cli_error(
            error=unexpected_error,
            context={"user_input": symbol},
            user_action="取引コマンド実行"
        )
        return 1

    return 0
```

### 5. ログレベルの使用指針

- `logger.debug()`: 開発者向け詳細情報
- `logger.info()`: 一般的な処理状況
- `logger.warning()`: 警告事項（処理は継続）
- `logger.error()`: エラー情報（処理に影響）
- `logger.critical()`: システムに重大な影響のある問題

### 6. パフォーマンス考慮事項

- 高頻度処理では`PerformanceCriticalLogger`を使用
- バッチ処理では`AggregatedLogger`でログを集約
- 機密情報のサニタイズはオプションで無効化可能

## テストでのエラーハンドリング

```python
def test_trading_error_handling():
    with pytest.raises(ValidationError) as exc_info:
        process_invalid_trading_data()

    assert exc_info.value.error_code == "TRADING_DATA_VALIDATION_ERROR"
    assert "取引データの形式" in str(exc_info.value)
```

## 設定項目

環境変数での設定:
- `ERROR_HANDLER_DEBUG_MODE`: デバッグモードの有効化
- `ERROR_HANDLER_ENABLE_SANITIZATION`: 機密情報サニタイズの有効化
- `LOG_LEVEL`: ログレベル設定

## まとめ

- 常にカスタム例外クラスを使用する
- `EnhancedErrorHandler`で統一的なエラー処理を行う
- ユーザーフレンドリーなエラーメッセージを提供する
- 機密情報の漏洩を防ぐ
- 構造化ロギングで一貫性を保つ

このガイドラインに従うことで、保守性が高く、ユーザーフレンドリーなエラーハンドリングシステムを構築できます。
