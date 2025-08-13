#!/usr/bin/env python3
"""
yfinance統一インポートユーティリティ - Issue #614対応

プロジェクト全体でyfinanceのインポートとエラーハンドリングを標準化
"""

import logging
from typing import Optional, Tuple

from .logging_config import get_context_logger

logger = get_context_logger(__name__)

# グローバル変数でyfinanceの利用可能性を管理
_yfinance_available: Optional[bool] = None
_yfinance_module = None


def get_yfinance() -> Tuple[Optional[object], bool]:
    """
    yfinanceモジュールの統一取得関数

    Returns:
        Tuple[Optional[object], bool]: (yfinanceモジュール, 利用可能フラグ)
    """
    global _yfinance_available, _yfinance_module

    # 初回チェックまたはモジュールが未設定の場合
    if _yfinance_available is None:
        try:
            import yfinance as yf
            _yfinance_module = yf
            _yfinance_available = True
            logger.info("yfinance利用可能")

        except ImportError as e:
            _yfinance_module = None
            _yfinance_available = False
            logger.warning(
                "yfinance未インストール - 次のコマンドでインストールしてください: "
                "pip install yfinance"
            )
            logger.debug(f"yfinanceインポートエラー: {e}")

    return _yfinance_module, _yfinance_available


def require_yfinance():
    """
    yfinanceが利用可能であることを必須とする関数

    Raises:
        ImportError: yfinanceが利用できない場合
    """
    yf_module, available = get_yfinance()

    if not available:
        raise ImportError(
            "yfinanceが必要ですが利用できません。"
            "次のコマンドでインストールしてください: pip install yfinance"
        )

    return yf_module


def is_yfinance_available() -> bool:
    """
    yfinanceが利用可能かチェック

    Returns:
        bool: yfinanceが利用可能な場合True
    """
    _, available = get_yfinance()
    return available


def get_yfinance_ticker(symbol: str):
    """
    yfinance Tickerオブジェクトを安全に取得

    Args:
        symbol: 銘柄コード

    Returns:
        yfinance.Ticker or None: Tickerオブジェクト（yfinanceが利用できない場合はNone）

    Raises:
        ImportError: yfinanceが利用できない場合
    """
    yf_module = require_yfinance()
    return yf_module.Ticker(symbol)


def safe_yfinance_operation(operation_name: str = "yfinance操作"):
    """
    yfinance操作を安全に実行するためのデコレータ

    Args:
        operation_name: 操作名（ログ出力用）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if not is_yfinance_available():
                    logger.error(f"{operation_name}失敗: yfinanceが利用できません")
                    return None

                return func(*args, **kwargs)

            except Exception as e:
                logger.error(f"{operation_name}エラー: {e}")
                logger.debug(f"{operation_name}詳細エラー", exc_info=True)
                return None

        return wrapper
    return decorator


# 後方互換性のための定数とエイリアス
def init_yfinance_compat():
    """後方互換性のためのyfinance初期化"""
    global YFINANCE_AVAILABLE, yf

    yf_module, available = get_yfinance()
    YFINANCE_AVAILABLE = available

    if available:
        yf = yf_module
    else:
        yf = None

    return yf_module, available


# 標準的な使用パターン
if __name__ == "__main__":
    # 使用例
    print("=== yfinance統一インポートユーティリティテスト ===")

    # 基本チェック
    yf_module, available = get_yfinance()
    print(f"yfinance利用可能: {available}")

    if available:
        print(f"yfinanceバージョン: {yf_module.__version__}")

        try:
            # Tickerオブジェクト作成テスト
            ticker = get_yfinance_ticker("AAPL")
            print(f"Tickerオブジェクト作成成功: {type(ticker)}")

        except Exception as e:
            print(f"Tickerオブジェクト作成エラー: {e}")

    print("テスト完了")