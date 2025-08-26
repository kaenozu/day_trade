"""
後方互換性のためのエイリアスファイル
元のenhanced_error_handler.pyの機能はerror_handling/パッケージに移動されました
"""

# すべての機能をerror_handlingパッケージから再エクスポート
from .error_handling import *  # noqa: F401, F403

# 後方互換性のメッセージ（デバッグ時に確認可能）
__compatibility_message__ = """
このファイルは後方互換性のために提供されています。
新しいコードでは以下のインポートを使用してください:

from src.day_trade.utils.error_handling import EnhancedErrorHandler
from src.day_trade.utils.error_handling import create_error_handler
"""