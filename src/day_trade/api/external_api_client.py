#!/usr/bin/env python3
"""
外部APIクライアントシステム（レガシー版 - 後方互換性のためのリダイレクト）

⚠️  警告: このファイルは非推奨です。
新しいモジュラー構造を使用してください: day_trade.api.external_api

このファイルは300行制限により分割されました：
- day_trade.api.external_api.api_client (メインクライアント - 326行)
- day_trade.api.external_api.auth_manager (認証管理 - 129行)
- day_trade.api.external_api.cache_manager (キャッシュ管理 - 127行)
- day_trade.api.external_api.data_normalizers (データ正規化 - 251行)
- day_trade.api.external_api.enums (列挙型 - 40行)
- day_trade.api.external_api.error_handlers (エラー処理 - 156行)
- day_trade.api.external_api.models (データモデル - 231行)
- day_trade.api.external_api.rate_limiter (レート制限 - 191行)
- day_trade.api.external_api.request_executor (リクエスト実行 - 170行)
- day_trade.api.external_api.url_builder (URL構築 - 137行)
"""

import warnings

warnings.warn(
    "external_api_client.py は非推奨です。新しいモジュラー構造 day_trade.api.external_api を使用してください。",
    DeprecationWarning,
    stacklevel=2
)

# 後方互換性のため、新しいモジュールからすべてのクラス・関数をインポート
from .external_api import *