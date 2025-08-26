#!/usr/bin/env python3
"""
包括的セキュリティ管制センター - 後方互換性レイヤー
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

このファイルは後方互換性のために残されています。
実際の実装は control_center/ モジュールに分割されています。
"""

# 後方互換性のためのインポート
from .control_center import *

import warnings

warnings.warn(
    "comprehensive_security_control_center.py は非推奨です。"
    "代わりに day_trade.security.control_center からインポートしてください。",
    DeprecationWarning,
    stacklevel=2
)