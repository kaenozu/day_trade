"""
バリデーション用ユーティリティ
"""

import re
from typing import List, Optional


def validate_stock_code(code: str) -> bool:
    """
    証券コードの妥当性を検証

    Args:
        code: 証券コード

    Returns:
        有効な証券コードの場合True
    """
    if not code:
        return False

    # 日本株の場合（4桁の数字 + オプションで市場コード）
    if re.match(r"^\d{4}(\.T)?$", code.upper()):
        return True

    # その他の形式（アルファベット含む）
    return bool(re.match(r"^[A-Z0-9]{1,10}(\.T)?$", code.upper()))


def validate_period(period: str) -> bool:
    """
    期間パラメータの妥当性を検証

    Args:
        period: 期間文字列

    Returns:
        有効な期間の場合True
    """
    valid_periods = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ]
    return period in valid_periods


def validate_interval(interval: str) -> bool:
    """
    間隔パラメータの妥当性を検証

    Args:
        interval: 間隔文字列

    Returns:
        有効な間隔の場合True
    """
    valid_intervals = [
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    ]
    return interval in valid_intervals


def normalize_stock_codes(codes: List[str]) -> List[str]:
    """
    証券コードのリストを正規化

    Args:
        codes: 証券コードのリスト

    Returns:
        正規化された証券コードのリスト
    """
    normalized = []
    for code in codes:
        if validate_stock_code(code):
            # 大文字に変換し、必要に応じて.Tを追加
            normalized_code = code.upper()
            if re.match(r"^\d{4}$", normalized_code):
                normalized_code += ".T"
            normalized.append(normalized_code)
    return normalized


def suggest_stock_code_correction(code: str) -> Optional[str]:
    """
    証券コードの修正候補を提案

    Args:
        code: 証券コード

    Returns:
        修正候補があれば返す
    """
    if not code:
        return None

    # 4桁の数字の場合、.Tを追加を提案
    if re.match(r"^\d{4}$", code):
        return f"{code}.T"

    # 小文字が含まれている場合、大文字化を提案
    if code != code.upper():
        return code.upper()

    return None
