"""
取引関連ユーティリティ関数

trade_manager.py からのリファクタリング抽出
共通ユーティリティ関数: safe_decimal_conversion, validate_positive_decimal, mask_sensitive_info
"""

import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Union


def safe_decimal_conversion(
    value: Union[str, int, float, Decimal], context: str = "値"
) -> Decimal:
    """
    安全なDecimal変換（浮動小数点誤差回避・強化版）

    金銭計算に使用する数値をDecimalに変換し、浮動小数点数の精度問題を回避する。
    企業レベルの会計処理に適した精度保証を提供。

    Args:
        value: 変換する値
        context: エラー時のコンテキスト情報

    Returns:
        Decimal: 変換された値（精度保証済み）

    Raises:
        ValueError: 変換不可能な値
        TypeError: 不正な型
    """
    if isinstance(value, Decimal):
        # 既にDecimalの場合も精度を確認
        if not value.is_finite():
            raise ValueError(f"Decimal値が無限大またはNaN: {context} = {value}")
        return value

    try:
        # 1. None値チェック
        if value is None:
            raise ValueError(f"{context}がNoneです")

        # 2. 型別変換処理
        if isinstance(value, int):
            # intの場合は直接変換（精度問題なし）
            return Decimal(value)

        elif isinstance(value, str):
            # 文字列の場合は空白除去と基本検証
            clean_value = str(value).strip()
            if not clean_value:
                raise ValueError(f"空の文字列: {context}")

            # 危険な文字列パターンをチェック
            dangerous_patterns = ["inf", "-inf", "nan", "null", "undefined"]
            if clean_value.lower() in dangerous_patterns:
                raise ValueError(f"無効な数値文字列: {context} = {clean_value}")

            # 基本的な数値形式チェック
            if not re.match(r"^-?\d*\.?\d*$", clean_value.replace(",", "")):
                raise ValueError(f"数値形式ではありません: {context} = {clean_value}")

            # カンマ区切りを除去
            clean_value = clean_value.replace(",", "")
            return Decimal(clean_value)

        elif isinstance(value, float):
            # floatの場合は特に慎重に処理
            # 1. 特殊値をチェック
            import math

            if math.isinf(value) or math.isnan(value):
                raise ValueError(f"無限大またはNaN: {context} = {value}")

            # 2. 極端に大きい値や小さい値をチェック
            if abs(value) > 1e15:  # 京を超える値
                raise ValueError(f"値が大きすぎます: {context} = {value}")
            if abs(value) < 1e-10 and value != 0:  # 極小値
                raise ValueError(f"値が小さすぎます: {context} = {value}")

            # 3. repr()を使って最高精度の文字列表現を取得
            # これによりfloat -> str -> Decimalの変換で精度を最大限保持
            decimal_str = repr(value)
            return Decimal(decimal_str)

        else:
            # その他の型は拒否
            raise TypeError(f"サポートされていない型: {type(value).__name__}")

    except (InvalidOperation, ValueError, TypeError) as e:
        # エラーメッセージに敏感な情報が含まれないようマスキング
        safe_value_str = mask_sensitive_info(str(value))
        raise ValueError(
            f"{context}の変換に失敗しました: {safe_value_str} ({type(value).__name__}) - {str(e)}"
        )


def quantize_decimal(value: Decimal, decimal_places: int = 2) -> Decimal:
    """
    Decimalの精度を統一（金銭計算用）

    Args:
        value: 対象のDecimal値
        decimal_places: 小数点以下の桁数

    Returns:
        Decimal: 精度調整された値
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"Decimal型が必要です: {type(value)}")

    # 量子化パターンを生成
    if decimal_places < 0:
        raise ValueError(
            f"小数点以下の桁数は0以上である必要があります: {decimal_places}"
        )

    quantum = Decimal("0.1") ** decimal_places
    return value.quantize(quantum)


def validate_positive_decimal(
    value: Decimal, context: str = "値", allow_zero: bool = False
) -> Decimal:
    """
    正のDecimal値の検証

    Args:
        value: 検証する値
        context: エラー時のコンテキスト情報
        allow_zero: ゼロを許可するか

    Returns:
        Decimal: 検証済みの値

    Raises:
        ValueError: 負数またはゼロ（allow_zero=Falseの場合）
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"{context}はDecimal型である必要があります: {type(value)}")

    if allow_zero:
        if value < 0:
            raise ValueError(f"{context}は0以上である必要があります: {value}")
    else:
        if value <= 0:
            raise ValueError(f"{context}は正数である必要があります: {value}")

    return value


def validate_file_path(filepath: str, operation: str = "ファイル操作") -> Path:
    """
    安全なファイルパス検証（パストラバーサル対策・強化版）

    Args:
        filepath: 検証するファイルパス
        operation: 操作種別（エラーメッセージ用）

    Returns:
        Path: 検証済みのPathオブジェクト

    Raises:
        ValueError: 不正なパス
        SecurityError: セキュリティ上問題のあるパス
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError(f"{operation}: 無効なファイルパス")

    try:
        path = Path(filepath)

        # パストラバーサル攻撃をチェック
        normalized_path = path.resolve()
        
        # 危険なパスパターンをチェック
        dangerous_patterns = ["../", "..\\", "/etc/", "/sys/", "/proc/", "C:\\Windows", "C:\\System"]
        path_str = str(normalized_path).lower()
        
        for pattern in dangerous_patterns:
            if pattern.lower() in path_str:
                raise ValueError(f"{operation}: 危険なパスパターンが検出されました")

        # ファイル名の基本検証
        if path.name:
            # 危険な文字をチェック
            dangerous_chars = ["<", ">", ":", "\"", "|", "?", "*"]
            for char in dangerous_chars:
                if char in path.name:
                    raise ValueError(f"{operation}: ファイル名に無効な文字が含まれています: {char}")

        return normalized_path

    except Exception as e:
        raise ValueError(f"{operation}: ファイルパス検証エラー - {str(e)}")


def mask_sensitive_info(text: str, mask_char: str = "*") -> str:
    """
    機密情報のマスキング（包括的セキュリティ強化版）

    金融取引データ、ファイルパス、価格情報、手数料等の
    機密性の高い情報を自動的にマスキングし、ログ出力時の
    情報漏洩を防止する。

    Args:
        text: マスキング対象のテキスト
        mask_char: マスク文字

    Returns:
        str: マスキング済みテキスト
    """
    if not text or not isinstance(text, str):
        return str(text) if text is not None else ""

    # 1. 金額・価格情報のマスキング
    # 価格パターン (例: "2500.00", "¥1,000", "$123.45")
    price_patterns = [
        r"[¥$€£]\s*[\d,]+\.?\d*",  # 通貨記号付き金額
        r'price["\']?\s*[:=]\s*[\d,]+\.?\d*',  # price: 1234.56
        r'amount["\']?\s*[:=]\s*[\d,]+\.?\d*',  # amount: 1234.56
        r'cost["\']?\s*[:=]\s*[\d,]+\.?\d*',  # cost: 1234.56
        r'value["\']?\s*[:=]\s*[\d,]+\.?\d*',  # value: 1234.56
        r'total["\']?\s*[:=]\s*[\d,]+\.?\d*',  # total: 1234.56
        r'balance["\']?\s*[:=]\s*[\d,]+\.?\d*',  # balance: 1234.56
    ]

    for pattern in price_patterns:
        text = re.sub(
            pattern,
            lambda m: _mask_financial_value(m.group(), mask_char),
            text,
            flags=re.IGNORECASE,
        )

    # 2. 手数料情報のマスキング
    commission_patterns = [
        r'commission["\']?\s*[:=]\s*[¥$€£]?[\d,]+\.?\d*',
        r'fee["\']?\s*[:=]\s*[¥$€£]?[\d,]+\.?\d*',
        r'費用["\']?\s*[:=]\s*[¥$€£]?[\d,]+\.?\d*',
        r'手数料["\']?\s*[:=]\s*[¥$€£]?[\d,]+\.?\d*',
    ]

    for pattern in commission_patterns:
        backslash = "\\"
        pattern_key = pattern.split("[")[0].split(backslash)[0]
        text = re.sub(
            pattern, f"{pattern_key}: {mask_char * 6}", text, flags=re.IGNORECASE
        )

    # 3. ファイルパスのマスキング
    # Windows/Unix パス情報
    path_patterns = [
        r"[C-Z]:[\\\/][\w\\\/.\\-_\s]+",  # C:\path\to\file
        r"\/[\w\/\.\-_\s]+",  # /path/to/file
        r"\.[\w\/\\\.\-_\s]+",  # ./relative/path
        r"~[\w\/\\\.\-_\s]*",  # ~/home/path
    ]

    for pattern in path_patterns:
        text = re.sub(pattern, lambda m: _mask_file_path(m.group(), mask_char), text)

    # 4. 取引ID・識別子のマスキング
    # 取引IDパターン
    id_patterns = [
        r'trade_id["\']?\s*[:=]\s*[A-Za-z0-9\-_]+',
        r'transaction_id["\']?\s*[:=]\s*[A-Za-z0-9\-_]+',
        r'order_id["\']?\s*[:=]\s*[A-Za-z0-9\-_]+',
        r'id["\']?\s*[:=]\s*[A-Za-z0-9\-_]{8,}',  # 長いID
    ]

    for pattern in id_patterns:
        text = re.sub(
            pattern,
            lambda m: _mask_transaction_id(m.group(), mask_char),
            text,
            flags=re.IGNORECASE,
        )

    # 5. 数量・ロット情報の部分マスキング
    quantity_patterns = [
        r'quantity["\']?\s*[:=]\s*[\d,]+',
        r'shares["\']?\s*[:=]\s*[\d,]+',
        r'volume["\']?\s*[:=]\s*[\d,]+',
        r'数量["\']?\s*[:=]\s*[\d,]+',
        r'株数["\']?\s*[:=]\s*[\d,]+',
    ]

    for pattern in quantity_patterns:
        text = re.sub(
            pattern,
            lambda m: _mask_quantity_value(m.group(), mask_char),
            text,
            flags=re.IGNORECASE,
        )

    # 6. 個人情報・機密データのマスキング
    sensitive_patterns = [
        r'api_key["\']?\s*[:=]\s*[A-Za-z0-9]+',
        r'secret["\']?\s*[:=]\s*[A-Za-z0-9]+',
        r'token["\']?\s*[:=]\s*[A-Za-z0-9]+',
        r'password["\']?\s*[:=]\s*\S+',
        r'user_id["\']?\s*[:=]\s*\S+',
    ]

    for pattern in sensitive_patterns:
        text = re.sub(
            pattern,
            lambda m: f"{m.group().split(':', 1)[0].split('=', 1)[0]}: {mask_char * 8}",
            text,
            flags=re.IGNORECASE,
        )

    # 7. IPアドレス・ネットワーク情報のマスキング
    network_patterns = [
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IPv4
        r"localhost:\d+",
        r"127\.0\.0\.1:\d+",
    ]

    for pattern in network_patterns:
        text = re.sub(pattern, f"IP_{mask_char * 4}", text)

    return text


def _mask_financial_value(match_str: str, mask_char: str = "*") -> str:
    """金融価格情報の部分マスキング"""
    # キーバリュー形式の場合は値部分のみマスク
    if ":" in match_str or "=" in match_str:
        key_part, value_part = re.split(r"[:=]\s*", match_str, 1)
        masked_value = _mask_number(value_part, mask_char)
        return f"{key_part}: {masked_value}"
    else:
        # 通貨記号付き金額の場合は数値部分をマスク
        return _mask_number(match_str, mask_char)


def _mask_file_path(path_str: str, mask_char: str = "*") -> str:
    """ファイルパス情報のマスキング"""
    # ファイル名のみ表示、ディレクトリ部分をマスク
    try:
        filename = os.path.basename(path_str)
        if len(filename) <= 3:
            return mask_char * len(path_str)

        # ファイル名の先頭2文字と拡張子のみ残す
        name, ext = os.path.splitext(filename)
        if len(name) <= 2:
            masked_name = name
        else:
            masked_name = name[:2] + mask_char * (len(name) - 2)

        return f"{mask_char * 8}/{masked_name}{ext}"
    except:
        return mask_char * min(len(path_str), 12)


def _mask_transaction_id(id_str: str, mask_char: str = "*") -> str:
    """取引ID・識別子のマスキング"""
    # キーバリュー形式の場合
    if ":" in id_str or "=" in id_str:
        key_part, value_part = re.split(r"[:=]\s*", id_str, 1)
        if len(value_part) <= 4:
            masked_value = mask_char * len(value_part)
        else:
            # 先頭2文字と末尾2文字を残す
            masked_value = (
                value_part[:2] + mask_char * (len(value_part) - 4) + value_part[-2:]
            )
        return f"{key_part}: {masked_value}"
    else:
        return _mask_number(id_str, mask_char)


def _mask_quantity_value(quantity_str: str, mask_char: str = "*") -> str:
    """数量情報の部分マスキング"""
    # キーバリュー形式の場合
    if ":" in quantity_str or "=" in quantity_str:
        key_part, value_part = re.split(r"[:=]\s*", quantity_str, 1)
        # 数量は桁数のみ隠す（実際の値は部分表示）
        masked_value = _mask_number(value_part, mask_char)
        return f"{key_part}: {masked_value}"
    else:
        return _mask_number(quantity_str, mask_char)


def _mask_number(number_str: str, mask_char: str = "*") -> str:
    """数値の部分マスキング（汎用）"""
    # 数値以外の文字（通貨記号、カンマなど）を保持
    # 数字のみを抽出
    digits = re.findall(r"\d", number_str)
    if not digits:
        return mask_char * len(number_str)

    # 先頭と末尾を残して中間をマスク
    if len(digits) <= 2:
        masked_digits = mask_char * len(digits)
    elif len(digits) <= 4:
        masked_digits = digits[0] + mask_char * (len(digits) - 2) + digits[-1]
    else:
        masked_digits = (
            "".join(digits[:2]) + mask_char * (len(digits) - 4) + "".join(digits[-2:])
        )

    # 元の形式を保持しながら数字を置換
    result = number_str
    digit_index = 0
    masked_result = ""

    for char in number_str:
        if char.isdigit():
            if digit_index < len(masked_digits):
                masked_result += masked_digits[digit_index]
                digit_index += 1
            else:
                masked_result += mask_char
        else:
            masked_result += char

    return masked_result