"""
Trade Manager Utilities
"""

import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Union
import re

def mask_sensitive_info(value: str) -> str:
    """機密情報をマスク化"""
    if not value:
        return ""
    # 値の最初と最後だけ表示し、中間をマスク
    if len(value) <= 3:
        return "*" * len(value)
    return value[0] + "*" * (len(value) - 2) + value[-1]


def safe_decimal_conversion(
    value: Union[str, int, float, Decimal], context: str = "値", default_value: Decimal = None
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
        # デフォルト値が指定されている場合はそれを返す
        if default_value is not None:
            return default_value

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
        Path: 正規化された安全なパス

    Raises:
        ValueError: 不正なパス
        SecurityError: セキュリティ上の問題
    """
    if not filepath:
        raise ValueError(f"{operation}にはファイルパスが必要です")

    if not isinstance(filepath, (str, Path)):
        raise TypeError(
            f"ファイルパスは文字列またはPathオブジェクトである必要があります: {type(filepath)}"
        )

    try:
        # 1. 基本的なセキュリティチェック
        filepath_str = str(filepath)

        # 危険なパターンを事前チェック
        dangerous_patterns = [
            "../",
            "..\\\\",
            "%2e%2e",
            "~/",
            "file://",
            "ftp://",
            "http://",
            "https://",
            "\\\\",
            "//",
            "..%2f",
            "..%5c",
            "\x00",
            "\x01",
            "\x02",
            "\x03",
            "\x04",  # NULL文字・制御文字
        ]

        filepath_lower = filepath_str.lower()
        for pattern in dangerous_patterns:
            if pattern in filepath_lower:
                raise ValueError(f"危険なパスパターンが検出されました: {pattern}")

        # 2. パス長制限
        if len(filepath_str) > 260:  # Windows MAX_PATH制限
            raise ValueError(f"ファイルパスが長すぎます: {len(filepath_str)} 文字")

        # 3. パスを正規化
        path = Path(filepath_str).resolve()

        # 4. 現在のディレクトリを基準とした検証
        current_dir = Path.cwd().resolve()

        # 許可されたベースディレクトリ
        allowed_base_dirs = [
            current_dir,
            current_dir / "data",
            current_dir / "output",
            current_dir / "exports",
            current_dir / "temp",
        ]

        # 絶対パスの場合は追加検証
        is_within_allowed = False
        try:
            for allowed_dir in allowed_base_dirs:
                try:
                    path.relative_to(allowed_dir)
                    is_within_allowed = True
                    break
                except ValueError:
                    continue
        except Exception:
            pass

        if not is_within_allowed:
            # 許可されたディレクトリ外の場合、システムディレクトリをチェック
            forbidden_paths = [
                # Unix/Linux系システムディレクトリ
                "/etc",
                "/usr",
                "/var",
                "/root",
                "/home",
                "/opt",
                "/sys",
                "/proc",
                "/dev",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
                "/boot",
                "/mnt",
                "/media",
                # Windows系システムディレクトリ
                "c:\\windows",
                "c:\\program files",
                "c:\\program files (x86)",
                "c:\\users\\default",
                "c:\\users\\public",
                "c:\\programdata",
                "c:\\system volume information",
                "c:\\recovery",
                # ネットワークパス
                "\\\\",
                "//",
                "smb://",
                "nfs://",
            ]

            path_str = str(path).lower()
            for forbidden in forbidden_paths:
                if path_str.startswith(forbidden.lower()):
                    raise ValueError(
                        f"システムディレクトリへのアクセスは禁止されています: {mask_sensitive_info(str(path))}"
                    )

        # 5. ファイル名の検証
        if path.name:
            # 危険な文字
            dangerous_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
            if any(char in path.name for char in dangerous_chars):
                raise ValueError(
                    f"ファイル名に無効な文字が含まれています: {mask_sensitive_info(path.name)}"
                )

            # 予約語チェック（Windows）
            reserved_names = [
                "con",
                "prn",
                "aux",
                "nul",
                "com1",
                "com2",
                "com3",
                "com4",
                "com5",
                "com6",
                "com7",
                "com8",
                "com9",
                "lpt1",
                "lpt2",
                "lpt3",
                "lpt4",
                "lpt5",
                "lpt6",
                "lpt7",
                "lpt8",
                "lpt9",
            ]
            if path.stem.lower() in reserved_names:
                raise ValueError(f"予約語をファイル名に使用できません: {path.stem}")

        # 6. ディレクトリ存在確認（親ディレクトリ）
        if not path.parent.exists():
            # 親ディレクトリが存在しない場合は作成を試みる（安全な範囲で）
            if is_within_allowed:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # logger.info(
                    #     f"ディレクトリを作成しました: {mask_sensitive_info(str(path.parent))}"
                    # )
                except PermissionError:
                    raise ValueError(
                        f"ディレクトリの作成権限がありません: {mask_sensitive_info(str(path.parent))}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"ディレクトリ作成に失敗しました: {mask_sensitive_info(str(e))}"
                    )
            else:
                raise ValueError(
                    f"親ディレクトリが存在しません: {mask_sensitive_info(str(path.parent))}"
                )

        return path

    except Exception as e:
        if isinstance(e, (ValueError, TypeError)):
            raise
        safe_error_msg = mask_sensitive_info(str(e))
        raise ValueError(
            f"{operation}のパス処理でエラーが発生しました: {safe_error_msg}"
        )


def _mask_financial_value(match_str: str, mask_char: str = "*") -> str:
    """金融価格情報の部分マスキング"""
    import re

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
    import re

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
    import re

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
    import re

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