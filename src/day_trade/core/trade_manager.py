"""
分析・シミュレーション記録管理機能
シミュレーション売買履歴を記録し、仮想損益計算を行う
※実際の取引は行わず、分析・学習用のみ
データベース永続化対応版
"""

import csv
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

from ..models.database import db_manager
from ..models.enums import TradeType
from ..models.stock import Stock
from ..models.stock import Trade as DBTrade
from ..utils.enhanced_error_handler import (
    get_default_error_handler,
)
from ..utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


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
            import re

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
            "..\\",
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
                    logger.info(
                        f"ディレクトリを作成しました: {mask_sensitive_info(str(path.parent))}"
                    )
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

    import re

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


class TradeStatus(Enum):
    """取引ステータス"""

    PENDING = "pending"  # 注文中
    EXECUTED = "executed"  # 約定済み
    CANCELLED = "cancelled"  # キャンセル
    PARTIAL = "partial"  # 一部約定


@dataclass
class Trade:
    """取引記録"""

    id: str
    symbol: str
    trade_type: TradeType
    quantity: int
    price: Decimal
    timestamp: datetime
    commission: Decimal = Decimal("0")
    status: TradeStatus = TradeStatus.EXECUTED
    notes: str = ""

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        data = asdict(self)
        data["trade_type"] = self.trade_type.value
        data["status"] = self.status.value
        data["price"] = str(self.price)
        data["commission"] = str(self.commission)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "Trade":
        """辞書から復元（安全なDecimal変換）"""
        try:
            # 必須フィールドの検証
            required_fields = [
                "id",
                "symbol",
                "trade_type",
                "quantity",
                "price",
                "timestamp",
            ]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"必須フィールド '{field}' が不足しています")

            # 安全なDecimal変換
            price = safe_decimal_conversion(data["price"], "価格")
            commission = safe_decimal_conversion(data.get("commission", "0"), "手数料")

            # 正数検証
            validate_positive_decimal(price, "価格")
            validate_positive_decimal(commission, "手数料", allow_zero=True)

            # 数量の検証
            quantity = int(data["quantity"])
            if quantity <= 0:
                raise ValueError(f"数量は正数である必要があります: {quantity}")

            return cls(
                id=str(data["id"]),
                symbol=str(data["symbol"]),
                trade_type=TradeType(data["trade_type"]),
                quantity=quantity,
                price=price,
                timestamp=datetime.fromisoformat(data["timestamp"]),
                commission=commission,
                status=TradeStatus(data.get("status", TradeStatus.EXECUTED.value)),
                notes=str(data.get("notes", "")),
            )
        except Exception as e:
            raise ValueError(f"取引データの復元に失敗しました: {str(e)}")


@dataclass
class BuyLot:
    """
    FIFO会計のための買いロット情報
    個別の買い取引を管理し、正確な売却対応を可能にする
    """

    quantity: int
    price: Decimal
    commission: Decimal
    timestamp: datetime
    trade_id: str

    def total_cost_per_share(self) -> Decimal:
        """1株あたりの総コスト（買い価格 + 手数料按分）"""
        if self.quantity == 0:
            return Decimal("0")
        return self.price + (self.commission / Decimal(self.quantity))


@dataclass
class Position:
    """
    ポジション情報（FIFO会計対応強化版）

    買いロットキューを使用してFIFO原則を厳密に適用
    """

    symbol: str
    quantity: int
    average_price: Decimal
    total_cost: Decimal
    current_price: Decimal = Decimal("0")

    # FIFO会計のための買いロットキュー
    buy_lots: Deque[BuyLot] = None

    def __post_init__(self):
        """初期化後の処理"""
        if self.buy_lots is None:
            self.buy_lots = deque()

    @property
    def market_value(self) -> Decimal:
        """時価総額"""
        return self.current_price * Decimal(self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        """含み損益"""
        return self.market_value - self.total_cost

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """含み損益率"""
        if self.total_cost == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.total_cost) * 100

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": str(self.average_price),
            "total_cost": str(self.total_cost),
            "current_price": str(self.current_price),
            "market_value": str(self.market_value),
            "unrealized_pnl": str(self.unrealized_pnl),
            "unrealized_pnl_percent": str(
                self.unrealized_pnl_percent.quantize(Decimal("0.01"))
            ),
        }


@dataclass
class RealizedPnL:
    """実現損益"""

    symbol: str
    quantity: int
    buy_price: Decimal
    sell_price: Decimal
    buy_commission: Decimal
    sell_commission: Decimal
    pnl: Decimal
    pnl_percent: Decimal
    buy_date: datetime
    sell_date: datetime

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "buy_price": str(self.buy_price),
            "sell_price": str(self.sell_price),
            "buy_commission": str(self.buy_commission),
            "sell_commission": str(self.sell_commission),
            "pnl": str(self.pnl),
            "pnl_percent": str(self.pnl_percent.quantize(Decimal("0.01"))),
            "buy_date": self.buy_date.isoformat(),
            "sell_date": self.sell_date.isoformat(),
        }


class TradeManager:
    """
    取引記録管理クラス（会計原則対応版）

    会計原則:
    - FIFO (First In, First Out): 先入れ先出し法を採用
    - 実現損益計算では最古の買い取引から順次売却するものとして計算
    - 手数料は取引毎に個別に管理、実現损益に反映
    - 税金は利益が出た場合のみ計算
    """

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
        load_from_db: bool = False,
    ):
        """
        初期化

        Args:
            commission_rate: 手数料率（デフォルト0.1%）
            tax_rate: 税率（デフォルト20%）
            load_from_db: データベースから取引履歴を読み込むかどうか
        """
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: List[RealizedPnL] = []
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self._trade_counter = 0

        # ロガーを初期化
        self.logger = get_context_logger(__name__)

        if load_from_db:
            self._load_trades_from_db()

    def _generate_trade_id(self) -> str:
        """取引IDを生成"""
        self._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self._trade_counter:04d}"

    def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算"""
        total_value = price * Decimal(quantity)
        commission = total_value * self.commission_rate
        # 最低100円の手数料
        return max(commission, Decimal("100"))

    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> str:
        """
        取引を追加（データベース永続化対応）

        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            timestamp: 取引日時
            commission: 手数料（Noneの場合は自動計算）
            notes: メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引ID
        """
        try:
            # 安全なDecimal変換
            safe_price = safe_decimal_conversion(price, "取引価格")
            validate_positive_decimal(safe_price, "取引価格")

            # 数量検証
            if not isinstance(quantity, int) or quantity <= 0:
                raise ValueError(f"数量は正の整数である必要があります: {quantity}")

            # コンテキスト情報をログに含める（価格情報はマスク）
            context_info = {
                "operation": "add_trade",
                "symbol": symbol,
                "trade_type": trade_type.value,
                "quantity": quantity,
                "price_masked": mask_sensitive_info(str(safe_price)),
                "persist_to_db": persist_to_db,
            }

            logger.info("取引追加処理開始", extra=context_info)

            if timestamp is None:
                timestamp = datetime.now()

            if commission is None:
                safe_commission = self._calculate_commission(safe_price, quantity)
            else:
                safe_commission = safe_decimal_conversion(commission, "手数料")
                validate_positive_decimal(safe_commission, "手数料", allow_zero=True)

            trade_id = self._generate_trade_id()
            # メモリ内データ構造のトレード
            memory_trade = Trade(
                id=trade_id,
                symbol=symbol,
                trade_type=trade_type,
                quantity=quantity,
                price=safe_price,
                timestamp=timestamp,
                commission=safe_commission,
                notes=notes,
            )

            if persist_to_db:
                # データベース永続化（トランザクション保護）
                with db_manager.transaction_scope() as session:
                    # 1. 銘柄マスタの存在確認・作成
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        logger.info("銘柄マスタに未登録、新規作成", extra=context_info)
                        stock = Stock(
                            code=symbol,
                            name=symbol,  # 名前が不明な場合はコードを使用
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()  # IDを確定

                    # 2. データベース取引記録を作成
                    db_trade = (
                        DBTrade.create_buy_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=safe_price,
                            commission=safe_commission,
                            memo=notes,
                        )
                        if trade_type == TradeType.BUY
                        else DBTrade.create_sell_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=safe_price,
                            commission=safe_commission,
                            memo=notes,
                        )
                    )

                    # 3. メモリ内データ構造を更新
                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)

                    # 中間状態をflushして整合性を確認
                    session.flush()

                    # ビジネスイベントログ（機密情報マスキング適用）
                    log_business_event(
                        "trade_added",
                        trade_id=mask_sensitive_info(str(trade_id)),
                        symbol=symbol,
                        trade_type=trade_type.value,
                        quantity=mask_sensitive_info(f"quantity: {quantity}"),
                        price=mask_sensitive_info(f"price: {str(price)}"),
                        commission=mask_sensitive_info(
                            f"commission: {str(commission)}"
                        ),
                        persisted=True,
                    )

                    logger.info(
                        "取引追加完了（DB永続化）",
                        extra={
                            **context_info,
                            "trade_id": mask_sensitive_info(str(trade_id)),
                            "db_trade_id": mask_sensitive_info(str(db_trade.id)),
                        },
                    )
            else:
                # メモリ内のみの処理（後方互換性）
                self.trades.append(memory_trade)
                self._update_position(memory_trade)

                log_business_event(
                    "trade_added",
                    trade_id=mask_sensitive_info(str(trade_id)),
                    symbol=symbol,
                    trade_type=trade_type.value,
                    quantity=mask_sensitive_info(f"quantity: {quantity}"),
                    price=mask_sensitive_info(f"price: {str(price)}"),
                    commission=mask_sensitive_info(f"commission: {str(commission)}"),
                    persisted=False,
                )

                logger.info(
                    "取引追加完了（メモリのみ）",
                    extra={
                        **context_info,
                        "trade_id": mask_sensitive_info(str(trade_id)),
                    },
                )

            return trade_id

        except Exception as e:
            logger.error(f"取引追加エラー: {mask_sensitive_info(str(e))}")
            log_error_with_context(
                e,
                {
                    "operation": "add_trade",
                    "symbol": symbol,
                    "trade_type": trade_type.value,
                    "quantity": quantity,
                    "price_masked": mask_sensitive_info(str(price)),
                    "persist_to_db": persist_to_db,
                },
            )
            raise

    def _update_position(self, trade: Trade) -> None:
        """
        ポジション更新（FIFO会計原則・強化版）

        買いロットキューを使用して厳密なFIFO管理を実現
        """
        symbol = trade.symbol

        if trade.trade_type == TradeType.BUY:
            if symbol in self.positions:
                # 既存ポジションに買いロットを追加
                position = self.positions[symbol]

                # 新しい買いロットを作成
                new_lot = BuyLot(
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=trade.commission,
                    timestamp=trade.timestamp,
                    trade_id=trade.id,
                )

                # 買いロットキューに追加（FIFO）
                position.buy_lots.append(new_lot)

                # ポジション全体を更新
                total_cost = (
                    position.total_cost
                    + (trade.price * Decimal(trade.quantity))
                    + trade.commission
                )
                total_quantity = position.quantity + trade.quantity
                average_price = total_cost / Decimal(total_quantity)

                position.quantity = total_quantity
                position.average_price = average_price
                position.total_cost = total_cost
            else:
                # 新規ポジション
                total_cost = (trade.price * Decimal(trade.quantity)) + trade.commission

                # 初期買いロット作成
                initial_lot = BuyLot(
                    quantity=trade.quantity,
                    price=trade.price,
                    commission=trade.commission,
                    timestamp=trade.timestamp,
                    trade_id=trade.id,
                )

                new_position = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    average_price=total_cost / Decimal(trade.quantity),
                    total_cost=total_cost,
                )
                new_position.buy_lots = deque([initial_lot])

                self.positions[symbol] = new_position

        elif trade.trade_type == TradeType.SELL:
            if symbol in self.positions:
                position = self.positions[symbol]

                if position.quantity >= trade.quantity:
                    # 厳密なFIFO会計による実現損益計算
                    self._process_sell_fifo(position, trade)
                else:
                    logger.warning(
                        f"銘柄 '{symbol}' の売却数量が保有数量 ({position.quantity}) を超過しています。売却数量: {trade.quantity}。取引は処理されません。",
                        extra={
                            "symbol": symbol,
                            "available_quantity": position.quantity,
                            "requested_quantity": trade.quantity,
                        },
                    )
            else:
                logger.warning(
                    f"ポジションを保有していない銘柄 '{symbol}' の売却を試みました。取引は無視されます。",
                    extra={"symbol": symbol, "trade_type": "SELL"},
                )

    def _process_sell_fifo(self, position: Position, sell_trade: Trade) -> None:
        """
        FIFO原則による売却処理（厳密版）

        買いロットキューから順次売却し、正確な実現損益を計算
        """
        remaining_sell_quantity = sell_trade.quantity
        sell_price = sell_trade.price
        sell_commission = sell_trade.commission
        symbol = sell_trade.symbol

        # 売却処理による実現損益を累積
        total_buy_cost = Decimal("0")
        total_buy_commission = Decimal("0")
        total_sold_quantity = 0
        earliest_buy_date = None

        while remaining_sell_quantity > 0 and position.buy_lots:
            # 最古のロット（FIFO）を取得
            oldest_lot = position.buy_lots.popleft()

            if earliest_buy_date is None:
                earliest_buy_date = oldest_lot.timestamp

            # 売却数量の決定
            quantity_to_sell = min(remaining_sell_quantity, oldest_lot.quantity)

            # 売却分のコスト按分
            lot_cost_per_share = oldest_lot.total_cost_per_share()
            buy_cost_for_this_sale = lot_cost_per_share * Decimal(quantity_to_sell)
            buy_commission_for_this_sale = (
                oldest_lot.commission
                * Decimal(quantity_to_sell)
                / Decimal(oldest_lot.quantity)
            )

            total_buy_cost += buy_cost_for_this_sale
            total_buy_commission += buy_commission_for_this_sale
            total_sold_quantity += quantity_to_sell

            # ロットを部分的に消費
            if oldest_lot.quantity > quantity_to_sell:
                # ロットの残りを戻す
                remaining_lot = BuyLot(
                    quantity=oldest_lot.quantity - quantity_to_sell,
                    price=oldest_lot.price,
                    commission=oldest_lot.commission
                    * Decimal(oldest_lot.quantity - quantity_to_sell)
                    / Decimal(oldest_lot.quantity),
                    timestamp=oldest_lot.timestamp,
                    trade_id=oldest_lot.trade_id,
                )
                position.buy_lots.appendleft(remaining_lot)

            remaining_sell_quantity -= quantity_to_sell

        if total_sold_quantity > 0:
            # 実現損益計算
            gross_proceeds = sell_price * Decimal(total_sold_quantity) - sell_commission
            cost_basis = total_buy_cost
            pnl_before_tax = gross_proceeds - cost_basis

            # 税金計算（利益が出た場合のみ）
            tax = Decimal("0")
            if pnl_before_tax > 0:
                tax = pnl_before_tax * self.tax_rate

            # 最終的な実現損益（税引き後）
            pnl = pnl_before_tax - tax

            # 収益率計算
            pnl_percent = (pnl / cost_basis * 100) if cost_basis > 0 else Decimal("0")

            # 平均買い価格の計算
            average_buy_price = (
                cost_basis / Decimal(total_sold_quantity)
                if total_sold_quantity > 0
                else Decimal("0")
            )

            # 実現損益を記録
            realized_pnl = RealizedPnL(
                symbol=symbol,
                quantity=total_sold_quantity,
                buy_price=average_buy_price,
                sell_price=sell_price,
                buy_commission=total_buy_commission,
                sell_commission=sell_commission,
                pnl=pnl,
                pnl_percent=pnl_percent,
                buy_date=earliest_buy_date or sell_trade.timestamp,
                sell_date=sell_trade.timestamp,
            )

            self.realized_pnl.append(realized_pnl)

            # ポジション情報を更新
            position.quantity -= total_sold_quantity
            position.total_cost -= cost_basis

            if position.quantity > 0:
                # 平均価格の再計算
                position.average_price = position.total_cost / Decimal(
                    position.quantity
                )
            else:
                # ポジション完全クローズ
                del self.positions[symbol]

    def _get_earliest_buy_date(self, symbol: str) -> datetime:
        """
        最も古い買い取引の日付を取得（FIFO会計原則・最適化版）

        買いロットキューから直接取得することで効率化
        """
        if symbol not in self.positions:
            logger.warning(f"銘柄 {symbol} のポジションが見つかりません")
            return datetime.now()

        position = self.positions[symbol]

        # 買いロットキューが空でない場合、最古のロット（先頭）の日付を返す
        if position.buy_lots:
            earliest_date = position.buy_lots[0].timestamp
            logger.debug(
                f"銘柄 {symbol} の最早買い取引日（ロットキューから取得）: {earliest_date}"
            )
            return earliest_date

        # フォールバック: 全取引から検索（互換性維持）
        buy_trades = [
            trade
            for trade in self.trades
            if trade.symbol == symbol and trade.trade_type == TradeType.BUY
        ]

        if not buy_trades:
            logger.warning(f"銘柄 {symbol} の買い取引が見つかりません")
            return datetime.now()

        # 最早の取引を効率的に検索
        earliest_date = min(trade.timestamp for trade in buy_trades)
        logger.debug(
            f"銘柄 {symbol} の最早買い取引日（フォールバック検索）: {earliest_date}"
        )
        return earliest_date

    def _load_trades_from_db(self) -> None:
        """データベースから取引履歴を読み込み（トランザクション保護版）"""
        load_logger = self.logger
        load_logger.info("データベースから取引履歴読み込み開始")

        try:
            # トランザクション内で一括処理
            with db_manager.transaction_scope() as session:
                # データベースから全取引を取得
                db_trades = (
                    session.query(DBTrade).order_by(DBTrade.trade_datetime).all()
                )

                load_logger.info("DB取引データ取得", extra={"count": len(db_trades)})

                # メモリ内データ構造を一旦クリア（原子性保証）
                trades_backup = self.trades.copy()
                positions_backup = self.positions.copy()
                realized_pnl_backup = self.realized_pnl.copy()
                counter_backup = self._trade_counter

                try:
                    # メモリ内データクリア
                    self.trades.clear()
                    self.positions.clear()
                    self.realized_pnl.clear()
                    self._trade_counter = 0

                    for db_trade in db_trades:
                        # セッションから切り離す前に必要な属性を読み込み
                        trade_id = db_trade.id
                        stock_code = db_trade.stock_code
                        trade_type_str = db_trade.trade_type
                        quantity = db_trade.quantity
                        price = db_trade.price
                        trade_datetime = db_trade.trade_datetime
                        commission = db_trade.commission or Decimal("0")
                        memo = db_trade.memo or ""

                        # メモリ内形式に変換
                        if isinstance(trade_type_str, TradeType):
                            trade_type = trade_type_str
                        else:
                            trade_type_str_upper = str(trade_type_str).upper()
                            trade_type = (
                                TradeType.BUY
                                if trade_type_str_upper in ["BUY", "buy"]
                                else TradeType.SELL
                            )

                        memory_trade = Trade(
                            id=f"DB_{trade_id}",  # DBから読み込んだことを示すプレフィックス
                            symbol=stock_code,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=Decimal(str(price)),
                            timestamp=trade_datetime,
                            commission=Decimal(str(commission)),
                            status=TradeStatus.EXECUTED,
                            notes=memo,
                        )

                        self.trades.append(memory_trade)
                        self._update_position(memory_trade)

                    # 取引カウンターを最大値+1に設定
                    if db_trades:
                        max_id = max(db_trade.id for db_trade in db_trades)
                        self._trade_counter = max_id + 1

                    load_logger.info(
                        "データベース読み込み完了",
                        extra={"loaded_trades": len(db_trades)},
                        trade_counter=self._trade_counter,
                    )

                except Exception as restore_error:
                    # メモリ内データの復元
                    self.trades = trades_backup
                    self.positions = positions_backup
                    self.realized_pnl = realized_pnl_backup
                    self._trade_counter = counter_backup
                    load_logger.error(
                        "読み込み処理失敗、メモリ内データを復元",
                        extra={"error": str(restore_error)},
                    )
                    raise restore_error

        except Exception as e:
            log_error_with_context(e, {"operation": "load_trades_from_db"})
            load_logger.error("データベース読み込み失敗", extra={"error": str(e)})
            raise

    def sync_with_db(self) -> None:
        """データベースとの同期を実行（原子性保証版）"""
        sync_logger = self.logger
        sync_logger.info("データベース同期開始")

        # 現在のメモリ内データをバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            # 現在のメモリ内データをクリア
            self.trades.clear()
            self.positions.clear()
            self.realized_pnl.clear()
            self._trade_counter = 0

            # データベースから再読み込み（トランザクション保護済み）
            self._load_trades_from_db()

            sync_logger.info("データベース同期完了")

        except Exception as e:
            # エラー時にはバックアップデータを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(e, {"operation": "sync_with_db"})
            sync_logger.error(
                "データベース同期失敗、メモリ内データを復元", extra={"error": str(e)}
            )
            raise

    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        """
        複数の取引を一括追加（トランザクション保護）

        Args:
            trades_data: 取引データのリスト
                [{"symbol": "7203", "trade_type": TradeType.BUY, "quantity": 100, "price": Decimal("2500"), ...}, ...]
            persist_to_db: データベースに永続化するかどうか

        Returns:
            作成された取引IDのリスト

        Raises:
            Exception: いずれかの取引処理でエラーが発生した場合、すべての処理がロールバック
        """
        batch_logger = logger.bind(
            operation="add_trades_batch",
            batch_size=len(trades_data),
            persist_to_db=persist_to_db,
        )
        batch_logger.info("一括取引追加処理開始")

        if not trades_data:
            batch_logger.warning("空の取引データが渡されました")
            return []

        trade_ids = []

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    for i, trade_data in enumerate(trades_data):
                        try:
                            # 取引データの検証と補完
                            symbol = trade_data["symbol"]
                            trade_type = trade_data["trade_type"]
                            quantity = trade_data["quantity"]
                            price = trade_data["price"]
                            timestamp = trade_data.get("timestamp", datetime.now())
                            commission = trade_data.get("commission")
                            notes = trade_data.get("notes", "")

                            if commission is None:
                                commission = self._calculate_commission(price, quantity)

                            trade_id = self._generate_trade_id()

                            # 1. 銘柄マスタの存在確認・作成
                            stock = (
                                session.query(Stock)
                                .filter(Stock.code == symbol)
                                .first()
                            )
                            if not stock:
                                stock = Stock(
                                    code=symbol,
                                    name=symbol,
                                    market="未定",
                                    sector="未定",
                                    industry="未定",
                                )
                                session.add(stock)
                                session.flush()

                            # 2. データベース取引記録を作成
                            (
                                DBTrade.create_buy_trade(
                                    session=session,
                                    stock_code=symbol,
                                    quantity=quantity,
                                    price=price,
                                    commission=commission,
                                    memo=notes,
                                )
                                if trade_type == TradeType.BUY
                                else DBTrade.create_sell_trade(
                                    session=session,
                                    stock_code=symbol,
                                    quantity=quantity,
                                    price=price,
                                    commission=commission,
                                    memo=notes,
                                )
                            )

                            # 3. メモリ内データ構造を更新
                            memory_trade = Trade(
                                id=trade_id,
                                symbol=symbol,
                                trade_type=trade_type,
                                quantity=quantity,
                                price=price,
                                timestamp=timestamp,
                                commission=commission,
                                notes=notes,
                            )

                            self.trades.append(memory_trade)
                            self._update_position(memory_trade)
                            trade_ids.append(trade_id)

                            # バッチ内の中間状態をflush
                            if (i + 1) % 10 == 0:  # 10件ごとにflush
                                session.flush()

                        except Exception as trade_error:
                            batch_logger.error(
                                "個別取引処理失敗",
                                trade_index=i,
                                symbol=trade_data.get("symbol", "unknown"),
                                error=str(trade_error),
                            )
                            raise trade_error

                    # 最終的なビジネスイベントログ
                    log_business_event(
                        "trades_batch_added",
                        batch_size=len(trades_data),
                        trade_ids=[mask_sensitive_info(str(tid)) for tid in trade_ids],
                        persisted=True,
                    )

                    batch_logger.info(
                        "一括取引追加完了（DB永続化）",
                        extra={"trade_count": len(trade_ids)},
                    )

            else:
                # メモリ内のみの処理
                for trade_data in trades_data:
                    symbol = trade_data["symbol"]
                    trade_type = trade_data["trade_type"]
                    quantity = trade_data["quantity"]
                    price = trade_data["price"]
                    timestamp = trade_data.get("timestamp", datetime.now())
                    commission = trade_data.get("commission")
                    notes = trade_data.get("notes", "")

                    if commission is None:
                        commission = self._calculate_commission(price, quantity)

                    trade_id = self._generate_trade_id()

                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=trade_type,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )

                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)
                    trade_ids.append(trade_id)

                log_business_event(
                    "trades_batch_added",
                    batch_size=len(trades_data),
                    trade_ids=[mask_sensitive_info(str(tid)) for tid in trade_ids],
                    persisted=False,
                )

                batch_logger.info(
                    "一括取引追加完了（メモリのみ）",
                    extra={"trade_count": len(trade_ids)},
                )

            return trade_ids

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "add_trades_batch",
                    "batch_size": len(trades_data),
                    "persist_to_db": persist_to_db,
                    "completed_trades": len(trade_ids),
                },
            )
            batch_logger.error(
                "一括取引追加失敗、すべての変更をロールバック", extra={"error": str(e)}
            )
            raise

    def clear_all_data(self, persist_to_db: bool = True) -> None:
        """
        すべての取引データを削除（トランザクション保護）

        Args:
            persist_to_db: データベースからも削除するかどうか

        Warning:
            この操作は取引履歴、ポジション、実現損益をすべて削除します
        """
        clear_logger = logger.bind(
            operation="clear_all_data", persist_to_db=persist_to_db
        )
        clear_logger.warning("全データ削除処理開始")

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            if persist_to_db:
                # データベースとメモリ両方をクリア
                with db_manager.transaction_scope() as session:
                    # データベースの取引データを削除
                    deleted_count = session.query(DBTrade).delete()
                    clear_logger.info(
                        "データベース取引データ削除",
                        extra={"deleted_count": deleted_count},
                    )

                    # メモリ内データクリア
                    self.trades.clear()
                    self.positions.clear()
                    self.realized_pnl.clear()
                    self._trade_counter = 0

                    log_business_event(
                        "all_data_cleared",
                        deleted_db_records=deleted_count,
                        persisted=True,
                    )

                    clear_logger.warning("全データ削除完了（DB + メモリ）")
            else:
                # メモリ内のみクリア
                self.trades.clear()
                self.positions.clear()
                self.realized_pnl.clear()
                self._trade_counter = 0

                log_business_event(
                    "all_data_cleared", deleted_db_records=0, persisted=False
                )

                clear_logger.warning("全データ削除完了（メモリのみ）")

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e, {"operation": "clear_all_data", "persist_to_db": persist_to_db}
            )
            clear_logger.error(
                "全データ削除失敗、メモリ内データを復元", extra={"error": str(e)}
            )
            raise

    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.positions.copy()

    def update_current_prices(self, prices: Dict[str, Decimal]) -> None:
        """現在価格を更新"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """取引履歴を取得"""
        if symbol:
            return [trade for trade in self.trades if trade.symbol == symbol]
        return self.trades.copy()

    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        if symbol:
            return [pnl for pnl in self.realized_pnl if pnl.symbol == symbol]
        return self.realized_pnl.copy()

    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオサマリーを取得"""
        total_cost = sum(pos.total_cost for pos in self.positions.values())
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = total_market_value - total_cost
        total_realized_pnl = sum(pnl.pnl for pnl in self.realized_pnl)

        return {
            "total_positions": len(self.positions),
            "total_cost": str(total_cost),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_unrealized_pnl + total_realized_pnl),
            "total_trades": len(self.trades),
            "winning_trades": len([pnl for pnl in self.realized_pnl if pnl.pnl > 0]),
            "losing_trades": len([pnl for pnl in self.realized_pnl if pnl.pnl < 0]),
            "win_rate": (
                f"{(len([pnl for pnl in self.realized_pnl if pnl.pnl > 0]) / max(len(self.realized_pnl), 1) * 100):.1f}%"
                if self.realized_pnl
                else "0.0%"
            ),
        }

    def export_to_csv(self, filepath: str, data_type: str = "trades") -> None:
        """
        CSVファイルにエクスポート（パス検証強化版）

        Args:
            filepath: 出力ファイルパス
            data_type: データタイプ ('trades', 'positions', 'realized_pnl')
        """
        try:
            # ファイルパス検証
            safe_path = validate_file_path(filepath, "CSV出力")

            if data_type == "trades":
                data = [trade.to_dict() for trade in self.trades]
                fieldnames = [
                    "id",
                    "symbol",
                    "trade_type",
                    "quantity",
                    "price",
                    "timestamp",
                    "commission",
                    "status",
                    "notes",
                ]

            elif data_type == "positions":
                data = [pos.to_dict() for pos in self.positions.values()]
                fieldnames = [
                    "symbol",
                    "quantity",
                    "average_price",
                    "total_cost",
                    "current_price",
                    "market_value",
                    "unrealized_pnl",
                    "unrealized_pnl_percent",
                ]

            elif data_type == "realized_pnl":
                data = [pnl.to_dict() for pnl in self.realized_pnl]
                fieldnames = [
                    "symbol",
                    "quantity",
                    "buy_price",
                    "sell_price",
                    "buy_commission",
                    "sell_commission",
                    "pnl",
                    "pnl_percent",
                    "buy_date",
                    "sell_date",
                ]

            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            with open(safe_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            logger.info(
                f"CSV出力完了: {mask_sensitive_info(str(safe_path))} ({len(data)}件)"
            )

        except Exception as e:
            logger.error(
                f"データのエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise

    def save_to_json(self, filepath: str) -> None:
        """JSON形式で保存（パス検証強化版）"""
        try:
            # ファイルパス検証
            safe_path = validate_file_path(filepath, "JSON保存")

            data = {
                "trades": [trade.to_dict() for trade in self.trades],
                "positions": {
                    symbol: pos.to_dict() for symbol, pos in self.positions.items()
                },
                "realized_pnl": [pnl.to_dict() for pnl in self.realized_pnl],
                "settings": {
                    "commission_rate": str(self.commission_rate),
                    "tax_rate": str(self.tax_rate),
                    "trade_counter": self._trade_counter,
                },
            }

            with open(safe_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"JSON保存完了: {mask_sensitive_info(str(safe_path))}")

        except Exception as e:
            logger.error(
                f"データの保存中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise

    def load_from_json(self, filepath: str) -> None:
        """JSON形式から読み込み（パス検証強化版）"""
        try:
            # ファイルパス検証
            safe_path = validate_file_path(filepath, "JSON読み込み")

            with open(safe_path, encoding="utf-8") as f:
                data = json.load(f)

            # 取引履歴復元
            self.trades = [
                Trade.from_dict(trade_data) for trade_data in data.get("trades", [])
            ]

            # ポジション復元
            self.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.positions[symbol] = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    average_price=Decimal(pos_data["average_price"]),
                    total_cost=Decimal(pos_data["total_cost"]),
                    current_price=Decimal(pos_data["current_price"]),
                )

            # 実現損益復元
            self.realized_pnl = []
            for pnl_data in data.get("realized_pnl", []):
                self.realized_pnl.append(
                    RealizedPnL(
                        symbol=pnl_data["symbol"],
                        quantity=pnl_data["quantity"],
                        buy_price=Decimal(pnl_data["buy_price"]),
                        sell_price=Decimal(pnl_data["sell_price"]),
                        buy_commission=Decimal(pnl_data["buy_commission"]),
                        sell_commission=Decimal(pnl_data["sell_commission"]),
                        pnl=Decimal(pnl_data["pnl"]),
                        pnl_percent=Decimal(pnl_data["pnl_percent"]),
                        buy_date=datetime.fromisoformat(pnl_data["buy_date"]),
                        sell_date=datetime.fromisoformat(pnl_data["sell_date"]),
                    )
                )

            # 設定復元
            settings = data.get("settings", {})
            if "commission_rate" in settings:
                self.commission_rate = Decimal(settings["commission_rate"])
            if "tax_rate" in settings:
                self.tax_rate = Decimal(settings["tax_rate"])
            if "trade_counter" in settings:
                self._trade_counter = settings["trade_counter"]

            logger.info(f"JSON読み込み完了: {mask_sensitive_info(str(safe_path))}")

        except Exception as e:
            logger.error(
                f"データの読み込み中にエラーが発生しました。ファイル形式が正しいか、破損していないか確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise

    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        株式買い注文を実行（完全なトランザクション保護）

        ポートフォリオ更新と取引履歴追加を単一のトランザクションで処理し、
        データの整合性を保証する。

        Args:
            symbol: 銘柄コード
            quantity: 購入数量
            price: 購入価格
            current_market_price: 現在の市場価格（ポートフォリオ評価用）
            notes: 取引メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引結果辞書（取引ID、更新後ポジション、手数料等）

        Raises:
            ValueError: 無効な購入パラメータ
            Exception: データベース処理エラー
        """
        buy_logger = logger.bind(
            operation="buy_stock",
            symbol=symbol,
            quantity=quantity,
            price=str(price),
            persist_to_db=persist_to_db,
        )

        buy_logger.info("株式買い注文処理開始")

        # 安全なDecimal変換と検証
        safe_price = safe_decimal_conversion(price, "購入価格")
        validate_positive_decimal(safe_price, "購入価格")

        if quantity <= 0:
            raise ValueError(f"購入数量は正数である必要があります: {quantity}")

        safe_current_market_price = None
        if current_market_price is not None:
            safe_current_market_price = safe_decimal_conversion(
                current_market_price, "現在価格"
            )
            validate_positive_decimal(safe_current_market_price, "現在価格")

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        counter_backup = self._trade_counter

        try:
            # 手数料計算
            commission = self._calculate_commission(safe_price, quantity)
            timestamp = datetime.now()

            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    # 1. 銘柄マスタの存在確認・作成
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        buy_logger.info("銘柄マスタに未登録、新規作成")
                        stock = Stock(
                            code=symbol,
                            name=symbol,
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()

                    # 2. 取引ID生成
                    trade_id = self._generate_trade_id()

                    # 3. データベース取引記録を作成
                    db_trade = DBTrade.create_buy_trade(
                        session=session,
                        stock_code=symbol,
                        quantity=quantity,
                        price=safe_price,
                        commission=commission,
                        memo=notes,
                    )

                    # 4. メモリ内取引記録作成
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=TradeType.BUY,
                        quantity=quantity,
                        price=safe_price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )

                    # 5. ポジション更新（原子的実行）
                    old_position = self.positions.get(symbol)
                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)
                    new_position = self.positions.get(symbol)

                    # 6. 現在価格更新（指定されている場合）
                    if safe_current_market_price and symbol in self.positions:
                        self.positions[symbol].current_price = safe_current_market_price

                    # 中間状態をflushして整合性を確認
                    session.flush()

                    # ビジネスイベントログ
                    log_business_event(
                        "stock_purchased",
                        trade_id=mask_sensitive_info(str(trade_id)),
                        symbol=symbol,
                        quantity=mask_sensitive_info(f"quantity: {quantity}"),
                        price=mask_sensitive_info(f"price: {str(price)}"),
                        commission=mask_sensitive_info(
                            f"commission: {str(commission)}"
                        ),
                        old_position=(
                            mask_sensitive_info(str(old_position.to_dict()))
                            if old_position
                            else None
                        ),
                        new_position=(
                            mask_sensitive_info(str(new_position.to_dict()))
                            if new_position
                            else None
                        ),
                        persisted=True,
                    )

                    buy_logger.info(
                        "株式買い注文完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                        commission=str(commission),
                    )
            else:
                # メモリ内のみの処理
                trade_id = self._generate_trade_id()

                memory_trade = Trade(
                    id=trade_id,
                    symbol=symbol,
                    trade_type=TradeType.BUY,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    commission=commission,
                    notes=notes,
                )

                old_position = self.positions.get(symbol)
                self.trades.append(memory_trade)
                self._update_position(memory_trade)
                new_position = self.positions.get(symbol)

                if safe_current_market_price and symbol in self.positions:
                    self.positions[symbol].current_price = safe_current_market_price

                log_business_event(
                    "stock_purchased",
                    trade_id=mask_sensitive_info(str(trade_id)),
                    symbol=symbol,
                    quantity=mask_sensitive_info(f"quantity: {quantity}"),
                    price=mask_sensitive_info(f"price: {str(price)}"),
                    commission=mask_sensitive_info(f"commission: {str(commission)}"),
                    old_position=(
                        mask_sensitive_info(str(old_position.to_dict()))
                        if old_position
                        else None
                    ),
                    new_position=(
                        mask_sensitive_info(str(new_position.to_dict()))
                        if new_position
                        else None
                    ),
                    persisted=False,
                )

                buy_logger.info(
                    "株式買い注文完了（メモリのみ）",
                    trade_id=trade_id,
                    commission=str(commission),
                )

            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": str(price),
                "commission": str(commission),
                "timestamp": timestamp.isoformat(),
                "position": (
                    self.positions[symbol].to_dict()
                    if symbol in self.positions
                    else None
                ),
                "total_cost": str(price * quantity + commission),
            }

            return result

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "buy_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": str(price),
                    "persist_to_db": persist_to_db,
                },
            )
            buy_logger.error(
                "株式買い注文失敗、変更をロールバック", extra={"error": str(e)}
            )
            raise

    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        株式売り注文を実行（完全なトランザクション保護）

        ポートフォリオ更新、取引履歴追加、実現損益計算を
        単一のトランザクションで処理し、データの整合性を保証する。

        Args:
            symbol: 銘柄コード
            quantity: 売却数量
            price: 売却価格
            current_market_price: 現在の市場価格（ポートフォリオ評価用）
            notes: 取引メモ
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引結果辞書（取引ID、更新後ポジション、実現損益等）

        Raises:
            ValueError: 無効な売却パラメータ（保有数量不足等）
            Exception: データベース処理エラー
        """
        sell_logger = logger.bind(
            operation="sell_stock",
            symbol=symbol,
            quantity=quantity,
            price=str(price),
            persist_to_db=persist_to_db,
        )

        sell_logger.info("株式売り注文処理開始")

        # パラメータ検証
        if quantity <= 0:
            raise ValueError(f"売却数量は正数である必要があります: {quantity}")
        if price <= 0:
            raise ValueError(f"売却価格は正数である必要があります: {price}")

        # ポジション存在確認
        if symbol not in self.positions:
            raise ValueError(f"銘柄 '{symbol}' のポジションが存在しません")

        current_position = self.positions[symbol]
        if current_position.quantity < quantity:
            raise ValueError(
                f"売却数量 ({quantity}) が保有数量 ({current_position.quantity}) を超過しています"
            )

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            # 手数料計算
            commission = self._calculate_commission(price, quantity)
            timestamp = datetime.now()

            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    # 1. 取引ID生成
                    trade_id = self._generate_trade_id()

                    # 2. データベース取引記録を作成
                    db_trade = DBTrade.create_sell_trade(
                        session=session,
                        stock_code=symbol,
                        quantity=quantity,
                        price=price,
                        commission=commission,
                        memo=notes,
                    )

                    # 3. メモリ内取引記録作成
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=TradeType.SELL,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )

                    # 4. ポジション更新と実現損益計算（原子的実行）
                    old_position = self.positions[symbol]
                    old_realized_pnl_count = len(self.realized_pnl)

                    self.trades.append(memory_trade)
                    self._update_position(memory_trade)

                    new_position = self.positions.get(symbol)
                    new_realized_pnl = None
                    if len(self.realized_pnl) > old_realized_pnl_count:
                        new_realized_pnl = self.realized_pnl[-1]

                    # 5. 現在価格更新（指定されている場合）
                    if current_market_price and symbol in self.positions:
                        self.positions[symbol].current_price = current_market_price

                    # 中間状態をflushして整合性を確認
                    session.flush()

                    # ビジネスイベントログ
                    log_business_event(
                        "stock_sold",
                        trade_id=mask_sensitive_info(str(trade_id)),
                        symbol=symbol,
                        quantity=mask_sensitive_info(f"quantity: {quantity}"),
                        price=mask_sensitive_info(f"price: {str(price)}"),
                        commission=mask_sensitive_info(
                            f"commission: {str(commission)}"
                        ),
                        old_position=mask_sensitive_info(str(old_position.to_dict())),
                        new_position=(
                            mask_sensitive_info(str(new_position.to_dict()))
                            if new_position
                            else None
                        ),
                        realized_pnl=(
                            mask_sensitive_info(str(new_realized_pnl.to_dict()))
                            if new_realized_pnl
                            else None
                        ),
                        persisted=True,
                    )

                    sell_logger.info(
                        "株式売り注文完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                        commission=str(commission),
                        realized_pnl=(
                            str(new_realized_pnl.pnl) if new_realized_pnl else None
                        ),
                    )
            else:
                # メモリ内のみの処理
                trade_id = self._generate_trade_id()

                memory_trade = Trade(
                    id=trade_id,
                    symbol=symbol,
                    trade_type=TradeType.SELL,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    commission=commission,
                    notes=notes,
                )

                old_position = self.positions[symbol]
                old_realized_pnl_count = len(self.realized_pnl)

                self.trades.append(memory_trade)
                self._update_position(memory_trade)

                new_position = self.positions.get(symbol)
                new_realized_pnl = None
                if len(self.realized_pnl) > old_realized_pnl_count:
                    new_realized_pnl = self.realized_pnl[-1]

                if safe_current_market_price and symbol in self.positions:
                    self.positions[symbol].current_price = safe_current_market_price

                log_business_event(
                    "stock_sold",
                    trade_id=mask_sensitive_info(str(trade_id)),
                    symbol=symbol,
                    quantity=mask_sensitive_info(f"quantity: {quantity}"),
                    price=mask_sensitive_info(f"price: {str(price)}"),
                    commission=mask_sensitive_info(f"commission: {str(commission)}"),
                    old_position=mask_sensitive_info(str(old_position.to_dict())),
                    new_position=(
                        mask_sensitive_info(str(new_position.to_dict()))
                        if new_position
                        else None
                    ),
                    realized_pnl=(
                        mask_sensitive_info(str(new_realized_pnl.to_dict()))
                        if new_realized_pnl
                        else None
                    ),
                    persisted=False,
                )

                sell_logger.info(
                    "株式売り注文完了（メモリのみ）",
                    trade_id=trade_id,
                    commission=str(commission),
                    realized_pnl=(
                        str(new_realized_pnl.pnl) if new_realized_pnl else None
                    ),
                )

            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": str(price),
                "commission": str(commission),
                "timestamp": timestamp.isoformat(),
                "position": new_position.to_dict() if new_position else None,
                "position_closed": new_position is None,
                "realized_pnl": (
                    new_realized_pnl.to_dict() if new_realized_pnl else None
                ),
                "gross_proceeds": str(price * quantity - commission),
            }

            return result

        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "sell_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": str(price),
                    "persist_to_db": persist_to_db,
                },
            )
            sell_logger.error(
                "株式売り注文失敗、変更をロールバック", extra={"error": str(e)}
            )
            raise

    def execute_trade_order(
        self, trade_order: Dict[str, Any], persist_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        取引注文を実行（買い/売りを統一インターフェースで処理）

        Args:
            trade_order: 取引注文辞書
                {
                    "action": "buy" | "sell",
                    "symbol": str,
                    "quantity": int,
                    "price": Decimal,
                    "current_market_price": Optional[Decimal],
                    "notes": str
                }
            persist_to_db: データベースに永続化するかどうか

        Returns:
            取引結果辞書
        """
        action = str(trade_order.get("action", "")).lower()

        if action == "buy":
            return self.buy_stock(
                symbol=trade_order["symbol"],
                quantity=trade_order["quantity"],
                price=trade_order["price"],
                current_market_price=trade_order.get("current_market_price"),
                notes=trade_order.get("notes", ""),
                persist_to_db=persist_to_db,
            )
        elif action == "sell":
            return self.sell_stock(
                symbol=trade_order["symbol"],
                quantity=trade_order["quantity"],
                price=trade_order["price"],
                current_market_price=trade_order.get("current_market_price"),
                notes=trade_order.get("notes", ""),
                persist_to_db=persist_to_db,
            )
        else:
            raise ValueError(
                f"無効な取引アクション: {action}. 'buy' または 'sell' を指定してください"
            )

    def calculate_tax_implications(
        self, year: int, accounting_method: str = "FIFO"
    ) -> Dict:
        """
        税務計算（会計原則対応版）

        Args:
            year: 税務年度
            accounting_method: 会計手法 ("FIFO", "LIFO", "AVERAGE")
        """
        try:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59, 59)

            # 年内の実現損益を効率的に取得
            year_pnl = [
                pnl
                for pnl in self.realized_pnl
                if year_start <= pnl.sell_date <= year_end
            ]

            if not year_pnl:
                return {
                    "year": year,
                    "accounting_method": accounting_method,
                    "total_trades": 0,
                    "total_gain": "0.00",
                    "total_loss": "0.00",
                    "net_gain": "0.00",
                    "tax_due": "0.00",
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "average_gain_per_winning_trade": "0.00",
                    "average_loss_per_losing_trade": "0.00",
                    "win_rate": "0.00%",
                }

            # 総利益と総损失を計算
            gains = [pnl.pnl for pnl in year_pnl if pnl.pnl > 0]
            losses = [pnl.pnl for pnl in year_pnl if pnl.pnl < 0]

            total_gain = sum(gains) if gains else Decimal("0")
            total_loss = sum(abs(loss) for loss in losses) if losses else Decimal("0")
            net_gain = total_gain - total_loss

            # 税額計算（利益が出た場合のみ）
            tax_due = net_gain * self.tax_rate if net_gain > 0 else Decimal("0")

            # 統計情報を追加
            winning_trades_count = len(gains)
            losing_trades_count = len(losses)
            total_trades = len(year_pnl)

            avg_gain = (
                total_gain / winning_trades_count
                if winning_trades_count > 0
                else Decimal("0")
            )
            avg_loss = (
                total_loss / losing_trades_count
                if losing_trades_count > 0
                else Decimal("0")
            )
            win_rate = (
                (winning_trades_count / total_trades * 100)
                if total_trades > 0
                else Decimal("0")
            )

            return {
                "year": year,
                "accounting_method": accounting_method,
                "total_trades": total_trades,
                "total_gain": str(total_gain.quantize(Decimal("0.01"))),
                "total_loss": str(total_loss.quantize(Decimal("0.01"))),
                "net_gain": str(net_gain.quantize(Decimal("0.01"))),
                "tax_due": str(tax_due.quantize(Decimal("0.01"))),
                "winning_trades": winning_trades_count,
                "losing_trades": losing_trades_count,
                "average_gain_per_winning_trade": str(
                    avg_gain.quantize(Decimal("0.01"))
                ),
                "average_loss_per_losing_trade": str(
                    avg_loss.quantize(Decimal("0.01"))
                ),
                "win_rate": f"{win_rate.quantize(Decimal('0.01'))}%",
            }

        except Exception as e:
            logger.error(
                f"税務計算中に予期せぬエラーが発生しました。入力データまたは計算ロジックを確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise


# 使用例
if __name__ == "__main__":
    from datetime import datetime, timedelta
    from decimal import Decimal

    # 取引管理システムを初期化
    tm = TradeManager(commission_rate=Decimal("0.001"), tax_rate=Decimal("0.2"))

    # サンプル取引を追加
    base_date = datetime.now() - timedelta(days=30)

    # トヨタ株の取引例
    tm.add_trade("7203", TradeType.BUY, 100, Decimal("2500"), base_date)
    tm.add_trade(
        "7203", TradeType.BUY, 200, Decimal("2450"), base_date + timedelta(days=1)
    )

    # 現在価格を更新
    tm.update_current_prices({"7203": Decimal("2600")})

    # ポジション情報表示
    position = tm.get_position("7203")
    if position:
        logger.info(
            "ポジション情報表示",
            section="position_info",
            symbol=position.symbol,
            quantity=position.quantity,
            average_price=str(position.average_price),
            total_cost=str(position.total_cost),
            current_price=str(position.current_price),
            market_value=str(position.market_value),
            unrealized_pnl=str(position.unrealized_pnl),
            unrealized_pnl_percent=str(position.unrealized_pnl_percent),
        )

    # 一部売却
    tm.add_trade(
        "7203", TradeType.SELL, 100, Decimal("2650"), base_date + timedelta(days=5)
    )

    # 実現損益表示
    realized_pnl = tm.get_realized_pnl_history("7203")
    if realized_pnl:
        logger.info("実現損益表示開始", extra={"section": "realized_pnl"})
        for pnl in realized_pnl:
            logger.info(
                "実現損益詳細",
                extra={
                    "section": "realized_pnl_detail",
                    "symbol": pnl.symbol,
                    "quantity": mask_sensitive_info(f"quantity: {pnl.quantity}"),
                    "buy_price": mask_sensitive_info(
                        f"buy_price: {str(pnl.buy_price)}"
                    ),
                    "sell_price": mask_sensitive_info(
                        f"sell_price: {str(pnl.sell_price)}"
                    ),
                    "pnl": mask_sensitive_info(f"pnl: {str(pnl.pnl)}"),
                    "pnl_percent": mask_sensitive_info(
                        f"pnl_percent: {str(pnl.pnl_percent)}"
                    ),
                },
            )

    # ポートフォリオサマリー（機密情報マスキング適用）
    summary = tm.get_portfolio_summary()
    # 機密情報を含む可能性のあるサマリーデータをマスキング
    masked_summary = {}
    for key, value in summary.items():
        if any(
            sensitive_key in key.lower()
            for sensitive_key in ["total", "value", "pnl", "price", "cost", "balance"]
        ):
            masked_summary[key] = mask_sensitive_info(f"{key}: {str(value)}")
        else:
            masked_summary[key] = value

    extra_data = {"section": "portfolio_summary", **masked_summary}
    logger.info("ポートフォリオサマリー", extra=extra_data)

    # CSV出力例
    logger.info("CSV出力開始", extra={"section": "csv_export"})
    try:
        tm.export_to_csv("trades.csv", "trades")
        tm.export_to_csv("positions.csv", "positions")
        tm.export_to_csv("realized_pnl.csv", "realized_pnl")
        logger.info("CSV出力完了", extra={"section": "csv_export", "status": "success"})
    except Exception as e:
        log_error_with_context(e, {"section": "csv_export", "operation": "export_csv"})
