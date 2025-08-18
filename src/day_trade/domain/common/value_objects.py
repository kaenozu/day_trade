"""
DDD値オブジェクト

ドメイン駆動設計における値オブジェクトの実装
型安全性とドメインロジックの整合性を確保
"""

from decimal import Decimal
from dataclasses import dataclass
from typing import Union
import re


@dataclass(frozen=True)
class Money:
    """金額を表す値オブジェクト"""
    amount: Decimal
    currency: str = "JPY"

    def __post_init__(self):
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, 'amount', Decimal(str(self.amount)))

        if self.amount < 0:
            raise ValueError("金額は0以上である必要があります")

        if not self.currency or len(self.currency) != 3:
            raise ValueError("通貨コードは3文字である必要があります")

    def add(self, other: 'Money') -> 'Money':
        """金額の加算"""
        if self.currency != other.currency:
            raise ValueError(f"異なる通貨での計算はできません: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def subtract(self, other: 'Money') -> 'Money':
        """金額の減算"""
        if self.currency != other.currency:
            raise ValueError(f"異なる通貨での計算はできません: {self.currency} vs {other.currency}")
        result = self.amount - other.amount
        if result < 0:
            raise ValueError("減算結果が負の値になります")
        return Money(result, self.currency)

    def multiply(self, multiplier: Union[int, float, Decimal]) -> 'Money':
        """金額の乗算"""
        if not isinstance(multiplier, Decimal):
            multiplier = Decimal(str(multiplier))
        result = self.amount * multiplier
        if result < 0:
            raise ValueError("乗算結果が負の値になります")
        return Money(result, self.currency)

    def divide(self, divisor: Union[int, float, Decimal]) -> 'Money':
        """金額の除算"""
        if not isinstance(divisor, Decimal):
            divisor = Decimal(str(divisor))
        if divisor == 0:
            raise ValueError("0で除算することはできません")
        return Money(self.amount / divisor, self.currency)

    def is_zero(self) -> bool:
        """金額が0かどうか"""
        return self.amount == 0

    def is_positive(self) -> bool:
        """金額が正の値かどうか"""
        return self.amount > 0

    def __str__(self) -> str:
        return f"{self.amount} {self.currency}"

    def __lt__(self, other: 'Money') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.amount < other.amount

    def __le__(self, other: 'Money') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.amount <= other.amount

    def __gt__(self, other: 'Money') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.amount > other.amount

    def __ge__(self, other: 'Money') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.amount >= other.amount


@dataclass(frozen=True)
class Price:
    """価格を表す値オブジェクト"""
    value: Decimal
    currency: str = "JPY"

    def __post_init__(self):
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(str(self.value)))

        if self.value <= 0:
            raise ValueError("価格は正の値である必要があります")

        if not self.currency or len(self.currency) != 3:
            raise ValueError("通貨コードは3文字である必要があります")

    def to_money(self, quantity: 'Quantity') -> Money:
        """価格×数量で金額を計算"""
        total_amount = self.value * quantity.value
        return Money(total_amount, self.currency)

    def calculate_commission(self, rate: Decimal) -> Money:
        """手数料計算"""
        if not isinstance(rate, Decimal):
            rate = Decimal(str(rate))
        if rate < 0 or rate > 1:
            raise ValueError("手数料率は0-1の範囲である必要があります")
        commission_amount = self.value * rate
        return Money(commission_amount, self.currency)

    def apply_spread(self, spread_percent: Decimal) -> 'Price':
        """スプレッド適用"""
        if not isinstance(spread_percent, Decimal):
            spread_percent = Decimal(str(spread_percent))
        spread_amount = self.value * (spread_percent / 100)
        return Price(self.value + spread_amount, self.currency)

    def __str__(self) -> str:
        return f"{self.value} {self.currency}"

    def __lt__(self, other: 'Price') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.value < other.value

    def __le__(self, other: 'Price') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.value <= other.value

    def __gt__(self, other: 'Price') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.value > other.value

    def __ge__(self, other: 'Price') -> bool:
        if self.currency != other.currency:
            raise ValueError("異なる通貨での比較はできません")
        return self.value >= other.value


@dataclass(frozen=True)
class Quantity:
    """数量を表す値オブジェクト"""
    value: int

    def __post_init__(self):
        if not isinstance(self.value, int):
            raise TypeError("数量は整数である必要があります")

        if self.value <= 0:
            raise ValueError("数量は正の値である必要があります")

        # 実用的な上限設定（1億株）
        if self.value > 100_000_000:
            raise ValueError("数量が上限を超えています")

    def add(self, other: 'Quantity') -> 'Quantity':
        """数量の加算"""
        return Quantity(self.value + other.value)

    def subtract(self, other: 'Quantity') -> 'Quantity':
        """数量の減算"""
        result = self.value - other.value
        if result <= 0:
            raise ValueError("減算結果が0以下になります")
        return Quantity(result)

    def multiply(self, multiplier: int) -> 'Quantity':
        """数量の乗算"""
        if not isinstance(multiplier, int):
            raise TypeError("乗数は整数である必要があります")
        if multiplier <= 0:
            raise ValueError("乗数は正の値である必要があります")
        return Quantity(self.value * multiplier)

    def is_even(self) -> bool:
        """偶数かどうか"""
        return self.value % 2 == 0

    def is_divisible_by(self, divisor: int) -> bool:
        """指定した値で割り切れるかどうか"""
        if divisor <= 0:
            raise ValueError("除数は正の値である必要があります")
        return self.value % divisor == 0

    def __str__(self) -> str:
        return f"{self.value}株"

    def __lt__(self, other: 'Quantity') -> bool:
        return self.value < other.value

    def __le__(self, other: 'Quantity') -> bool:
        return self.value <= other.value

    def __gt__(self, other: 'Quantity') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'Quantity') -> bool:
        return self.value >= other.value


@dataclass(frozen=True)
class Symbol:
    """銘柄コードを表す値オブジェクト"""
    code: str

    def __post_init__(self):
        if not self.code:
            raise ValueError("銘柄コードが未指定です")

        # 銘柄コードの基本検証
        if len(self.code) < 3 or len(self.code) > 10:
            raise ValueError("銘柄コードは3-10文字である必要があります")

        # 日本株式コード（4桁数字）の検証
        if self._is_japanese_stock_code():
            if not self.code.isdigit() or len(self.code) != 4:
                raise ValueError("日本株式コードは4桁の数字である必要があります")

        # 英数字とハイフンのみ許可
        if not re.match(r'^[A-Z0-9\-]+$', self.code.upper()):
            raise ValueError("銘柄コードは英数字とハイフンのみ使用可能です")

    def _is_japanese_stock_code(self) -> bool:
        """日本株式コードかどうかの判定"""
        return (
            len(self.code) == 4 and
            self.code.isdigit() and
            1000 <= int(self.code) <= 9999
        )

    def is_etf(self) -> bool:
        """ETFかどうかの判定（簡易版）"""
        if not self._is_japanese_stock_code():
            return False
        code_int = int(self.code)
        # 日本のETFコード範囲（簡易版）
        return (1300 <= code_int <= 1699) or (2500 <= code_int <= 2599)

    def is_reit(self) -> bool:
        """REITかどうかの判定（簡易版）"""
        if not self._is_japanese_stock_code():
            return False
        code_int = int(self.code)
        # 日本のREITコード範囲（簡易版）
        return 3000 <= code_int <= 3999

    def get_market_category(self) -> str:
        """市場カテゴリの取得（簡易版）"""
        if not self._is_japanese_stock_code():
            return "その他"

        code_int = int(self.code)

        if 1000 <= code_int <= 1999:
            return "食品・化学・医薬品等"
        elif 2000 <= code_int <= 2999:
            return "繊維・紙パルプ・石油石炭等"
        elif 3000 <= code_int <= 3999:
            return "ゴム・ガラス土石・鉄鋼等"
        elif 4000 <= code_int <= 4999:
            return "機械・電気機器等"
        elif 5000 <= code_int <= 5999:
            return "陸運・海運・空運・通信等"
        elif 6000 <= code_int <= 6999:
            return "電力・ガス・小売・金融等"
        elif 7000 <= code_int <= 7999:
            return "不動産・サービス等"
        elif 8000 <= code_int <= 8999:
            return "銀行・証券・保険等"
        elif 9000 <= code_int <= 9999:
            return "情報通信・サービス等"
        else:
            return "その他"

    def __str__(self) -> str:
        return self.code

    def __lt__(self, other: 'Symbol') -> bool:
        return self.code < other.code

    def __le__(self, other: 'Symbol') -> bool:
        return self.code <= other.code

    def __gt__(self, other: 'Symbol') -> bool:
        return self.code > other.code

    def __ge__(self, other: 'Symbol') -> bool:
        return self.code >= other.code


# ファクトリー関数
def create_money(amount: Union[int, float, str, Decimal], currency: str = "JPY") -> Money:
    """金額オブジェクトの作成"""
    return Money(Decimal(str(amount)), currency)


def create_price(value: Union[int, float, str, Decimal], currency: str = "JPY") -> Price:
    """価格オブジェクトの作成"""
    return Price(Decimal(str(value)), currency)


def create_quantity(value: int) -> Quantity:
    """数量オブジェクトの作成"""
    return Quantity(value)


def create_symbol(code: str) -> Symbol:
    """銘柄コードオブジェクトの作成"""
    return Symbol(code.upper().strip())


@dataclass(frozen=True)
class Percentage:
    """パーセンテージを表す値オブジェクト"""
    value: Decimal

    def __post_init__(self):
        if not isinstance(self.value, Decimal):
            object.__setattr__(self, 'value', Decimal(str(self.value)))

        # パーセンテージの範囲チェック（-100% ～ +1000%）
        if self.value < -100 or self.value > 1000:
            raise ValueError("パーセンテージは-100%から1000%の範囲である必要があります")

    def to_decimal(self) -> Decimal:
        """小数値に変換（例：5% → 0.05）"""
        return self.value / 100

    def apply_to(self, amount: Money) -> Money:
        """金額にパーセンテージを適用"""
        decimal_value = self.to_decimal()
        return amount.multiply(decimal_value)

    def add(self, other: 'Percentage') -> 'Percentage':
        """パーセンテージの加算"""
        return Percentage(self.value + other.value)

    def subtract(self, other: 'Percentage') -> 'Percentage':
        """パーセンテージの減算"""
        return Percentage(self.value - other.value)

    def is_positive(self) -> bool:
        """正の値かどうか"""
        return self.value > 0

    def is_negative(self) -> bool:
        """負の値かどうか"""
        return self.value < 0

    def is_zero(self) -> bool:
        """ゼロかどうか"""
        return self.value == 0

    def __str__(self) -> str:
        return f"{self.value}%"

    def __lt__(self, other: 'Percentage') -> bool:
        return self.value < other.value

    def __le__(self, other: 'Percentage') -> bool:
        return self.value <= other.value

    def __gt__(self, other: 'Percentage') -> bool:
        return self.value > other.value

    def __ge__(self, other: 'Percentage') -> bool:
        return self.value >= other.value


def create_percentage(value: Union[int, float, str, Decimal]) -> Percentage:
    """パーセンテージオブジェクトの作成"""
    return Percentage(Decimal(str(value)))


# 便利な定数
ZERO_JPY = Money(Decimal('0'), 'JPY')
ZERO_USD = Money(Decimal('0'), 'USD')
ZERO_QUANTITY = Quantity(1)  # 最小単位
ZERO_PERCENT = Percentage(Decimal('0'))