"""
基本的なフォーマット機能
通貨、パーセンテージ、出来高、色などの基本的なフォーマット処理
"""

from typing import Union


def format_currency(
    amount: Union[int, float], currency: str = "¥", decimal_places: int = 0
) -> str:
    """
    通貨フォーマット

    Args:
        amount: 金額
        currency: 通貨記号
        decimal_places: 小数点以下の桁数

    Returns:
        フォーマット済み金額文字列
    """
    if amount is None:
        return "N/A"

    if decimal_places == 0:
        return f"{currency}{amount:,.0f}"
    else:
        return f"{currency}{amount:,.{decimal_places}f}"


def format_percentage(
    value: Union[int, float], decimal_places: int = 2, show_sign: bool = True
) -> str:
    """
    パーセンテージフォーマット

    Args:
        value: 値
        decimal_places: 小数点以下の桁数
        show_sign: 符号を表示するか

    Returns:
        フォーマット済みパーセンテージ文字列
    """
    if value is None:
        return "N/A"

    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_volume(volume: Union[int, float]) -> str:
    """
    出来高フォーマット

    Args:
        volume: 出来高

    Returns:
        フォーマット済み出来高文字列
    """
    if volume is None:
        return "N/A"

    volume = int(volume)
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:,}"


def get_change_color(value: Union[int, float]) -> str:
    """
    変化値に基づく色を取得

    Args:
        value: 変化値

    Returns:
        色名
    """
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    else:
        return "white"


def format_large_number(number: Union[int, float], precision: int = 1) -> str:
    """
    大きな数値を適切にフォーマット

    Args:
        number: 数値
        precision: 小数点以下桁数

    Returns:
        フォーマット済み数値文字列
    """
    if number is None:
        return "N/A"

    abs_number = abs(number)
    sign = "-" if number < 0 else ""

    if abs_number >= 1_000_000_000_000:  # 1兆以上
        return f"{sign}{abs_number / 1_000_000_000_000:.{precision}f}T"
    elif abs_number >= 1_000_000_000:  # 10億以上
        return f"{sign}{abs_number / 1_000_000_000:.{precision}f}B"
    elif abs_number >= 1_000_000:  # 100万以上
        return f"{sign}{abs_number / 1_000_000:.{precision}f}M"
    elif abs_number >= 1_000:  # 1000以上
        return f"{sign}{abs_number / 1_000:.{precision}f}K"
    else:
        return f"{sign}{abs_number:.{precision}f}"