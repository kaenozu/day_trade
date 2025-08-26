"""
株価データ取得バリデーション機能
シンボル、日付範囲などの検証機能を提供
"""

from datetime import datetime
from typing import Union

from .exceptions import InvalidSymbolError


class BaseValidators:
    """
    基本的なバリデーション機能を提供するクラス
    """

    def __init__(self):
        self.logger = None  # 子クラスで設定される

    def _validate_symbol(self, symbol: str) -> None:
        """シンボルの妥当性をチェック"""
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"無効なシンボル: {symbol}")

        if len(symbol) < 2:
            raise InvalidSymbolError(f"シンボルが短すぎます: {symbol}")

    def _validate_date_range(
        self, start_date: Union[str, datetime], end_date: Union[str, datetime]
    ) -> None:
        """日付範囲の妥当性をチェック"""
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError as e:
                raise InvalidSymbolError(f"無効な開始日形式: {start_date}") from e

        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError as e:
                raise InvalidSymbolError(f"無効な終了日形式: {end_date}") from e

        if start_date >= end_date:
            raise InvalidSymbolError(
                f"開始日が終了日以降です: {start_date} >= {end_date}"
            )

        if end_date > datetime.now() and self.logger:
            self.logger.warning(f"終了日が未来の日付です: {end_date}")

    def _format_symbol(self, code: str, market: str = "T") -> str:
        """
        証券コードをyfinance形式にフォーマット

        Args:
            code: 証券コード（例：7203）
            market: 市場コード（T:東証、デフォルト）

        Returns:
            フォーマット済みシンボル（例：7203.T）
        """
        # すでに市場コードが付いている場合はそのまま返す
        if "." in code:
            return code
        return f"{code}.{market}"