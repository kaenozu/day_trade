#!/usr/bin/env python3
"""
多角的データ収集システム - 基底クラス

Issue #322: ML Data Shortage Problem Resolution
データ収集器の基底クラス定義
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import CollectedData


class DataCollector(ABC):
    """データ収集器基底クラス"""

    @abstractmethod
    async def collect_data(self, symbol: str, **kwargs) -> CollectedData:
        """
        データ収集実行

        Args:
            symbol: 銘柄シンボル
            **kwargs: 追加パラメータ

        Returns:
            CollectedData: 収集されたデータ
        """
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """
        ヘルス状態取得

        Returns:
            Dict[str, Any]: ヘルス状態情報
        """
        pass