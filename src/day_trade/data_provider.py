#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Provider Interface
"""

from abc import ABC, abstractmethod
from typing import Dict

class AbstractDataProvider(ABC):
    @abstractmethod
    def get_latest_prices(self) -> Dict[str, float]:
        """銘柄ごとの最新価格を取得する"""
        pass

class ManualDataProvider(AbstractDataProvider):
    """テスト用の手動データプロバイダー"""
    def __init__(self, price_data: Dict[str, float]):
        self.price_data = price_data

    def get_latest_prices(self) -> Dict[str, float]:
        return self.price_data
