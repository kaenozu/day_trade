"""
ルール基底クラス

シグナルルールの基底クラス定義
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import pandas as pd

from .config import SignalRulesConfig, get_shared_config


class SignalRule(ABC):
    """シグナルルールの基底クラス"""

    def __init__(self, name: str, weight: float = 1.0):
        """
        Args:
            name: ルール名
            weight: 重み（デフォルト: 1.0）
        """
        self.name = name
        self.weight = weight

    def _get_config_with_fallback(self, config: Optional[SignalRulesConfig] = None) -> SignalRulesConfig:
        """
        Issue #649対応: configハンドリングの最適化
        configが提供されない場合は共有configを使用

        Args:
            config: 設定オブジェクト（任意）

        Returns:
            有効な設定オブジェクト
        """
        if config is None:
            return get_shared_config()
        return config

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        """
        ルールを評価

        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            patterns: チャートパターン認識結果
            config: 設定オブジェクト（任意）

        Returns:
            条件が満たされたか、信頼度スコア（0-100）
        """
        pass

    def __str__(self) -> str:
        """文字列表現"""
        return f"{self.name} (weight: {self.weight})"

    def __repr__(self) -> str:
        """詳細文字列表現"""
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"