"""
シグナルルールの基底クラス
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from .config import SignalRulesConfig, _get_shared_config


class SignalRule:
    """シグナルルールの基底クラス"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def _get_config_with_fallback(
        self, config: Optional[SignalRulesConfig] = None
    ) -> SignalRulesConfig:
        """
        Issue #649対応: configハンドリングの最適化
        configが提供されない場合は共有configを使用
        """
        if config is None:
            return _get_shared_config()
        return config

    def evaluate(
        self,
        df: pd.DataFrame,
        indicators: pd.DataFrame,
        patterns: Dict,
        config: Optional[SignalRulesConfig] = None,
    ) -> Tuple[bool, float]:
        """
        ルールを評価

        Returns:
            条件が満たされたか、信頼度スコア
        """
        raise NotImplementedError