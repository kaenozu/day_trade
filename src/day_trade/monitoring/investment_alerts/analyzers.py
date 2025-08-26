#!/usr/bin/env python3
"""
投資機会アラートシステム - 機会分析器の統合クラス
"""

from typing import Dict, Optional

from .enums import OpportunityType
from .models import InvestmentOpportunity, OpportunityConfig
from .technical_analyzers import TechnicalAnalyzer
from .volume_analyzers import VolumeAnalyzer


class OpportunityAnalyzer:
    """投資機会分析器の統合クラス"""
    
    def __init__(self):
        """分析器の初期化"""
        self.technical_analyzer = TechnicalAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()

    async def analyze_opportunity(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """機会タイプ別分析の振り分け"""
        
        if config.opportunity_type == OpportunityType.TECHNICAL_BREAKOUT:
            return await self.technical_analyzer.analyze_technical_breakout(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.MOMENTUM_SIGNAL:
            return await self.technical_analyzer.analyze_momentum_signal(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.REVERSAL_PATTERN:
            return await self.technical_analyzer.analyze_reversal_pattern(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.VOLUME_ANOMALY:
            return await self.volume_analyzer.analyze_volume_anomaly(
                symbol, config, indicators, current_price
            )
        elif config.opportunity_type == OpportunityType.VOLATILITY_SQUEEZE:
            return await self.technical_analyzer.analyze_volatility_squeeze(
                symbol, config, indicators, current_price
            )
        else:
            # その他の機会タイプは基本的な分析
            return await self.volume_analyzer.analyze_generic_opportunity(
                symbol, config, indicators, current_price
            )

