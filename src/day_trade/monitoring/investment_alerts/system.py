#!/usr/bin/env python3
"""
投資機会アラートシステム - メインシステム
"""

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .detectors import OpportunityDetector
from .enums import OpportunityType, OpportunitySeverity
from .models import AlertConfig, OpportunityConfig
from .opportunity_manager import OpportunityManager
from .summary_generator import SummaryGenerator

logger = get_context_logger(__name__)


class InvestmentOpportunityAlertSystem:
    """投資機会アラートシステム"""

    def __init__(self, config: Optional[AlertConfig] = None):
        """アラートシステムの初期化"""
        self.config = config or AlertConfig()
        self.opportunity_configs: Dict[str, OpportunityConfig] = {}
        self.detector = OpportunityDetector()
        self.opportunity_manager = OpportunityManager(self.config)
        self.summary_generator = SummaryGenerator()

        # 監視タスク
        self.monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

        # デフォルト設定
        self._setup_default_opportunity_configs()

    def _setup_default_opportunity_configs(self) -> None:
        """デフォルト機会検出設定"""
        from .enums import TimeHorizon
        
        default_configs = [
            # テクニカル分析ブレイクアウト
            OpportunityConfig(
                rule_id="technical_breakout_topix",
                rule_name="TOPIX テクニカルブレイクアウト",
                opportunity_type=OpportunityType.TECHNICAL_BREAKOUT,
                symbols=["TOPIX", "N225"],
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence_threshold=0.8,
                profit_potential_threshold=5.0,
                check_interval_minutes=15,
            ),
            # モメンタム分析
            OpportunityConfig(
                rule_id="momentum_large_cap",
                rule_name="大型株モメンタム",
                opportunity_type=OpportunityType.MOMENTUM_SIGNAL,
                symbols=["7203", "8306", "9984", "4502"],  # トヨタ、三菱UFJ、SBG、武田
                time_horizon=TimeHorizon.MEDIUM_TERM,
                confidence_threshold=0.75,
                profit_potential_threshold=7.0,
                rsi_oversold=25,
                rsi_overbought=75,
            ),
            # リバーサルパターン
            OpportunityConfig(
                rule_id="reversal_pattern_detection",
                rule_name="反転パターン検出",
                opportunity_type=OpportunityType.REVERSAL_PATTERN,
                symbols=["7182", "6501", "8058"],  # ゆうちょ、日立、三菱商事
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence_threshold=0.7,
                profit_potential_threshold=6.0,
                risk_reward_ratio=2.5,
            ),
            # 出来高異常
            OpportunityConfig(
                rule_id="volume_anomaly_alert",
                rule_name="出来高異常検出",
                opportunity_type=OpportunityType.VOLUME_ANOMALY,
                symbols=["*"],  # 全銘柄
                time_horizon=TimeHorizon.INTRADAY,
                confidence_threshold=0.8,
                volume_spike_threshold=3.0,
                check_interval_minutes=5,
            ),
            # 株価割安検出
            OpportunityConfig(
                rule_id="undervaluation_screening",
                rule_name="割安株スクリーニング",
                opportunity_type=OpportunityType.PRICE_UNDERVALUATION,
                symbols=["*"],
                time_horizon=TimeHorizon.LONG_TERM,
                confidence_threshold=0.85,
                pe_ratio_threshold=12.0,
                profit_potential_threshold=15.0,
            ),
            # 配当機会
            OpportunityConfig(
                rule_id="dividend_opportunity",
                rule_name="高配当投資機会",
                opportunity_type=OpportunityType.DIVIDEND_OPPORTUNITY,
                symbols=["8306", "7182", "9437"],  # 三菱UFJ、ゆうちょ、NTTドコモ
                time_horizon=TimeHorizon.LONG_TERM,
                dividend_yield_threshold=4.0,
                confidence_threshold=0.7,
            ),
            # ボラティリティスクイーズ
            OpportunityConfig(
                rule_id="volatility_squeeze",
                rule_name="ボラティリティスクイーズ",
                opportunity_type=OpportunityType.VOLATILITY_SQUEEZE,
                symbols=["N225", "TOPIX"],
                time_horizon=TimeHorizon.SHORT_TERM,
                confidence_threshold=0.75,
                bollinger_deviation=1.5,
            ),
        ]

        for config in default_configs:
            self.add_opportunity_config(config)

    def add_opportunity_config(self, config: OpportunityConfig) -> None:
        """機会検出設定追加"""
        self.opportunity_configs[config.rule_id] = config
        logger.info(f"投資機会検出設定追加: {config.rule_id}")

    def remove_opportunity_config(self, rule_id: str) -> None:
        """機会検出設定削除"""
        if rule_id in self.opportunity_configs:
            del self.opportunity_configs[rule_id]
            logger.info(f"投資機会検出設定削除: {rule_id}")

    def register_alert_handler(
        self,
        severity: OpportunitySeverity,
        handler: Callable,
    ) -> None:
        """アラートハンドラー登録"""
        self.opportunity_manager.register_alert_handler(severity, handler)

    async def start_monitoring(self) -> None:
        """監視開始"""
        if self._is_running:
            logger.warning("投資機会監視は既に実行中です")
            return

        self._is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("投資機会アラート監視開始")

    async def stop_monitoring(self) -> None:
        """監視停止"""
        self._is_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("投資機会アラート監視停止")

    async def _monitoring_loop(self) -> None:
        """監視ループ"""
        while self._is_running:
            try:
                # 市場状況更新
                await self.detector.update_market_condition()

                # 価格データ更新
                await self.detector.update_price_data()

                # テクニカル指標計算
                await self.detector.calculate_technical_indicators()

                # 機会検出実行
                await self._detect_opportunities()

                # 履歴クリーンアップ
                await self.opportunity_manager.cleanup_opportunity_history()

                # 最短チェック間隔で実行（デフォルト5分）
                min_interval = min(
                    [
                        config.check_interval_minutes
                        for config in self.opportunity_configs.values()
                    ],
                    default=5,
                )
                await asyncio.sleep(min_interval * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"投資機会監視ループエラー: {e}")
                await asyncio.sleep(60)

    async def _detect_opportunities(self) -> None:
        """投資機会検出"""
        # 有効な設定のみを対象に機会検出を実行
        enabled_configs = {
            rule_id: config
            for rule_id, config in self.opportunity_configs.items()
            if config.enabled
        }

        if not enabled_configs:
            return

        # 機会検出実行
        opportunities = await self.detector.detect_opportunities(enabled_configs)

        for opportunity in opportunities:
            await self.opportunity_manager.process_opportunity(opportunity)


    async def get_active_opportunities(
        self,
        symbol: Optional[str] = None,
        opportunity_type: Optional[OpportunityType] = None,
        severity: Optional[OpportunitySeverity] = None,
    ) -> List[InvestmentOpportunity]:
        """アクティブ機会取得"""
        return await self.opportunity_manager.get_active_opportunities(
            symbol, opportunity_type, severity
        )

    async def get_opportunity_summary(self) -> Dict[str, Any]:
        """機会概要取得"""
        active_configs_count = len(
            [
                config
                for config in self.opportunity_configs.values()
                if config.enabled
            ]
        )
        
        return await self.summary_generator.generate_opportunity_summary(
            self.opportunity_manager.opportunities,
            active_configs_count,
            len(self.opportunity_configs),
            self._is_running,
            self.detector.market_condition,
        )

    async def execute_opportunity(
        self,
        opportunity_id: str,
        execution_note: Optional[str] = None,
    ) -> bool:
        """機会実行"""
        return await self.opportunity_manager.execute_opportunity(
            opportunity_id, execution_note
        )