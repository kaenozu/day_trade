#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Risk Management System - 高度リスク管理システム

実践的リスク管理・動的ストップロス実装
Issue #797実装：実践的リスク管理システム
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class RiskLevel(Enum):
    """リスクレベル"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class PositionType(Enum):
    """ポジションタイプ"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class RiskEvent(Enum):
    """リスクイベント"""
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    MARGIN_CALL = "margin_call"

@dataclass
class Position:
    """ポジション情報"""
    symbol: str
    position_type: PositionType
    entry_price: float
    current_price: float
    quantity: int
    entry_time: datetime
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    risk_amount: float = 0.0
    unrealized_pnl: float = 0.0

@dataclass
class RiskMetrics:
    """リスク指標"""
    portfolio_value: float
    total_exposure: float
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation_risk: float

@dataclass
class RiskAlert:
    """リスクアラート"""
    alert_id: str
    risk_event: RiskEvent
    severity: RiskLevel
    symbol: str
    description: str
    recommendation: str
    timestamp: datetime
    auto_action_taken: bool = False

@dataclass
class RiskLimits:
    """リスク制限"""
    max_position_size_pct: float = 10.0  # ポートフォリオの10%まで
    max_sector_exposure_pct: float = 30.0  # セクター集中度30%まで
    max_single_loss_pct: float = 2.0  # 単一取引損失2%まで
    max_daily_loss_pct: float = 5.0  # 日次損失5%まで
    max_var_pct: float = 3.0  # VaR 3%まで
    min_liquidity_ratio: float = 0.2  # 流動性比率20%以上
    max_leverage: float = 2.0  # レバレッジ2倍まで

class AdvancedRiskManagementSystem:
    """高度リスク管理システム"""

    def __init__(self, initial_capital: float = 1000000):
        self.logger = logging.getLogger(__name__)

        # 初期設定
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.risk_limits = RiskLimits()
        self.risk_alerts: List[RiskAlert] = []

        # データベース設定
        self.db_path = Path("ml_models_data/risk_management.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # リスク履歴
        self.risk_history: List[RiskMetrics] = []
        self.pnl_history: List[float] = []

        # 動的ストップロス設定
        self.trailing_stop_configs = {
            "default": {
                "initial_stop_pct": 0.02,  # 初期ストップ2%
                "trailing_pct": 0.015,     # トレーリング1.5%
                "acceleration_factor": 0.02,  # 加速係数
                "max_acceleration": 0.2    # 最大加速
            },
            "volatile": {
                "initial_stop_pct": 0.03,
                "trailing_pct": 0.025,
                "acceleration_factor": 0.025,
                "max_acceleration": 0.25
            },
            "stable": {
                "initial_stop_pct": 0.015,
                "trailing_pct": 0.01,
                "acceleration_factor": 0.015,
                "max_acceleration": 0.15
            }
        }

        self.logger.info("Advanced risk management system initialized")

    async def calculate_position_risk(self, symbol: str, position_type: PositionType,
                                    entry_price: float, quantity: int) -> Dict[str, float]:
        """ポジションリスク計算"""

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "3mo")

            if data is None or len(data) < 30:
                raise ValueError("リスク計算に十分なデータがありません")

            # ボラティリティ計算
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # 年率ボラティリティ

            # VaR計算
            position_value = entry_price * quantity
            var_95 = np.percentile(returns, 5) * position_value
            var_99 = np.percentile(returns, 1) * position_value

            # Expected Shortfall (CVaR)
            es_95 = returns[returns <= np.percentile(returns, 5)].mean() * position_value

            # 最大ドローダウン計算
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # 流動性リスク（簡易版）
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            liquidity_ratio = min(1.0, avg_volume / (quantity * 10))  # 10日で処分可能

            # 相関リスク（市場との相関）
            # 簡易版として、トレンドの強さで代替
            trend_strength = abs(returns.rolling(20).mean().iloc[-1])
            correlation_risk = min(1.0, trend_strength * 10)

            risk_metrics = {
                'volatility': volatility,
                'var_95': abs(var_95),
                'var_99': abs(var_99),
                'expected_shortfall': abs(es_95),
                'max_drawdown': abs(max_drawdown),
                'liquidity_ratio': liquidity_ratio,
                'correlation_risk': correlation_risk,
                'position_value': position_value
            }

            return risk_metrics

        except Exception as e:
            self.logger.error(f"リスク計算エラー {symbol}: {e}")
            # デフォルトリスク値
            return {
                'volatility': 0.25,
                'var_95': entry_price * quantity * 0.05,
                'var_99': entry_price * quantity * 0.08,
                'expected_shortfall': entry_price * quantity * 0.1,
                'max_drawdown': 0.15,
                'liquidity_ratio': 0.5,
                'correlation_risk': 0.5,
                'position_value': entry_price * quantity
            }

    async def calculate_dynamic_stop_loss(self, symbol: str, position: Position,
                                        current_price: float) -> Tuple[float, float]:
        """動的ストップロス計算"""

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "1mo")

            if data is None or len(data) < 20:
                # デフォルト設定
                config = self.trailing_stop_configs["default"]
            else:
                # ボラティリティベースの設定選択
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std()

                if volatility > 0.03:
                    config = self.trailing_stop_configs["volatile"]
                elif volatility < 0.015:
                    config = self.trailing_stop_configs["stable"]
                else:
                    config = self.trailing_stop_configs["default"]

                # ATR計算
                high = data['High']
                low = data['Low']
                close = data['Close']

                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]

                # ATRベースの調整
                atr_multiplier = atr / current_price
                config["initial_stop_pct"] *= (1 + atr_multiplier)
                config["trailing_pct"] *= (1 + atr_multiplier)

            # 初期ストップロス
            if position.position_type == PositionType.LONG:
                initial_stop = position.entry_price * (1 - config["initial_stop_pct"])

                # トレーリングストップ計算
                if position.trailing_stop is None:
                    trailing_stop = max(initial_stop, current_price * (1 - config["trailing_pct"]))
                else:
                    # 価格が上昇した場合のみストップロスを引き上げ
                    new_trailing = current_price * (1 - config["trailing_pct"])
                    trailing_stop = max(position.trailing_stop, new_trailing)

            else:  # SHORT
                initial_stop = position.entry_price * (1 + config["initial_stop_pct"])

                if position.trailing_stop is None:
                    trailing_stop = min(initial_stop, current_price * (1 + config["trailing_pct"]))
                else:
                    # 価格が下降した場合のみストップロスを引き下げ
                    new_trailing = current_price * (1 + config["trailing_pct"])
                    trailing_stop = min(position.trailing_stop, new_trailing)

            return initial_stop, trailing_stop

        except Exception as e:
            self.logger.error(f"動的ストップロス計算エラー {symbol}: {e}")
            # フォールバック
            if position.position_type == PositionType.LONG:
                return (position.entry_price * 0.98, current_price * 0.985)
            else:
                return (position.entry_price * 1.02, current_price * 1.015)

    async def assess_portfolio_risk(self) -> RiskMetrics:
        """ポートフォリオリスク評価"""

        if not self.positions:
            return RiskMetrics(
                portfolio_value=self.current_capital,
                total_exposure=0,
                value_at_risk_95=0,
                value_at_risk_99=0,
                expected_shortfall=0,
                max_drawdown=0,
                sharpe_ratio=0,
                volatility=0,
                beta=0,
                correlation_risk=0
            )

        try:
            # ポートフォリオ価値計算
            total_position_value = 0
            total_var_95 = 0
            total_var_99 = 0
            total_es = 0

            for symbol, position in self.positions.items():
                position_value = position.current_price * position.quantity
                total_position_value += position_value

                # 個別ポジションリスク取得
                risk_metrics = await self.calculate_position_risk(
                    symbol, position.position_type, position.entry_price, position.quantity
                )

                total_var_95 += risk_metrics['var_95']
                total_var_99 += risk_metrics['var_99']
                total_es += risk_metrics['expected_shortfall']

            portfolio_value = self.current_capital + sum(pos.unrealized_pnl for pos in self.positions.values())

            # ポートフォリオレベルの指標計算
            if len(self.pnl_history) > 0:
                returns = np.array(self.pnl_history)
                portfolio_volatility = np.std(returns) * np.sqrt(252)

                if portfolio_volatility > 0:
                    sharpe_ratio = np.mean(returns) / portfolio_volatility * np.sqrt(252)
                else:
                    sharpe_ratio = 0

                # 最大ドローダウン
                cumulative_pnl = np.cumsum(returns)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = cumulative_pnl - running_max
                max_drawdown = np.min(drawdown) / self.initial_capital
            else:
                portfolio_volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0

            # 相関リスク（簡易版）
            correlation_risk = min(1.0, len(self.positions) / 10)  # 分散度の逆

            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_position_value,
                value_at_risk_95=total_var_95,
                value_at_risk_99=total_var_99,
                expected_shortfall=total_es,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=portfolio_volatility,
                beta=1.0,  # 簡易版では1.0
                correlation_risk=correlation_risk
            )

            self.risk_history.append(risk_metrics)
            return risk_metrics

        except Exception as e:
            self.logger.error(f"ポートフォリオリスク評価エラー: {e}")
            # デフォルト値
            return RiskMetrics(
                portfolio_value=self.current_capital,
                total_exposure=0,
                value_at_risk_95=0,
                value_at_risk_99=0,
                expected_shortfall=0,
                max_drawdown=0,
                sharpe_ratio=0,
                volatility=0,
                beta=1.0,
                correlation_risk=0
            )

    async def check_risk_limits(self, risk_metrics: RiskMetrics) -> List[RiskAlert]:
        """リスク制限チェック"""

        alerts = []

        # 1. ポジションサイズチェック
        for symbol, position in self.positions.items():
            position_pct = (position.current_price * position.quantity) / risk_metrics.portfolio_value * 100

            if position_pct > self.risk_limits.max_position_size_pct:
                alert = RiskAlert(
                    alert_id=f"pos_size_{symbol}_{int(datetime.now().timestamp())}",
                    risk_event=RiskEvent.POSITION_SIZE_EXCEEDED,
                    severity=RiskLevel.HIGH,
                    symbol=symbol,
                    description=f"ポジションサイズ超過: {position_pct:.1f}% (制限: {self.risk_limits.max_position_size_pct}%)",
                    recommendation=f"ポジションサイズを{self.risk_limits.max_position_size_pct}%以下に削減",
                    timestamp=datetime.now()
                )
                alerts.append(alert)

        # 2. VaRチェック
        var_pct = risk_metrics.value_at_risk_95 / risk_metrics.portfolio_value * 100
        if var_pct > self.risk_limits.max_var_pct:
            alert = RiskAlert(
                alert_id=f"var_limit_{int(datetime.now().timestamp())}",
                risk_event=RiskEvent.VOLATILITY_SPIKE,
                severity=RiskLevel.HIGH,
                symbol="PORTFOLIO",
                description=f"VaR制限超過: {var_pct:.1f}% (制限: {self.risk_limits.max_var_pct}%)",
                recommendation="ポジション縮小またはヘッジ実行",
                timestamp=datetime.now()
            )
            alerts.append(alert)

        # 3. 日次損失チェック
        if len(self.pnl_history) > 0:
            daily_pnl_pct = self.pnl_history[-1] / self.initial_capital * 100
            if daily_pnl_pct < -self.risk_limits.max_daily_loss_pct:
                alert = RiskAlert(
                    alert_id=f"daily_loss_{int(datetime.now().timestamp())}",
                    risk_event=RiskEvent.MARGIN_CALL,
                    severity=RiskLevel.VERY_HIGH,
                    symbol="PORTFOLIO",
                    description=f"日次損失制限超過: {daily_pnl_pct:.1f}% (制限: -{self.risk_limits.max_daily_loss_pct}%)",
                    recommendation="すべてのポジション緊急見直し",
                    timestamp=datetime.now()
                )
                alerts.append(alert)

        # 4. 最大ドローダウンチェック
        if abs(risk_metrics.max_drawdown) > 0.1:  # 10%以上のドローダウン
            alert = RiskAlert(
                alert_id=f"drawdown_{int(datetime.now().timestamp())}",
                risk_event=RiskEvent.LIQUIDITY_CRISIS,
                severity=RiskLevel.HIGH,
                symbol="PORTFOLIO",
                description=f"大幅ドローダウン: {risk_metrics.max_drawdown*100:.1f}%",
                recommendation="リスク管理の抜本的見直し",
                timestamp=datetime.now()
            )
            alerts.append(alert)

        # 5. ボラティリティスパイクチェック
        if risk_metrics.volatility > 0.3:  # 年率30%以上のボラティリティ
            alert = RiskAlert(
                alert_id=f"volatility_{int(datetime.now().timestamp())}",
                risk_event=RiskEvent.VOLATILITY_SPIKE,
                severity=RiskLevel.MODERATE,
                symbol="PORTFOLIO",
                description=f"高ボラティリティ: {risk_metrics.volatility*100:.1f}%",
                recommendation="ポジション調整を検討",
                timestamp=datetime.now()
            )
            alerts.append(alert)

        self.risk_alerts.extend(alerts)
        return alerts

    async def execute_risk_controls(self, alerts: List[RiskAlert]) -> Dict[str, bool]:
        """リスク統制実行"""

        actions_taken = {}

        for alert in alerts:
            action_key = f"{alert.risk_event.value}_{alert.symbol}"

            try:
                if alert.risk_event == RiskEvent.POSITION_SIZE_EXCEEDED:
                    # ポジションサイズ削減
                    success = await self._reduce_position_size(alert.symbol)
                    actions_taken[action_key] = success

                elif alert.risk_event == RiskEvent.VOLATILITY_SPIKE:
                    # ストップロス強化
                    success = await self._tighten_stop_loss(alert.symbol)
                    actions_taken[action_key] = success

                elif alert.risk_event == RiskEvent.MARGIN_CALL:
                    # 緊急ポジション縮小
                    success = await self._emergency_position_reduction()
                    actions_taken[action_key] = success

                elif alert.risk_event == RiskEvent.LIQUIDITY_CRISIS:
                    # 流動性確保
                    success = await self._ensure_liquidity()
                    actions_taken[action_key] = success

                else:
                    # その他のアラートは監視のみ
                    actions_taken[action_key] = True

                # アラートに自動対応フラグ設定
                alert.auto_action_taken = actions_taken[action_key]

            except Exception as e:
                self.logger.error(f"リスク統制実行エラー {action_key}: {e}")
                actions_taken[action_key] = False

        return actions_taken

    async def _reduce_position_size(self, symbol: str) -> bool:
        """ポジションサイズ削減"""

        try:
            if symbol in self.positions:
                position = self.positions[symbol]
                # ポジションサイズを20%削減
                reduction_qty = int(position.quantity * 0.2)
                if reduction_qty > 0:
                    position.quantity -= reduction_qty
                    self.logger.info(f"ポジションサイズ削減: {symbol} -{reduction_qty}株")
                    return True
            return False

        except Exception as e:
            self.logger.error(f"ポジションサイズ削減エラー {symbol}: {e}")
            return False

    async def _tighten_stop_loss(self, symbol: str) -> bool:
        """ストップロス強化"""

        try:
            if symbol in self.positions:
                position = self.positions[symbol]

                # ストップロスを5%厳しく設定
                if position.position_type == PositionType.LONG:
                    position.stop_loss *= 1.05  # より高いストップロス
                else:
                    position.stop_loss *= 0.95  # より低いストップロス

                self.logger.info(f"ストップロス強化: {symbol} -> {position.stop_loss}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"ストップロス強化エラー {symbol}: {e}")
            return False

    async def _emergency_position_reduction(self) -> bool:
        """緊急ポジション縮小"""

        try:
            reduced_positions = 0

            for symbol, position in self.positions.items():
                # 全ポジションを50%削減
                reduction_qty = int(position.quantity * 0.5)
                if reduction_qty > 0:
                    position.quantity -= reduction_qty
                    reduced_positions += 1

            self.logger.warning(f"緊急ポジション縮小: {reduced_positions}銘柄で50%削減")
            return reduced_positions > 0

        except Exception as e:
            self.logger.error(f"緊急ポジション縮小エラー: {e}")
            return False

    async def _ensure_liquidity(self) -> bool:
        """流動性確保"""

        try:
            # 最も流動性の低いポジションから順に削減
            # 簡易版では全ポジション25%削減
            for symbol, position in self.positions.items():
                reduction_qty = int(position.quantity * 0.25)
                if reduction_qty > 0:
                    position.quantity -= reduction_qty

            self.logger.info("流動性確保: 全ポジション25%削減")
            return True

        except Exception as e:
            self.logger.error(f"流動性確保エラー: {e}")
            return False

    async def run_comprehensive_risk_assessment(self, symbols: List[str]) -> Dict[str, Any]:
        """包括的リスク評価実行"""

        print("=== 🛡️ 包括的リスク管理システム評価 ===")

        # 1. ダミーポジション設定（テスト用）
        await self._setup_test_positions(symbols)

        # 2. ポートフォリオリスク評価
        print("\n📊 ポートフォリオリスク評価中...")
        risk_metrics = await self.assess_portfolio_risk()

        # 3. リスク制限チェック
        print("🚨 リスク制限チェック中...")
        risk_alerts = await self.check_risk_limits(risk_metrics)

        # 4. 動的ストップロス計算
        print("⚡ 動的ストップロス計算中...")
        stop_loss_data = {}
        for symbol in self.positions:
            position = self.positions[symbol]
            initial_stop, trailing_stop = await self.calculate_dynamic_stop_loss(
                symbol, position, position.current_price
            )
            stop_loss_data[symbol] = {
                'initial_stop': initial_stop,
                'trailing_stop': trailing_stop,
                'current_price': position.current_price
            }

        # 5. リスク統制実行
        print("🔧 リスク統制実行中...")
        control_actions = await self.execute_risk_controls(risk_alerts)

        # 結果表示
        await self._display_risk_assessment_report(risk_metrics, risk_alerts, stop_loss_data, control_actions)

        # 結果保存
        await self._save_risk_assessment(risk_metrics, risk_alerts)

        return {
            'risk_metrics': risk_metrics,
            'risk_alerts': risk_alerts,
            'stop_loss_data': stop_loss_data,
            'control_actions': control_actions
        }

    async def _setup_test_positions(self, symbols: List[str]):
        """テスト用ポジション設定"""

        try:
            for i, symbol in enumerate(symbols[:5]):  # 最大5銘柄
                # データ取得
                from real_data_provider_v2 import real_data_provider
                data = await real_data_provider.get_stock_data(symbol, "1mo")

                if data is not None and len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                    entry_price = current_price * (0.95 + np.random.random() * 0.1)  # エントリー価格を適当に設定
                    quantity = int(self.initial_capital * 0.1 / current_price)  # 10%の資金でポジション

                    position = Position(
                        symbol=symbol,
                        position_type=PositionType.LONG,
                        entry_price=entry_price,
                        current_price=current_price,
                        quantity=quantity,
                        entry_time=datetime.now() - timedelta(days=np.random.randint(1, 30)),
                        stop_loss=entry_price * 0.98,
                        take_profit=entry_price * 1.05,
                        unrealized_pnl=(current_price - entry_price) * quantity
                    )

                    self.positions[symbol] = position

        except Exception as e:
            self.logger.error(f"テストポジション設定エラー: {e}")

    async def _display_risk_assessment_report(self, risk_metrics: RiskMetrics,
                                            risk_alerts: List[RiskAlert],
                                            stop_loss_data: Dict[str, Any],
                                            control_actions: Dict[str, bool]):
        """リスク評価レポート表示"""

        print(f"\n" + "=" * 80)
        print(f"🛡️ リスク管理システム評価レポート")
        print(f"=" * 80)

        # ポートフォリオ概要
        print(f"\n📊 ポートフォリオ概要:")
        print(f"  総資産価値: ¥{risk_metrics.portfolio_value:,.0f}")
        print(f"  総エクスポージャー: ¥{risk_metrics.total_exposure:,.0f}")
        print(f"  レバレッジ: {risk_metrics.total_exposure/risk_metrics.portfolio_value:.2f}x")
        print(f"  ポジション数: {len(self.positions)}銘柄")

        # リスク指標
        print(f"\n📈 リスク指標:")
        print(f"  VaR (95%): ¥{risk_metrics.value_at_risk_95:,.0f} ({risk_metrics.value_at_risk_95/risk_metrics.portfolio_value*100:.1f}%)")
        print(f"  VaR (99%): ¥{risk_metrics.value_at_risk_99:,.0f} ({risk_metrics.value_at_risk_99/risk_metrics.portfolio_value*100:.1f}%)")
        print(f"  Expected Shortfall: ¥{risk_metrics.expected_shortfall:,.0f}")
        print(f"  最大ドローダウン: {risk_metrics.max_drawdown*100:.1f}%")
        print(f"  ポートフォリオボラティリティ: {risk_metrics.volatility*100:.1f}%")
        print(f"  シャープレシオ: {risk_metrics.sharpe_ratio:.2f}")

        # 動的ストップロス
        print(f"\n⚡ 動的ストップロス:")
        print(f"{'銘柄':<8} {'現在価格':<10} {'初期SL':<10} {'トレーリング':<10} {'SL率':<8}")
        print(f"-" * 50)
        for symbol, data in stop_loss_data.items():
            sl_pct = (1 - data['trailing_stop'] / data['current_price']) * 100
            print(f"{symbol:<8} {data['current_price']:>9.0f} {data['initial_stop']:>9.0f} "
                  f"{data['trailing_stop']:>9.0f} {sl_pct:>6.1f}%")

        # リスクアラート
        if risk_alerts:
            print(f"\n🚨 リスクアラート: {len(risk_alerts)}件")
            for alert in risk_alerts[:5]:  # 上位5件表示
                severity_emoji = {
                    RiskLevel.VERY_LOW: "🟢",
                    RiskLevel.LOW: "🟡",
                    RiskLevel.MODERATE: "🟠",
                    RiskLevel.HIGH: "🔴",
                    RiskLevel.VERY_HIGH: "💀",
                    RiskLevel.EXTREME: "☠️"
                }
                print(f"  {severity_emoji.get(alert.severity, '❓')} {alert.description}")
                print(f"    推奨: {alert.recommendation}")
        else:
            print(f"\n✅ リスクアラートなし")

        # 統制アクション
        successful_actions = sum(control_actions.values())
        total_actions = len(control_actions)

        print(f"\n🔧 リスク統制:")
        print(f"  実行されたアクション: {successful_actions}/{total_actions}")

        if control_actions:
            for action, success in control_actions.items():
                status = "✅" if success else "❌"
                print(f"    {status} {action}")

        # 総合評価
        overall_risk_score = self._calculate_overall_risk_score(risk_metrics, len(risk_alerts))

        print(f"\n🎯 総合リスク評価:")
        if overall_risk_score >= 85:
            print(f"  🟢 LOW RISK ({overall_risk_score:.1f}/100): 安全なリスクレベル")
        elif overall_risk_score >= 70:
            print(f"  🟡 MODERATE RISK ({overall_risk_score:.1f}/100): 注意深い監視が必要")
        elif overall_risk_score >= 50:
            print(f"  🟠 HIGH RISK ({overall_risk_score:.1f}/100): リスク削減が必要")
        else:
            print(f"  🔴 VERY HIGH RISK ({overall_risk_score:.1f}/100): 緊急対応が必要")

        print(f"=" * 80)

    def _calculate_overall_risk_score(self, risk_metrics: RiskMetrics, alert_count: int) -> float:
        """総合リスクスコア計算"""

        score = 100

        # VaRペナルティ
        var_pct = risk_metrics.value_at_risk_95 / risk_metrics.portfolio_value * 100
        if var_pct > 5:
            score -= (var_pct - 5) * 5

        # ボラティリティペナルティ
        if risk_metrics.volatility > 0.25:
            score -= (risk_metrics.volatility - 0.25) * 100

        # ドローダウンペナルティ
        if abs(risk_metrics.max_drawdown) > 0.1:
            score -= (abs(risk_metrics.max_drawdown) - 0.1) * 200

        # アラートペナルティ
        score -= alert_count * 10

        # 分散ボーナス
        if len(self.positions) >= 3:
            score += 5

        return max(0, min(100, score))

    async def _save_risk_assessment(self, risk_metrics: RiskMetrics, risk_alerts: List[RiskAlert]):
        """リスク評価結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # テーブル作成
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_assessments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        assessment_date TEXT,
                        portfolio_value REAL,
                        total_exposure REAL,
                        var_95 REAL,
                        var_99 REAL,
                        max_drawdown REAL,
                        volatility REAL,
                        sharpe_ratio REAL,
                        alert_count INTEGER,
                        risk_score REAL,
                        created_at TEXT
                    )
                ''')

                # リスク評価保存
                risk_score = self._calculate_overall_risk_score(risk_metrics, len(risk_alerts))

                cursor.execute('''
                    INSERT INTO risk_assessments
                    (assessment_date, portfolio_value, total_exposure, var_95, var_99,
                     max_drawdown, volatility, sharpe_ratio, alert_count, risk_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    risk_metrics.portfolio_value,
                    risk_metrics.total_exposure,
                    risk_metrics.value_at_risk_95,
                    risk_metrics.value_at_risk_99,
                    risk_metrics.max_drawdown,
                    risk_metrics.volatility,
                    risk_metrics.sharpe_ratio,
                    len(risk_alerts),
                    risk_score,
                    datetime.now().isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"リスク評価保存エラー: {e}")

# グローバルインスタンス
risk_management_system = AdvancedRiskManagementSystem()

# テスト実行
async def test_advanced_risk_management():
    """高度リスク管理システムテスト"""

    print("=== 高度リスク管理システムテスト ===")

    test_symbols = ["7203", "8306", "4751", "6861", "9984"]

    # 包括的リスク評価実行
    assessment_results = await risk_management_system.run_comprehensive_risk_assessment(test_symbols)

    print(f"\n✅ 高度リスク管理システムテスト完了")

    return assessment_results

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_advanced_risk_management())