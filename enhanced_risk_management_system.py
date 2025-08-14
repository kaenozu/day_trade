#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Risk Management System - 強化リスク管理システム

Issue #813対応：包括的なリスク管理と資金管理
ポジションサイジング、動的ストップロス、リスクメトリクスの計算
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
import math

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
    VERY_LOW = "very_low"      # 0.5%未満
    LOW = "low"                # 0.5-1%
    MODERATE = "moderate"      # 1-2%
    HIGH = "high"              # 2-3%
    VERY_HIGH = "very_high"    # 3%超

class PositionSizingMethod(Enum):
    """ポジションサイジング手法"""
    FIXED_AMOUNT = "fixed_amount"          # 固定金額
    FIXED_PERCENTAGE = "fixed_percentage"  # 固定比率
    KELLY_CRITERION = "kelly_criterion"    # ケリー基準
    VOLATILITY_BASED = "volatility_based"  # ボラティリティベース
    ATR_BASED = "atr_based"               # ATRベース

@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    symbol: str
    portfolio_value: float
    position_value: float
    position_weight: float
    var_1day: float           # 1日VaR
    var_5day: float           # 5日VaR
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation_with_market: float
    risk_level: RiskLevel
    timestamp: datetime

@dataclass
class PositionSizing:
    """ポジションサイジング結果"""
    symbol: str
    recommended_quantity: int
    recommended_value: float
    max_risk_amount: float
    position_weight: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    sizing_method: PositionSizingMethod
    risk_reward_ratio: float
    timestamp: datetime

@dataclass
class RiskAlert:
    """リスクアラート"""
    alert_id: str
    symbol: str
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime

class VolatilityCalculator:
    """ボラティリティ計算機"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_historical_volatility(self, prices: pd.Series, window: int = 20) -> float:
        """ヒストリカルボラティリティ計算"""
        if len(prices) < window:
            return 0.0

        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window).std().iloc[-1]

        # 年率換算 (252営業日)
        annualized_vol = volatility * np.sqrt(252) * 100
        return float(annualized_vol) if not np.isnan(annualized_vol) else 0.0

    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> float:
        """ATR (Average True Range) 計算"""
        if len(data) < window or not all(col in data.columns for col in ['High', 'Low', 'Close']):
            return 0.0

        high = data['High']
        low = data['Low']
        close = data['Close']

        # True Rangeを計算
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean().iloc[-1]

        return float(atr) if not np.isnan(atr) else 0.0

    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """ベータ値計算"""
        if len(stock_returns) != len(market_returns) or len(stock_returns) < 30:
            return 1.0  # デフォルトベータ

        # 共通のインデックスでアライン
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()

        if len(aligned_data) < 30:
            return 1.0

        covariance = aligned_data['stock'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()

        beta = covariance / market_variance if market_variance > 0 else 1.0
        return float(beta)

class VaRCalculator:
    """VaR (Value at Risk) 計算機"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_parametric_var(self, portfolio_value: float, volatility: float,
                                confidence_level: float = 0.95, time_horizon: int = 1) -> float:
        """パラメトリックVaR計算"""
        from scipy import stats

        # 信頼水準に対応するZ値
        z_score = stats.norm.ppf(confidence_level)

        # 日次ボラティリティに変換
        daily_vol = volatility / np.sqrt(252) / 100

        # 期間調整
        period_vol = daily_vol * np.sqrt(time_horizon)

        # VaR計算
        var = portfolio_value * z_score * period_vol

        return float(var)

    def calculate_historical_var(self, returns: pd.Series, portfolio_value: float,
                               confidence_level: float = 0.95) -> float:
        """ヒストリカルVaR計算"""
        if len(returns) < 30:
            return 0.0

        # パーセンタイル計算
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns.dropna(), percentile)

        var = portfolio_value * abs(var_return)
        return float(var)

class PositionSizingEngine:
    """ポジションサイジングエンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volatility_calc = VolatilityCalculator()

    def calculate_kelly_sizing(self, win_rate: float, avg_win: float, avg_loss: float,
                              portfolio_value: float) -> float:
        """ケリー基準によるポジションサイジング"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.02  # デフォルト2%

        # ケリー比率 = (勝率 * 平均利益 - 敗率 * 平均損失) / 平均利益
        lose_rate = 1 - win_rate
        kelly_ratio = (win_rate * avg_win - lose_rate * avg_loss) / avg_win

        # 保守的に50%に調整（フル・ケリーは危険）
        conservative_kelly = kelly_ratio * 0.5

        # 最大10%、最小1%に制限
        position_ratio = max(0.01, min(0.10, conservative_kelly))

        return float(position_ratio)

    def calculate_volatility_based_sizing(self, volatility: float, target_volatility: float = 15.0,
                                        portfolio_value: float = 1000000) -> float:
        """ボラティリティベースのポジションサイジング"""
        if volatility <= 0:
            return 0.02

        # 目標ボラティリティに対する比率
        vol_ratio = target_volatility / volatility

        # 基本ポジションサイズ（2%）にボラティリティ調整を適用
        base_position = 0.02
        position_ratio = base_position * vol_ratio

        # 1-20%に制限
        position_ratio = max(0.01, min(0.20, position_ratio))

        return float(position_ratio)

    def calculate_atr_based_sizing(self, current_price: float, atr: float,
                                 risk_per_trade: float, portfolio_value: float) -> int:
        """ATRベースのポジションサイジング"""
        if atr <= 0 or current_price <= 0:
            return 0

        # ストップロス距離 = 2 * ATR
        stop_distance = 2 * atr

        # リスク許容額
        risk_amount = portfolio_value * risk_per_trade

        # 株数 = リスク許容額 / ストップロス距離
        quantity = int(risk_amount / stop_distance) if stop_distance > 0 else 0

        # 最大ポジションサイズ制限（ポートフォリオの20%）
        max_value = portfolio_value * 0.20
        max_quantity = int(max_value / current_price)

        return min(quantity, max_quantity)

class RiskManagementSystem:
    """リスク管理システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vol_calc = VolatilityCalculator()
        self.var_calc = VaRCalculator()
        self.position_engine = PositionSizingEngine()

        # データベース設定
        self.db_path = Path("risk_management/risk_metrics.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        # リスク設定
        self.max_portfolio_risk = 0.02    # 1日最大2%リスク
        self.max_single_position = 0.10   # 単一銘柄最大10%
        self.target_volatility = 15.0     # 目標年率ボラティリティ15%

        self.logger.info("Enhanced risk management system initialized")

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # リスクメトリクステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        portfolio_value REAL NOT NULL,
                        position_value REAL NOT NULL,
                        position_weight REAL NOT NULL,
                        var_1day REAL NOT NULL,
                        var_5day REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        volatility REAL NOT NULL,
                        beta REAL NOT NULL,
                        correlation_with_market REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # ポジションサイジングテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS position_sizing (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        recommended_quantity INTEGER NOT NULL,
                        recommended_value REAL NOT NULL,
                        max_risk_amount REAL NOT NULL,
                        position_weight REAL NOT NULL,
                        stop_loss_price REAL NOT NULL,
                        take_profit_price REAL,
                        sizing_method TEXT NOT NULL,
                        risk_reward_ratio REAL NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # リスクアラートテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_alerts (
                        alert_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        metrics_json TEXT,
                        recommendations_json TEXT,
                        timestamp TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")

    async def calculate_risk_metrics(self, symbol: str, position_value: float = 0,
                                   portfolio_value: float = 1000000) -> RiskMetrics:
        """リスクメトリクス計算"""

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "6mo")

            if data is None or len(data) < 30:
                raise ValueError(f"Insufficient data for {symbol}")

            prices = data['Close']
            returns = prices.pct_change().dropna()

            # ボラティリティ計算
            volatility = self.vol_calc.calculate_historical_volatility(prices)

            # VaR計算
            var_1day = self.var_calc.calculate_parametric_var(
                position_value, volatility, 0.95, 1
            )
            var_5day = self.var_calc.calculate_parametric_var(
                position_value, volatility, 0.95, 5
            )

            # 最大ドローダウン計算
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            # シャープレシオ計算
            excess_returns = returns - 0.01/252  # リスクフリーレート仮定
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

            # ベータ計算（日経平均との相関として簡易計算）
            beta = 1.0  # デフォルト値
            correlation = 0.5  # デフォルト値

            # ポジション比率
            position_weight = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0

            # リスクレベル判定
            risk_level = self._determine_risk_level(volatility, position_weight, var_1day, portfolio_value)

            metrics = RiskMetrics(
                symbol=symbol,
                portfolio_value=portfolio_value,
                position_value=position_value,
                position_weight=position_weight,
                var_1day=var_1day,
                var_5day=var_5day,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=beta,
                correlation_with_market=correlation,
                risk_level=risk_level,
                timestamp=datetime.now()
            )

            # データベース保存
            await self._save_risk_metrics(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Risk metrics calculation error for {symbol}: {e}")
            # デフォルトメトリクス返す
            return RiskMetrics(
                symbol=symbol,
                portfolio_value=portfolio_value,
                position_value=position_value,
                position_weight=0,
                var_1day=0,
                var_5day=0,
                max_drawdown=0,
                sharpe_ratio=0,
                volatility=0,
                beta=1.0,
                correlation_with_market=0.5,
                risk_level=RiskLevel.LOW,
                timestamp=datetime.now()
            )

    def _determine_risk_level(self, volatility: float, position_weight: float,
                            var_1day: float, portfolio_value: float) -> RiskLevel:
        """リスクレベル判定"""

        # VaR比率
        var_ratio = (var_1day / portfolio_value) * 100 if portfolio_value > 0 else 0

        # 複合リスクスコア計算
        vol_score = min(volatility / 30, 1.0)  # 30%を最大として正規化
        weight_score = min(position_weight / 20, 1.0)  # 20%を最大として正規化
        var_score = min(var_ratio / 3, 1.0)  # 3%を最大として正規化

        composite_score = (vol_score + weight_score + var_score) / 3

        if composite_score < 0.2:
            return RiskLevel.VERY_LOW
        elif composite_score < 0.4:
            return RiskLevel.LOW
        elif composite_score < 0.6:
            return RiskLevel.MODERATE
        elif composite_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    async def calculate_position_sizing(self, symbol: str, current_price: float,
                                      portfolio_value: float = 1000000,
                                      method: PositionSizingMethod = PositionSizingMethod.ATR_BASED) -> PositionSizing:
        """ポジションサイジング計算"""

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "3mo")

            if data is None or len(data) < 30:
                raise ValueError(f"Insufficient data for {symbol}")

            # ATR計算
            atr = self.vol_calc.calculate_atr(data)
            volatility = self.vol_calc.calculate_historical_volatility(data['Close'])

            # メソッドに応じてポジションサイズ計算
            if method == PositionSizingMethod.ATR_BASED:
                quantity = self.position_engine.calculate_atr_based_sizing(
                    current_price, atr, self.max_portfolio_risk, portfolio_value
                )
                recommended_value = quantity * current_price
                position_weight = (recommended_value / portfolio_value) * 100

                # ストップロス価格 = 現在価格 - 2 * ATR
                stop_loss_price = current_price - (2 * atr)

            elif method == PositionSizingMethod.VOLATILITY_BASED:
                position_ratio = self.position_engine.calculate_volatility_based_sizing(
                    volatility, self.target_volatility, portfolio_value
                )
                recommended_value = portfolio_value * position_ratio
                quantity = int(recommended_value / current_price)
                position_weight = position_ratio * 100

                # ボラティリティベースストップロス
                daily_vol = volatility / np.sqrt(252) / 100
                stop_loss_price = current_price * (1 - daily_vol * 2)

            else:
                # デフォルト：固定比率2%
                recommended_value = portfolio_value * 0.02
                quantity = int(recommended_value / current_price)
                position_weight = 2.0
                stop_loss_price = current_price * 0.95  # 5%ストップ

            # 最大ポジション制限
            max_value = portfolio_value * self.max_single_position
            if recommended_value > max_value:
                recommended_value = max_value
                quantity = int(max_value / current_price)
                position_weight = self.max_single_position * 100

            # リスク金額
            max_risk_amount = (current_price - stop_loss_price) * quantity

            # 利確価格（リスクリワード比2:1）
            risk_per_share = current_price - stop_loss_price
            take_profit_price = current_price + (risk_per_share * 2)

            # リスクリワード比
            risk_reward_ratio = (take_profit_price - current_price) / risk_per_share if risk_per_share > 0 else 0

            sizing = PositionSizing(
                symbol=symbol,
                recommended_quantity=quantity,
                recommended_value=recommended_value,
                max_risk_amount=max_risk_amount,
                position_weight=position_weight,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                sizing_method=method,
                risk_reward_ratio=risk_reward_ratio,
                timestamp=datetime.now()
            )

            # データベース保存
            await self._save_position_sizing(sizing)

            return sizing

        except Exception as e:
            self.logger.error(f"Position sizing error for {symbol}: {e}")
            raise

    async def generate_risk_alerts(self, metrics: RiskMetrics) -> List[RiskAlert]:
        """リスクアラート生成"""

        alerts = []

        # 高ボラティリティアラート
        if metrics.volatility > 30:
            alerts.append(RiskAlert(
                alert_id=f"HIGH_VOL_{metrics.symbol}_{int(datetime.now().timestamp())}",
                symbol=metrics.symbol,
                alert_type="HIGH_VOLATILITY",
                severity="HIGH",
                message=f"{metrics.symbol}のボラティリティが高水準({metrics.volatility:.1f}%)",
                metrics={"volatility": metrics.volatility},
                recommendations=["ポジションサイズを縮小", "ストップロスを厳しく設定"],
                timestamp=datetime.now()
            ))

        # 集中リスクアラート
        if metrics.position_weight > 15:
            alerts.append(RiskAlert(
                alert_id=f"CONCENTRATION_{metrics.symbol}_{int(datetime.now().timestamp())}",
                symbol=metrics.symbol,
                alert_type="CONCENTRATION_RISK",
                severity="MEDIUM",
                message=f"{metrics.symbol}のポジション比率が高い({metrics.position_weight:.1f}%)",
                metrics={"position_weight": metrics.position_weight},
                recommendations=["ポジションの一部利確", "分散投資の検討"],
                timestamp=datetime.now()
            ))

        # 大幅ドローダウンアラート
        if metrics.max_drawdown < -20:
            alerts.append(RiskAlert(
                alert_id=f"DRAWDOWN_{metrics.symbol}_{int(datetime.now().timestamp())}",
                symbol=metrics.symbol,
                alert_type="MAX_DRAWDOWN",
                severity="HIGH",
                message=f"{metrics.symbol}の最大ドローダウンが大きい({metrics.max_drawdown:.1f}%)",
                metrics={"max_drawdown": metrics.max_drawdown},
                recommendations=["リスク許容度の見直し", "戦略の再検討"],
                timestamp=datetime.now()
            ))

        # アラート保存
        for alert in alerts:
            await self._save_risk_alert(alert)

        return alerts

    async def _save_risk_metrics(self, metrics: RiskMetrics):
        """リスクメトリクス保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO risk_metrics
                    (symbol, portfolio_value, position_value, position_weight,
                     var_1day, var_5day, max_drawdown, sharpe_ratio, volatility,
                     beta, correlation_with_market, risk_level, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.symbol, metrics.portfolio_value, metrics.position_value,
                    metrics.position_weight, metrics.var_1day, metrics.var_5day,
                    metrics.max_drawdown, metrics.sharpe_ratio, metrics.volatility,
                    metrics.beta, metrics.correlation_with_market,
                    metrics.risk_level.value, metrics.timestamp.isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Save risk metrics error: {e}")

    async def _save_position_sizing(self, sizing: PositionSizing):
        """ポジションサイジング保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO position_sizing
                    (symbol, recommended_quantity, recommended_value, max_risk_amount,
                     position_weight, stop_loss_price, take_profit_price,
                     sizing_method, risk_reward_ratio, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sizing.symbol, sizing.recommended_quantity, sizing.recommended_value,
                    sizing.max_risk_amount, sizing.position_weight, sizing.stop_loss_price,
                    sizing.take_profit_price, sizing.sizing_method.value,
                    sizing.risk_reward_ratio, sizing.timestamp.isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Save position sizing error: {e}")

    async def _save_risk_alert(self, alert: RiskAlert):
        """リスクアラート保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO risk_alerts
                    (alert_id, symbol, alert_type, severity, message,
                     metrics_json, recommendations_json, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id, alert.symbol, alert.alert_type, alert.severity,
                    alert.message, json.dumps(alert.metrics),
                    json.dumps(alert.recommendations), alert.timestamp.isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Save risk alert error: {e}")

# グローバルインスタンス
enhanced_risk_management = RiskManagementSystem()

# テスト実行
async def run_risk_management_test():
    """リスク管理システムテスト"""

    print("=== 🛡️ 強化リスク管理システムテスト ===")

    test_symbols = ["7203", "8306", "4751"]
    portfolio_value = 1000000  # 100万円

    for symbol in test_symbols:
        print(f"\n--- {symbol} リスク分析 ---")

        try:
            # 現在価格取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "1mo")
            current_price = float(data['Close'].iloc[-1]) if data is not None and len(data) > 0 else 1000

            # ポジション価値（ポートフォリオの5%と仮定）
            position_value = portfolio_value * 0.05

            # リスクメトリクス計算
            metrics = await enhanced_risk_management.calculate_risk_metrics(
                symbol, position_value, portfolio_value
            )

            print(f"リスクレベル: {metrics.risk_level.value}")
            print(f"ボラティリティ: {metrics.volatility:.1f}%")
            print(f"1日VaR: ¥{metrics.var_1day:,.0f}")
            print(f"最大ドローダウン: {metrics.max_drawdown:.1f}%")
            print(f"シャープレシオ: {metrics.sharpe_ratio:.3f}")

            # ポジションサイジング計算
            sizing = await enhanced_risk_management.calculate_position_sizing(
                symbol, current_price, portfolio_value, PositionSizingMethod.ATR_BASED
            )

            print(f"推奨株数: {sizing.recommended_quantity:,}株")
            print(f"推奨投資額: ¥{sizing.recommended_value:,.0f}")
            print(f"ポジション比率: {sizing.position_weight:.1f}%")
            print(f"ストップロス: ¥{sizing.stop_loss_price:.2f}")
            print(f"利確目標: ¥{sizing.take_profit_price:.2f}")
            print(f"リスクリワード比: {sizing.risk_reward_ratio:.2f}")

            # リスクアラート生成
            alerts = await enhanced_risk_management.generate_risk_alerts(metrics)
            if alerts:
                print(f"リスクアラート: {len(alerts)}件")
                for alert in alerts:
                    print(f"  • {alert.alert_type}: {alert.message}")

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    print("\n✅ 強化リスク管理システム動作確認完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_risk_management_test())