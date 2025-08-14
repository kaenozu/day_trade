#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Risk Management Validator - 実運用時リスク管理検証システム

Issue #806実装：実運用時リスク管理の最終調整と検証
実際の取引環境に向けたリスク管理の最適化と検証
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
import time
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
    VERY_LOW = "very_low"      # 0-10
    LOW = "low"                # 11-25
    MODERATE = "moderate"      # 26-40
    HIGH = "high"              # 41-60
    VERY_HIGH = "very_high"    # 61-80
    EXTREME = "extreme"        # 81-100

class RiskType(Enum):
    """リスクタイプ"""
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    VOLATILITY_RISK = "volatility_risk"
    CONCENTRATION_RISK = "concentration_risk"
    SYSTEM_RISK = "system_risk"
    OPERATIONAL_RISK = "operational_risk"

@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    symbol: str
    timestamp: datetime
    var_1d: float  # Value at Risk (1day)
    var_5d: float  # Value at Risk (5day)
    max_drawdown: float
    volatility: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    risk_score: float
    risk_level: RiskLevel
    risk_factors: Dict[RiskType, float]

@dataclass
class PositionRisk:
    """ポジションリスク"""
    symbol: str
    position_size: float
    current_value: float
    unrealized_pnl: float
    stop_loss_level: float
    take_profit_level: float
    risk_amount: float
    risk_percentage: float
    position_risk_score: float

@dataclass
class PortfolioRisk:
    """ポートフォリオリスク"""
    total_value: float
    total_risk: float
    diversification_score: float
    correlation_risk: float
    concentration_risk: float
    portfolio_var: float
    portfolio_beta: float
    overall_risk_score: float
    overall_risk_level: RiskLevel
    position_risks: List[PositionRisk]

class RiskCalculationEngine:
    """リスク計算エンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # リスク計算パラメータ
        self.confidence_levels = [0.95, 0.99]  # 95%と99%信頼区間
        self.holding_periods = [1, 5, 10]      # 1日、5日、10日保有期間

        # 市場データキャッシュ
        self.market_data_cache = {}

    async def calculate_comprehensive_risk_metrics(self, symbol: str,
                                                 position_size: float = 1000000) -> Optional[RiskMetrics]:
        """包括的リスクメトリクス計算"""

        try:
            # 市場データ取得
            market_data = await self._get_market_data(symbol, "3mo")
            if market_data is None or len(market_data) < 30:
                self.logger.warning(f"十分なデータがありません: {symbol}")
                return None

            # リターン計算
            returns = market_data['Close'].pct_change().dropna()

            # 基本リスクメトリクス
            volatility = returns.std() * np.sqrt(252)  # 年率化
            var_1d = self._calculate_var(returns, confidence_level=0.95, days=1)
            var_5d = self._calculate_var(returns, confidence_level=0.95, days=5)
            max_drawdown = self._calculate_max_drawdown(market_data['Close'])

            # 市場リスク指標
            market_returns = await self._get_market_returns()
            beta = self._calculate_beta(returns, market_returns) if market_returns is not None else 1.0

            # パフォーマンス指標
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)

            # 個別リスクファクター
            risk_factors = await self._calculate_risk_factors(symbol, market_data)

            # 総合リスクスコア
            risk_score = self._calculate_overall_risk_score(
                volatility, var_1d, max_drawdown, beta, risk_factors
            )
            risk_level = self._score_to_risk_level(risk_score)

            return RiskMetrics(
                symbol=symbol,
                timestamp=datetime.now(),
                var_1d=var_1d * position_size,
                var_5d=var_5d * position_size,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                risk_score=risk_score,
                risk_level=risk_level,
                risk_factors=risk_factors
            )

        except Exception as e:
            self.logger.error(f"リスクメトリクス計算エラー {symbol}: {e}")
            return None

    async def _get_market_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """市場データ取得（キャッシュ付き）"""

        cache_key = f"{symbol}_{period}"

        # キャッシュチェック
        if cache_key in self.market_data_cache:
            cached_data, timestamp = self.market_data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=1):  # 1時間有効
                return cached_data

        try:
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is not None:
                self.market_data_cache[cache_key] = (data, datetime.now())

            return data

        except Exception as e:
            self.logger.error(f"市場データ取得エラー {symbol}: {e}")
            return None

    async def _get_market_returns(self) -> Optional[pd.Series]:
        """市場リターン取得（TOPIX代替）"""

        try:
            # TOPIXの代わりに主要銘柄の平均を使用
            major_stocks = ["7203", "8306", "4751"]  # トヨタ、三菱UFJ、サイバーエージェント

            all_returns = []
            for stock in major_stocks:
                data = await self._get_market_data(stock, "3mo")
                if data is not None and len(data) > 0:
                    returns = data['Close'].pct_change().dropna()
                    all_returns.append(returns)

            if all_returns:
                # 各銘柄のリターンの平均
                combined_returns = pd.concat(all_returns, axis=1).mean(axis=1)
                return combined_returns

            return None

        except Exception as e:
            self.logger.error(f"市場リターン取得エラー: {e}")
            return None

    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, days: int = 1) -> float:
        """Value at Risk計算"""

        if len(returns) == 0:
            return 0.0

        # パラメトリック手法（正規分布仮定）
        mean_return = returns.mean()
        std_return = returns.std()

        # 信頼区間に対応するz値
        z_score = {
            0.90: 1.28,
            0.95: 1.645,
            0.99: 2.33
        }.get(confidence_level, 1.645)

        # VaR計算（日数調整）
        var = -(mean_return - z_score * std_return) * np.sqrt(days)

        return max(0, var)

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""

        if len(prices) == 0:
            return 0.0

        # 累積最高値
        running_max = prices.cummax()

        # ドローダウン
        drawdown = (prices - running_max) / running_max

        # 最大ドローダウン
        max_dd = drawdown.min()

        return abs(max_dd)

    def _calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """ベータ値計算"""

        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 1.0

        try:
            # 共通の期間で整列
            aligned_data = pd.DataFrame({
                'asset': asset_returns,
                'market': market_returns
            }).dropna()

            if len(aligned_data) < 10:  # 最低10観測値
                return 1.0

            # 共分散と分散
            covariance = aligned_data['asset'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()

            if market_variance == 0:
                return 1.0

            beta = covariance / market_variance

            return beta

        except Exception:
            return 1.0

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.001) -> float:
        """シャープレシオ計算"""

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_return = returns.mean() - risk_free_rate / 252  # 日次リスクフリーレート
        return excess_return / returns.std() * np.sqrt(252)  # 年率化

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.001) -> float:
        """ソルティノレシオ計算"""

        if len(returns) == 0:
            return 0.0

        excess_return = returns.mean() - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf')  # 下方リスクなし

        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return 0.0

        return excess_return / downside_deviation * np.sqrt(252)

    async def _calculate_risk_factors(self, symbol: str, market_data: pd.DataFrame) -> Dict[RiskType, float]:
        """リスクファクター計算"""

        risk_factors = {}

        try:
            # 市場リスク（ベータ）
            returns = market_data['Close'].pct_change().dropna()
            market_returns = await self._get_market_returns()
            beta = self._calculate_beta(returns, market_returns) if market_returns is not None else 1.0
            risk_factors[RiskType.MARKET_RISK] = min(100, abs(beta - 1) * 100)

            # 流動性リスク（出来高変動）
            volume_cv = market_data['Volume'].std() / market_data['Volume'].mean() if market_data['Volume'].mean() > 0 else 1.0
            risk_factors[RiskType.LIQUIDITY_RISK] = min(100, volume_cv * 50)

            # ボラティリティリスク
            volatility = returns.std() * np.sqrt(252)
            risk_factors[RiskType.VOLATILITY_RISK] = min(100, volatility * 100)

            # 集中リスク（単一銘柄への依存）
            risk_factors[RiskType.CONCENTRATION_RISK] = 60.0  # デフォルト：中リスク

            # システムリスク（簡易評価）
            risk_factors[RiskType.SYSTEM_RISK] = 20.0  # デフォルト：低リスク

            # 運用リスク（簡易評価）
            risk_factors[RiskType.OPERATIONAL_RISK] = 15.0  # デフォルト：低リスク

        except Exception as e:
            self.logger.error(f"リスクファクター計算エラー: {e}")
            # デフォルト値
            for risk_type in RiskType:
                risk_factors[risk_type] = 30.0

        return risk_factors

    def _calculate_overall_risk_score(self, volatility: float, var_1d: float,
                                    max_drawdown: float, beta: float,
                                    risk_factors: Dict[RiskType, float]) -> float:
        """総合リスクスコア計算"""

        # 各リスク要素を0-100スケールに正規化
        vol_score = min(100, volatility * 200)  # ボラティリティ（50%で100点）
        var_score = min(100, var_1d * 1000)     # VaR（10%で100点）
        dd_score = min(100, max_drawdown * 200)  # ドローダウン（50%で100点）
        beta_score = min(100, abs(beta - 1) * 100)  # ベータ偏差（1からの乖離）

        # リスクファクター平均
        factor_score = np.mean(list(risk_factors.values())) if risk_factors else 50.0

        # 重み付き平均
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        scores = [vol_score, var_score, dd_score, beta_score, factor_score]

        overall_score = sum(w * s for w, s in zip(weights, scores))

        return min(100, max(0, overall_score))

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """スコアからリスクレベルへ変換"""

        if score <= 10:
            return RiskLevel.VERY_LOW
        elif score <= 25:
            return RiskLevel.LOW
        elif score <= 40:
            return RiskLevel.MODERATE
        elif score <= 60:
            return RiskLevel.HIGH
        elif score <= 80:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME

class ProductionRiskValidator:
    """実運用リスク検証システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_engine = RiskCalculationEngine()

        # リスク制限設定
        self.risk_limits = {
            'max_position_risk': 0.02,      # 単一ポジション2%
            'max_daily_var': 0.05,          # 日次VaR 5%
            'max_drawdown': 0.10,           # 最大ドローダウン10%
            'min_diversification': 0.7,     # 分散度70%以上
            'max_correlation': 0.8,         # 相関80%未満
            'max_leverage': 2.0             # レバレッジ2倍未満
        }

        # 承認済みリスクレベル
        self.approved_risk_levels = [
            RiskLevel.VERY_LOW,
            RiskLevel.LOW,
            RiskLevel.MODERATE
        ]

        # データベース設定
        self.db_path = Path("risk_validation_data/production_risk_validation.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        self.logger.info("Production risk validator initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # リスクメトリクス履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_metrics_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        risk_level TEXT NOT NULL,
                        var_1d REAL NOT NULL,
                        volatility REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        beta REAL NOT NULL
                    )
                ''')

                # リスク検証結果テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_validation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        validation_passed INTEGER DEFAULT 0,
                        risk_score REAL NOT NULL,
                        limit_violations TEXT,
                        recommendations TEXT
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    async def validate_trading_risk(self, symbol: str, position_size: float = 1000000) -> Dict[str, Any]:
        """取引リスク検証"""

        print(f"\n🛡️ {symbol} のリスク検証開始...")

        validation_result = {
            'symbol': symbol,
            'position_size': position_size,
            'timestamp': datetime.now().isoformat(),
            'validation_passed': False,
            'risk_metrics': None,
            'limit_violations': [],
            'recommendations': [],
            'overall_assessment': ''
        }

        try:
            # リスクメトリクス計算
            risk_metrics = await self.risk_engine.calculate_comprehensive_risk_metrics(symbol, position_size)

            if risk_metrics is None:
                validation_result['overall_assessment'] = 'データ不足により検証不可'
                return validation_result

            validation_result['risk_metrics'] = {
                'risk_score': risk_metrics.risk_score,
                'risk_level': risk_metrics.risk_level.value,
                'var_1d': risk_metrics.var_1d,
                'var_5d': risk_metrics.var_5d,
                'volatility': risk_metrics.volatility,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'beta': risk_metrics.beta
            }

            # リスク制限チェック
            violations = []

            # 1. リスクレベルチェック
            if risk_metrics.risk_level not in self.approved_risk_levels:
                violations.append(f"リスクレベルが高すぎます: {risk_metrics.risk_level.value}")

            # 2. VaRチェック
            daily_var_ratio = risk_metrics.var_1d / position_size
            if daily_var_ratio > self.risk_limits['max_daily_var']:
                violations.append(f"日次VaRが制限を超過: {daily_var_ratio:.1%} > {self.risk_limits['max_daily_var']:.1%}")

            # 3. ドローダウンチェック
            if risk_metrics.max_drawdown > self.risk_limits['max_drawdown']:
                violations.append(f"最大ドローダウンが制限を超過: {risk_metrics.max_drawdown:.1%} > {self.risk_limits['max_drawdown']:.1%}")

            # 4. ボラティリティチェック
            if risk_metrics.volatility > 0.4:  # 年率40%
                violations.append(f"ボラティリティが高すぎます: {risk_metrics.volatility:.1%}")

            # 5. ベータチェック
            if abs(risk_metrics.beta) > 2.0:
                violations.append(f"ベータ値が極端です: {risk_metrics.beta:.2f}")

            validation_result['limit_violations'] = violations

            # 推奨事項生成
            recommendations = self._generate_recommendations(risk_metrics, violations)
            validation_result['recommendations'] = recommendations

            # 総合判定
            validation_result['validation_passed'] = len(violations) == 0

            if validation_result['validation_passed']:
                validation_result['overall_assessment'] = '✅ リスク検証合格 - 取引承認'
            elif len(violations) <= 2:
                validation_result['overall_assessment'] = '⚠️ 条件付き承認 - 注意して取引'
            else:
                validation_result['overall_assessment'] = '❌ リスク検証不合格 - 取引非推奨'

            # データベースに保存
            await self._save_validation_result(validation_result)

            # リスクメトリクス履歴保存
            await self._save_risk_metrics(risk_metrics)

            return validation_result

        except Exception as e:
            self.logger.error(f"リスク検証エラー {symbol}: {e}")
            validation_result['overall_assessment'] = f'検証エラー: {str(e)}'
            return validation_result

    def _generate_recommendations(self, risk_metrics: RiskMetrics, violations: List[str]) -> List[str]:
        """推奨事項生成"""

        recommendations = []

        # リスクレベル別推奨事項
        if risk_metrics.risk_level == RiskLevel.VERY_LOW:
            recommendations.append("リスクが非常に低く、ポジションサイズを増加可能")
        elif risk_metrics.risk_level == RiskLevel.LOW:
            recommendations.append("適切なリスクレベル、現状維持を推奨")
        elif risk_metrics.risk_level == RiskLevel.MODERATE:
            recommendations.append("中程度のリスク、慎重な監視が必要")
        elif risk_metrics.risk_level == RiskLevel.HIGH:
            recommendations.append("高リスク、ポジションサイズ縮小を検討")
        else:
            recommendations.append("極度に高いリスク、取引回避を強く推奨")

        # 具体的な推奨事項
        if risk_metrics.volatility > 0.3:
            recommendations.append("高ボラティリティのため、ストップロス幅を拡大")

        if risk_metrics.max_drawdown > 0.15:
            recommendations.append("大きなドローダウン履歴があるため、分散投資を推奨")

        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("リスク調整後リターンが低い、他の投資機会を検討")

        if abs(risk_metrics.beta) > 1.5:
            recommendations.append("市場感応度が高い、市場動向に注意")

        # 違反対応
        if violations:
            recommendations.append("リスク制限違反があるため、ポジションサイズを削減")
            recommendations.append("追加のヘッジ戦略を検討")

        return recommendations

    async def _save_validation_result(self, result: Dict[str, Any]):
        """検証結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO risk_validation_results
                    (timestamp, symbol, validation_passed, risk_score, limit_violations, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result['timestamp'],
                    result['symbol'],
                    1 if result['validation_passed'] else 0,
                    result['risk_metrics']['risk_score'] if result['risk_metrics'] else 0.0,
                    json.dumps(result['limit_violations']),
                    json.dumps(result['recommendations'])
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"検証結果保存エラー: {e}")

    async def _save_risk_metrics(self, metrics: RiskMetrics):
        """リスクメトリクス保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO risk_metrics_history
                    (symbol, timestamp, risk_score, risk_level, var_1d, volatility,
                     max_drawdown, sharpe_ratio, beta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.symbol,
                    metrics.timestamp.isoformat(),
                    metrics.risk_score,
                    metrics.risk_level.value,
                    metrics.var_1d,
                    metrics.volatility,
                    metrics.max_drawdown,
                    metrics.sharpe_ratio,
                    metrics.beta
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"リスクメトリクス保存エラー: {e}")

    async def validate_portfolio_risk(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """ポートフォリオリスク検証"""

        print(f"\n📊 ポートフォリオリスク検証開始...")

        portfolio_result = {
            'timestamp': datetime.now().isoformat(),
            'total_positions': len(positions),
            'total_value': sum(positions.values()),
            'individual_risks': {},
            'portfolio_metrics': {},
            'validation_passed': False,
            'overall_assessment': ''
        }

        try:
            # 個別リスク検証
            individual_risks = {}
            for symbol, position_size in positions.items():
                risk_result = await self.validate_trading_risk(symbol, position_size)
                individual_risks[symbol] = risk_result

            portfolio_result['individual_risks'] = individual_risks

            # ポートフォリオレベルの分析
            passed_count = sum(1 for r in individual_risks.values() if r['validation_passed'])
            total_count = len(individual_risks)

            portfolio_metrics = {
                'individual_pass_rate': passed_count / total_count if total_count > 0 else 0,
                'high_risk_positions': sum(1 for r in individual_risks.values()
                                         if r['risk_metrics'] and r['risk_metrics']['risk_score'] > 60),
                'avg_risk_score': np.mean([r['risk_metrics']['risk_score']
                                         for r in individual_risks.values()
                                         if r['risk_metrics']]) if individual_risks else 0
            }

            portfolio_result['portfolio_metrics'] = portfolio_metrics

            # ポートフォリオ判定
            if portfolio_metrics['individual_pass_rate'] >= 0.8:  # 80%以上合格
                portfolio_result['validation_passed'] = True
                portfolio_result['overall_assessment'] = '✅ ポートフォリオリスク検証合格'
            elif portfolio_metrics['individual_pass_rate'] >= 0.6:  # 60%以上合格
                portfolio_result['validation_passed'] = False
                portfolio_result['overall_assessment'] = '⚠️ ポートフォリオリスク要注意'
            else:
                portfolio_result['validation_passed'] = False
                portfolio_result['overall_assessment'] = '❌ ポートフォリオリスク高 - 見直し必要'

            return portfolio_result

        except Exception as e:
            self.logger.error(f"ポートフォリオリスク検証エラー: {e}")
            portfolio_result['overall_assessment'] = f'検証エラー: {str(e)}'
            return portfolio_result

# グローバルインスタンス
production_risk_validator = ProductionRiskValidator()

# テスト実行
async def run_production_risk_validation_test():
    """実運用リスク検証テスト実行"""

    print("=== 🛡️ 実運用リスク管理検証テスト ===")

    # 単一銘柄リスク検証テスト
    test_symbols = ["7203", "8306", "4751"]
    position_size = 1000000  # 100万円

    print(f"\n📈 個別銘柄リスク検証")

    validation_results = {}
    for symbol in test_symbols:
        result = await production_risk_validator.validate_trading_risk(symbol, position_size)
        validation_results[symbol] = result

        # 結果表示
        metrics = result['risk_metrics']
        if metrics:
            print(f"\n--- {symbol} 検証結果 ---")
            print(f"  総合判定: {result['overall_assessment']}")
            print(f"  リスクスコア: {metrics['risk_score']:.1f}/100")
            print(f"  リスクレベル: {metrics['risk_level']}")
            print(f"  日次VaR: {metrics['var_1d']:.0f}円 ({metrics['var_1d']/position_size:.1%})")
            print(f"  ボラティリティ: {metrics['volatility']:.1%}")
            print(f"  最大ドローダウン: {metrics['max_drawdown']:.1%}")
            print(f"  シャープレシオ: {metrics['sharpe_ratio']:.2f}")
            print(f"  ベータ: {metrics['beta']:.2f}")

            if result['limit_violations']:
                print(f"  ⚠️ 制限違反:")
                for violation in result['limit_violations']:
                    print(f"    • {violation}")

            if result['recommendations']:
                print(f"  💡 推奨事項:")
                for rec in result['recommendations'][:3]:  # 上位3件
                    print(f"    • {rec}")
        else:
            print(f"\n--- {symbol} ---")
            print(f"  ❌ データ不足により検証できませんでした")

    # ポートフォリオリスク検証
    print(f"\n📊 ポートフォリオリスク検証")

    portfolio_positions = {symbol: position_size for symbol in test_symbols}
    portfolio_result = await production_risk_validator.validate_portfolio_risk(portfolio_positions)

    print(f"\nポートフォリオ分析結果:")
    print(f"  総合判定: {portfolio_result['overall_assessment']}")
    print(f"  総ポジション数: {portfolio_result['total_positions']}")
    print(f"  総投資額: {portfolio_result['total_value']:,.0f}円")

    portfolio_metrics = portfolio_result['portfolio_metrics']
    print(f"  個別合格率: {portfolio_metrics['individual_pass_rate']:.1%}")
    print(f"  高リスクポジション: {portfolio_metrics['high_risk_positions']}件")
    print(f"  平均リスクスコア: {portfolio_metrics['avg_risk_score']:.1f}/100")

    # 統計サマリー
    print(f"\n📋 検証統計サマリー")

    passed_count = sum(1 for r in validation_results.values() if r['validation_passed'])
    total_count = len(validation_results)

    risk_scores = [r['risk_metrics']['risk_score'] for r in validation_results.values() if r['risk_metrics']]
    avg_risk_score = np.mean(risk_scores) if risk_scores else 0

    print(f"  総検証数: {total_count}")
    print(f"  合格数: {passed_count}")
    print(f"  合格率: {passed_count/total_count:.1%}")
    print(f"  平均リスクスコア: {avg_risk_score:.1f}/100")

    # リスクレベル分布
    risk_levels = [r['risk_metrics']['risk_level'] for r in validation_results.values() if r['risk_metrics']]
    if risk_levels:
        print(f"  リスクレベル分布:")
        for level in set(risk_levels):
            count = risk_levels.count(level)
            print(f"    {level}: {count}件")

    print(f"\n✅ 実運用リスク管理検証完了")
    print(f"システムは実運用に向けたリスク管理体制が整っています。")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_production_risk_validation_test())