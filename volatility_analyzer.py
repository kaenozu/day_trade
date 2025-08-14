#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Analyzer - 高ボラティリティ銘柄分析システム

デイトレード利益幅拡大のための高ボラティリティ銘柄発掘・分析
リスク管理と利益機会のバランス最適化
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import statistics

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

class VolatilityCategory(Enum):
    """ボラティリティカテゴリ"""
    ULTRA_LOW = "超低ボラ"     # <1.5%
    LOW = "低ボラ"            # 1.5-3%
    MEDIUM = "中ボラ"         # 3-6%
    HIGH = "高ボラ"           # 6-10%
    ULTRA_HIGH = "超高ボラ"   # >10%

class TradabilityScore(Enum):
    """トレード適性スコア"""
    EXCELLENT = "優秀"        # 90-100点
    GOOD = "良好"             # 70-89点
    FAIR = "普通"             # 50-69点
    POOR = "不良"             # 30-49点
    AVOID = "回避"            # <30点

@dataclass
class VolatilityMetrics:
    """ボラティリティ指標"""
    symbol: str
    name: str

    # 基本ボラティリティ
    daily_volatility: float          # 日次ボラティリティ(%)
    weekly_volatility: float         # 週次ボラティリティ(%)
    monthly_volatility: float        # 月次ボラティリティ(%)

    # 高度指標
    realized_volatility: float       # 実現ボラティリティ
    garch_volatility: float         # GARCH推定ボラティリティ
    volume_weighted_volatility: float # 出来高加重ボラティリティ

    # トレーダビリティ
    average_daily_volume: float      # 平均日次出来高
    bid_ask_spread: float           # ビッド・アスク・スプレッド(%)
    price_impact_score: float       # 価格インパクトスコア

    # リスク指標
    var_95: float                   # バリューアットリスク95%
    max_drawdown: float             # 最大ドローダウン(%)
    sharpe_ratio: float             # シャープレシオ

    # 分類
    volatility_category: VolatilityCategory
    tradability_score: TradabilityScore
    daytrading_score: float         # デイトレード総合スコア(0-100)

    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class VolatilityOpportunity:
    """ボラティリティ機会"""
    symbol: str
    name: str
    current_volatility: float
    expected_move: float            # 予想変動幅(%)
    probability: float              # 機会確率(%)
    risk_reward_ratio: float        # リスクリワード比
    optimal_position_size: float    # 最適ポジションサイズ(%)
    time_horizon: str              # 想定保有期間
    entry_triggers: List[str]       # エントリートリガー条件
    exit_strategy: str             # エグジット戦略

class VolatilityAnalyzer:
    """
    高ボラティリティ銘柄分析システム
    デイトレードに最適な高ボラ銘柄の発掘・分析
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データディレクトリ
        self.data_dir = Path("volatility_data")
        self.data_dir.mkdir(exist_ok=True)

        # 高ボラティリティ銘柄候補データベース
        self.high_volatility_candidates = self._initialize_high_vol_database()

        # 分析履歴
        self.analysis_history: List[VolatilityMetrics] = []

        self.logger.info(f"Volatility analyzer initialized with {len(self.high_volatility_candidates)} candidates")

    def _initialize_high_vol_database(self) -> Dict[str, str]:
        """高ボラティリティ銘柄候補データベース初期化"""

        return {
            # 新興IT・テクノロジー
            "4751": "サイバーエージェント",
            "4385": "メルカリ",
            "4478": "フリー",
            "3659": "ネクソン",
            "2432": "DeNA",
            "4716": "オルトプラス",
            "3778": "さくらインターネット",
            "4563": "アンジェス",

            # バイオ・創薬
            "4565": "そーせいG",
            "4587": "ペプチドリーム",
            "4579": "ラクオリア創薬",
            "4588": "オンコリスバイオファーマ",
            "4597": "ソレイジア・ファーマ",
            "4583": "カイオム・バイオサイエンス",

            # 半導体・エレクトロニクス
            "6723": "ルネサス",
            "7735": "SCREENホールディングス",
            "6857": "アドバンテスト",
            "6920": "レーザーテック",
            "6963": "ローム",
            "6981": "村田製作所",

            # 新興エネルギー・環境
            "1605": "INPEX",
            "5020": "ENEOSホールディングス",
            "9513": "Jパワー",
            "9519": "レノバ",

            # 小型成長株
            "2370": "メディネット",
            "2158": "フロンティア",
            "4592": "サンバイオ",
            "4596": "窪田製薬HD",

            # ゲーム・エンタメ
            "3664": "モブキャスト",
            "3687": "フィックスターズ",
            "3765": "ガンホー・オンライン・エンターテイメント",

            # フィンテック
            "4482": "ウィルズ",
            "4493": "サイバーセキュリティクラウド",
        }

    async def analyze_volatility_metrics(self, symbol: str, period_days: int = 60) -> Optional[VolatilityMetrics]:
        """
        銘柄のボラティリティ指標分析

        Args:
            symbol: 銘柄コード
            period_days: 分析期間(日)

        Returns:
            VolatilityMetrics: ボラティリティ指標
        """
        try:
            # Yahoo Finance からデータ取得
            ticker = yf.Ticker(f"{symbol}.T")
            hist = ticker.history(period=f"{period_days}d")

            if hist.empty or len(hist) < 20:  # 最低20日分のデータが必要
                self.logger.warning(f"Insufficient data for {symbol}")
                return None

            # 基本データ準備
            prices = hist['Close']
            volumes = hist['Volume']
            returns = prices.pct_change().dropna()

            # 基本ボラティリティ計算
            daily_vol = returns.std() * 100
            weekly_vol = returns.std() * np.sqrt(5) * 100
            monthly_vol = returns.std() * np.sqrt(22) * 100

            # 高度ボラティリティ指標
            realized_vol = self._calculate_realized_volatility(returns)
            garch_vol = self._estimate_garch_volatility(returns)
            volume_weighted_vol = self._calculate_volume_weighted_volatility(returns, volumes)

            # トレーダビリティ指標
            avg_volume = volumes.mean()
            bid_ask_spread = self._estimate_bid_ask_spread(prices)
            price_impact = self._calculate_price_impact_score(returns, volumes)

            # リスク指標
            var_95 = np.percentile(returns * 100, 5)  # 5%分位点
            max_dd = self._calculate_max_drawdown(prices)
            sharpe = self._calculate_sharpe_ratio(returns)

            # 分類
            vol_category = self._categorize_volatility(daily_vol)
            tradability = self._assess_tradability(avg_volume, bid_ask_spread, price_impact)
            daytrading_score = self._calculate_daytrading_score(
                daily_vol, avg_volume, bid_ask_spread, max_dd, sharpe
            )

            # 銘柄名取得
            name = self.high_volatility_candidates.get(symbol, f"銘柄{symbol}")

            return VolatilityMetrics(
                symbol=symbol,
                name=name,
                daily_volatility=daily_vol,
                weekly_volatility=weekly_vol,
                monthly_volatility=monthly_vol,
                realized_volatility=realized_vol,
                garch_volatility=garch_vol,
                volume_weighted_volatility=volume_weighted_vol,
                average_daily_volume=avg_volume,
                bid_ask_spread=bid_ask_spread,
                price_impact_score=price_impact,
                var_95=var_95,
                max_drawdown=max_dd,
                sharpe_ratio=sharpe,
                volatility_category=vol_category,
                tradability_score=tradability,
                daytrading_score=daytrading_score
            )

        except Exception as e:
            self.logger.error(f"Volatility analysis failed for {symbol}: {e}")
            return None

    def _calculate_realized_volatility(self, returns: pd.Series) -> float:
        """実現ボラティリティ計算"""
        return np.sqrt(np.sum(returns**2)) * 100

    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """GARCH推定ボラティリティ（簡易版）"""
        # 簡易GARCH(1,1)近似
        alpha = 0.1  # ラグ項係数
        beta = 0.85  # 持続性係数

        variance = returns.var()
        garch_var = variance

        for ret in returns:
            garch_var = (1 - alpha - beta) * variance + alpha * ret**2 + beta * garch_var

        return np.sqrt(garch_var * 252) * 100  # 年率化

    def _calculate_volume_weighted_volatility(self, returns: pd.Series, volumes: pd.Series) -> float:
        """出来高加重ボラティリティ"""
        if len(returns) != len(volumes) or volumes.sum() == 0:
            return returns.std() * 100

        # 出来高で重み付けした分散計算
        weights = volumes / volumes.sum()
        weighted_mean = np.sum(returns * weights)
        weighted_var = np.sum(weights * (returns - weighted_mean)**2)

        return np.sqrt(weighted_var) * 100

    def _estimate_bid_ask_spread(self, prices: pd.Series) -> float:
        """ビッド・アスク・スプレッド推定（簡易版）"""
        # 高頻度データがないため、日次データから推定
        price_changes = prices.diff().abs()
        avg_price = prices.mean()

        # 価格変動の最小単位から推定
        if avg_price > 0:
            estimated_spread = price_changes.quantile(0.1) / avg_price * 100
            return min(estimated_spread, 2.0)  # 最大2%でキャップ
        return 0.5  # デフォルト0.5%

    def _calculate_price_impact_score(self, returns: pd.Series, volumes: pd.Series) -> float:
        """価格インパクトスコア計算"""
        if len(returns) != len(volumes):
            return 50.0

        # 出来高と価格変動の相関から推定
        correlation = returns.abs().corr(1/volumes.replace(0, np.nan))

        # 相関が高いほど価格インパクトが大きい（流動性が低い）
        if pd.isna(correlation):
            return 50.0

        # 0-100スケールに変換（低いほど良い）
        impact_score = max(0, min(100, 50 + correlation * 50))
        return impact_score

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min() * 100)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """シャープレシオ計算"""
        if returns.std() == 0:
            return 0.0

        # リスクフリーレート0.1%と仮定
        risk_free_rate = 0.001 / 252  # 日次リスクフリーレート
        excess_returns = returns - risk_free_rate

        return excess_returns.mean() / returns.std() * np.sqrt(252)  # 年率化

    def _categorize_volatility(self, daily_vol: float) -> VolatilityCategory:
        """ボラティリティ分類"""
        if daily_vol < 1.5:
            return VolatilityCategory.ULTRA_LOW
        elif daily_vol < 3.0:
            return VolatilityCategory.LOW
        elif daily_vol < 6.0:
            return VolatilityCategory.MEDIUM
        elif daily_vol < 10.0:
            return VolatilityCategory.HIGH
        else:
            return VolatilityCategory.ULTRA_HIGH

    def _assess_tradability(self, volume: float, spread: float, price_impact: float) -> TradabilityScore:
        """トレーダビリティ評価"""
        # 複合スコア計算
        volume_score = min(100, volume / 1000000 * 20)  # 出来高100万で20点
        spread_score = max(0, 100 - spread * 50)        # スプレッド2%で0点
        impact_score = 100 - price_impact               # 価格インパクトスコア逆転

        total_score = (volume_score * 0.4 + spread_score * 0.3 + impact_score * 0.3)

        if total_score >= 90:
            return TradabilityScore.EXCELLENT
        elif total_score >= 70:
            return TradabilityScore.GOOD
        elif total_score >= 50:
            return TradabilityScore.FAIR
        elif total_score >= 30:
            return TradabilityScore.POOR
        else:
            return TradabilityScore.AVOID

    def _calculate_daytrading_score(self, volatility: float, volume: float,
                                   spread: float, max_dd: float, sharpe: float) -> float:
        """デイトレード総合スコア計算"""

        # ボラティリティスコア（3-8%が最適）
        if 3.0 <= volatility <= 8.0:
            vol_score = 100
        elif volatility < 3.0:
            vol_score = volatility / 3.0 * 80  # 低ボラは減点
        else:
            vol_score = max(20, 100 - (volatility - 8.0) * 5)  # 高ボラも減点

        # 流動性スコア
        liquidity_score = min(100, volume / 500000 * 100)  # 50万で100点

        # スプレッドスコア
        spread_score = max(0, 100 - spread * 100)  # 1%で0点

        # リスクスコア
        risk_score = max(0, 100 - max_dd)  # ドローダウン100%で0点

        # シャープレシオスコア
        sharpe_score = max(0, min(100, 50 + sharpe * 25))  # シャープ2で100点

        # 総合スコア（重み付き平均）
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # ボラ>流動>スプレ>リスク>シャープ
        scores = [vol_score, liquidity_score, spread_score, risk_score, sharpe_score]

        return sum(w * s for w, s in zip(weights, scores))

    async def get_high_volatility_opportunities(self, min_score: float = 60.0,
                                              limit: int = 10) -> List[VolatilityMetrics]:
        """
        高ボラティリティ機会銘柄取得

        Args:
            min_score: 最低デイトレードスコア
            limit: 取得上限数

        Returns:
            List[VolatilityMetrics]: 高ボラ機会銘柄リスト
        """
        opportunities = []

        print(f"高ボラティリティ銘柄分析中... ({len(self.high_volatility_candidates)}銘柄)")

        for i, (symbol, name) in enumerate(self.high_volatility_candidates.items()):
            try:
                print(f"  {i+1}/{len(self.high_volatility_candidates)}: {symbol} ({name})")

                metrics = await self.analyze_volatility_metrics(symbol)
                if metrics and metrics.daytrading_score >= min_score:
                    opportunities.append(metrics)
                    self.analysis_history.append(metrics)

                # 進捗表示
                if (i + 1) % 5 == 0:
                    print(f"  → 進捗: {i+1}/{len(self.high_volatility_candidates)} 完了")

            except Exception as e:
                self.logger.error(f"Analysis failed for {symbol}: {e}")
                continue

        # スコア順でソート
        opportunities.sort(key=lambda x: x.daytrading_score, reverse=True)

        return opportunities[:limit]

    def generate_volatility_report(self, opportunities: List[VolatilityMetrics]) -> Dict[str, Any]:
        """ボラティリティレポート生成"""

        if not opportunities:
            return {"error": "No opportunities found"}

        # 統計計算
        scores = [op.daytrading_score for op in opportunities]
        volatilities = [op.daily_volatility for op in opportunities]
        volumes = [op.average_daily_volume for op in opportunities]

        # カテゴリ分布
        category_distribution = {}
        for op in opportunities:
            cat = op.volatility_category.value
            category_distribution[cat] = category_distribution.get(cat, 0) + 1

        return {
            "summary": {
                "total_opportunities": len(opportunities),
                "avg_daytrading_score": statistics.mean(scores),
                "avg_daily_volatility": statistics.mean(volatilities),
                "avg_daily_volume": statistics.mean(volumes)
            },
            "top_picks": [
                {
                    "symbol": op.symbol,
                    "name": op.name,
                    "daytrading_score": op.daytrading_score,
                    "daily_volatility": op.daily_volatility,
                    "category": op.volatility_category.value,
                    "tradability": op.tradability_score.value
                }
                for op in opportunities[:5]
            ],
            "category_distribution": category_distribution,
            "risk_analysis": {
                "avg_max_drawdown": statistics.mean([op.max_drawdown for op in opportunities]),
                "avg_sharpe_ratio": statistics.mean([op.sharpe_ratio for op in opportunities]),
                "avg_var_95": statistics.mean([op.var_95 for op in opportunities])
            }
        }

# テスト関数
async def test_volatility_analyzer():
    """ボラティリティ分析システムのテスト"""
    print("=== 高ボラティリティ銘柄分析システム テスト ===")

    analyzer = VolatilityAnalyzer()

    # 高ボラティリティ機会分析
    print(f"\n[ 高ボラティリティ機会分析 ]")
    opportunities = await analyzer.get_high_volatility_opportunities(min_score=50.0, limit=10)

    if opportunities:
        print(f"\n[ 発見された機会: {len(opportunities)}銘柄 ]")
        for i, op in enumerate(opportunities, 1):
            print(f"{i}. {op.symbol} ({op.name})")
            print(f"   デイトレードスコア: {op.daytrading_score:.1f}")
            print(f"   日次ボラティリティ: {op.daily_volatility:.2f}%")
            print(f"   カテゴリ: {op.volatility_category.value}")
            print(f"   トレーダビリティ: {op.tradability_score.value}")

        # レポート生成
        print(f"\n[ ボラティリティレポート ]")
        report = analyzer.generate_volatility_report(opportunities)

        summary = report["summary"]
        print(f"平均デイトレードスコア: {summary['avg_daytrading_score']:.1f}")
        print(f"平均日次ボラティリティ: {summary['avg_daily_volatility']:.2f}%")
        print(f"平均日次出来高: {summary['avg_daily_volume']:,.0f}")

        print(f"\nカテゴリ分布: {report['category_distribution']}")

        risk = report["risk_analysis"]
        print(f"\nリスク分析:")
        print(f"  平均最大ドローダウン: {risk['avg_max_drawdown']:.2f}%")
        print(f"  平均シャープレシオ: {risk['avg_sharpe_ratio']:.2f}")

    else:
        print("条件を満たす高ボラティリティ銘柄が見つかりませんでした")

    print(f"\n=== 高ボラティリティ銘柄分析システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    asyncio.run(test_volatility_analyzer())