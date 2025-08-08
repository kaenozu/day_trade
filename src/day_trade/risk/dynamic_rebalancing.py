#!/usr/bin/env python3
"""
動的リバランシングシステム

Issue #316: 高優先：リスク管理機能強化
市況変化に応じた自動ポートフォリオ調整システム
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 必要に応じてsklearnをインポート
# from sklearn.preprocessing import StandardScaler


class MarketRegime(Enum):
    """市場レジーム"""

    BULL_MARKET = "bull"  # 強気相場
    BEAR_MARKET = "bear"  # 弱気相場
    SIDEWAYS = "sideways"  # 横ばい相場
    HIGH_VOLATILITY = "high_vol"  # 高ボラティリティ相場
    LOW_VOLATILITY = "low_vol"  # 低ボラティリティ相場


class RebalancingTrigger(Enum):
    """リバランシングトリガー"""

    TIME_BASED = "time"  # 時間ベース
    THRESHOLD_BASED = "threshold"  # 閾値ベース
    VOLATILITY_BASED = "volatility"  # ボラティリティベース
    MOMENTUM_BASED = "momentum"  # モメンタムベース
    REGIME_CHANGE = "regime"  # レジーム変化


@dataclass
class RebalancingSignal:
    """リバランシングシグナル"""

    timestamp: datetime
    trigger_type: RebalancingTrigger
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    rebalancing_strength: float  # 0-1の強度
    market_regime: MarketRegime
    confidence: float
    reason: str


@dataclass
class RiskMetrics:
    """リスクメトリクス"""

    portfolio_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    value_at_risk_95: float
    expected_shortfall: float
    tracking_error: Optional[float] = None


class DynamicRebalancingEngine:
    """動的リバランシングエンジン"""

    def __init__(
        self,
        lookback_window: int = 252,
        volatility_window: int = 30,
        momentum_window: int = 60,
        rebalancing_threshold: float = 0.05,
    ):
        """
        初期化

        Args:
            lookback_window: 分析期間（営業日）
            volatility_window: ボラティリティ計算期間
            momentum_window: モメンタム計算期間
            rebalancing_threshold: リバランシング閾値
        """
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.rebalancing_threshold = rebalancing_threshold

        # 市場レジーム検出パラメータ
        self.regime_detection_window = 60
        self.volatility_threshold_high = 0.25  # 年率25%
        self.volatility_threshold_low = 0.15  # 年率15%
        self.trend_threshold = 0.02  # 月次2%

        # リバランシング制約
        self.max_position_weight = 0.40  # 最大40%
        self.min_position_weight = 0.05  # 最小5%
        self.max_turnover = 0.30  # 最大回転率30%

        print("動的リバランシングエンジン初期化完了")

    def detect_market_regime(
        self,
        price_data: Dict[str, pd.DataFrame],
        market_index_data: Optional[pd.DataFrame] = None,
    ) -> MarketRegime:
        """
        市場レジーム検出

        Args:
            price_data: 銘柄価格データ
            market_index_data: 市場インデックスデータ

        Returns:
            検出された市場レジーム
        """
        try:
            # 市場全体のリターンを計算
            if (
                market_index_data is not None
                and len(market_index_data) > self.regime_detection_window
            ):
                market_returns = market_index_data["Close"].pct_change().dropna()
            else:
                # インデックスデータがない場合は、個別銘柄から市場を推定
                all_returns = []
                for _, data in price_data.items():
                    if len(data) > self.regime_detection_window:
                        returns = data["Close"].pct_change().dropna()
                        all_returns.append(returns.tail(self.regime_detection_window))

                if all_returns:
                    market_returns = pd.concat(all_returns, axis=1).mean(axis=1)
                else:
                    return MarketRegime.SIDEWAYS

            # 最近のデータのみ使用
            recent_returns = market_returns.tail(self.regime_detection_window)

            if len(recent_returns) < 30:
                return MarketRegime.SIDEWAYS

            # ボラティリティ計算（年率）
            volatility = recent_returns.std() * np.sqrt(252)

            # トレンド計算（月次リターン）
            monthly_return = (1 + recent_returns.tail(21)).prod() - 1  # 約1ヶ月

            # レジーム判定
            if volatility > self.volatility_threshold_high:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < self.volatility_threshold_low:
                if monthly_return > self.trend_threshold:
                    return MarketRegime.BULL_MARKET
                elif monthly_return < -self.trend_threshold:
                    return MarketRegime.BEAR_MARKET
                else:
                    return MarketRegime.LOW_VOLATILITY
            else:
                if monthly_return > self.trend_threshold:
                    return MarketRegime.BULL_MARKET
                elif monthly_return < -self.trend_threshold:
                    return MarketRegime.BEAR_MARKET
                else:
                    return MarketRegime.SIDEWAYS

        except Exception as e:
            print(f"市場レジーム検出エラー: {e}")
            return MarketRegime.SIDEWAYS

    def calculate_regime_based_weights(
        self,
        price_data: Dict[str, pd.DataFrame],
        market_regime: MarketRegime,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        レジームに基づく目標ウェイト計算

        Args:
            price_data: 価格データ
            market_regime: 市場レジーム
            current_weights: 現在のウェイト

        Returns:
            目標ウェイト
        """
        try:
            symbols = list(price_data.keys())

            # 各銘柄のリスク・リターン特性計算
            risk_return_metrics = {}

            for symbol in symbols:
                data = price_data[symbol]
                if len(data) < 60:
                    continue

                returns = data["Close"].pct_change().dropna()
                recent_returns = returns.tail(self.lookback_window)

                if len(recent_returns) < 30:
                    continue

                # 基本統計
                mean_return = recent_returns.mean() * 252  # 年率
                volatility = recent_returns.std() * np.sqrt(252)  # 年率
                sharpe = mean_return / volatility if volatility > 0 else 0

                # モメンタム指標
                momentum_returns = returns.tail(self.momentum_window)
                momentum_score = (1 + momentum_returns).prod() - 1

                # 最大ドローダウン
                cumulative = (1 + recent_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdowns = (cumulative - running_max) / running_max
                max_drawdown = drawdowns.min()

                risk_return_metrics[symbol] = {
                    "return": mean_return,
                    "volatility": volatility,
                    "sharpe": sharpe,
                    "momentum": momentum_score,
                    "max_drawdown": max_drawdown,
                }

            if not risk_return_metrics:
                return current_weights

            # レジーム別の重み付け戦略
            if market_regime == MarketRegime.BULL_MARKET:
                target_weights = self._bull_market_allocation(
                    risk_return_metrics, current_weights
                )
            elif market_regime == MarketRegime.BEAR_MARKET:
                target_weights = self._bear_market_allocation(
                    risk_return_metrics, current_weights
                )
            elif market_regime == MarketRegime.HIGH_VOLATILITY:
                target_weights = self._high_volatility_allocation(
                    risk_return_metrics, current_weights
                )
            elif market_regime == MarketRegime.LOW_VOLATILITY:
                target_weights = self._low_volatility_allocation(
                    risk_return_metrics, current_weights
                )
            else:  # SIDEWAYS
                target_weights = self._balanced_allocation(
                    risk_return_metrics, current_weights
                )

            # 制約適用
            target_weights = self._apply_constraints(target_weights)

            return target_weights

        except Exception as e:
            print(f"ウェイト計算エラー: {e}")
            return current_weights

    def _bull_market_allocation(
        self, metrics: Dict[str, Dict], current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """強気相場でのアロケーション"""
        # モメンタムとシャープレシオ重視
        scores = {}
        for symbol, data in metrics.items():
            scores[symbol] = 0.6 * data["momentum"] + 0.4 * data["sharpe"]

        return self._normalize_weights(scores)

    def _bear_market_allocation(
        self, metrics: Dict[str, Dict], current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """弱気相場でのアロケーション"""
        # 低ボラティリティと小さなドローダウン重視
        scores = {}
        for symbol, data in metrics.items():
            volatility_score = max(
                0, 1 - data["volatility"] / 0.5
            )  # ボラティリティペナルティ
            drawdown_score = max(
                0, 1 + data["max_drawdown"] / 0.3
            )  # ドローダウンペナルティ
            scores[symbol] = 0.5 * volatility_score + 0.5 * drawdown_score

        return self._normalize_weights(scores)

    def _high_volatility_allocation(
        self, metrics: Dict[str, Dict], current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """高ボラティリティ相場でのアロケーション"""
        # リスク分散重視、低ボラティリティ銘柄を選好
        scores = {}
        for symbol, data in metrics.items():
            volatility_penalty = 1 / (1 + data["volatility"])
            risk_adjusted_return = data["sharpe"]
            scores[symbol] = 0.3 * risk_adjusted_return + 0.7 * volatility_penalty

        return self._normalize_weights(scores)

    def _low_volatility_allocation(
        self, metrics: Dict[str, Dict], current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """低ボラティリティ相場でのアロケーション"""
        # リターン重視、若干のリスクテイク
        scores = {}
        for symbol, data in metrics.items():
            scores[symbol] = 0.7 * data["return"] + 0.3 * data["sharpe"]

        return self._normalize_weights(scores)

    def _balanced_allocation(
        self, metrics: Dict[str, Dict], current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """バランス型アロケーション"""
        # リスク調整後リターン重視
        scores = {}
        for symbol, data in metrics.items():
            scores[symbol] = data["sharpe"]

        return self._normalize_weights(scores)

    def _normalize_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """スコアをウェイトに正規化"""
        if not scores:
            return {}

        # 負のスコアを0にクリップ
        clipped_scores = {k: max(0, v) for k, v in scores.items()}

        total_score = sum(clipped_scores.values())
        if total_score == 0:
            # 等重み配分
            equal_weight = 1.0 / len(clipped_scores)
            return {k: equal_weight for k in clipped_scores}

        return {k: v / total_score for k, v in clipped_scores.items()}

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """制約条件適用"""
        constrained_weights = weights.copy()

        # 最大・最小ウェイト制約
        for symbol in constrained_weights:
            constrained_weights[symbol] = max(
                self.min_position_weight,
                min(self.max_position_weight, constrained_weights[symbol]),
            )

        # 再正規化
        total = sum(constrained_weights.values())
        if total > 0:
            constrained_weights = {k: v / total for k, v in constrained_weights.items()}

        return constrained_weights

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        last_rebalance_date: Optional[datetime] = None,
    ) -> Tuple[bool, List[RebalancingTrigger]]:
        """
        リバランシング必要性判定

        Args:
            current_weights: 現在のウェイト
            target_weights: 目標ウェイト
            price_data: 価格データ
            last_rebalance_date: 前回リバランシング日

        Returns:
            (リバランシング必要性, トリガーリスト)
        """
        triggers = []

        try:
            # 1. 閾値ベース判定
            max_deviation = 0
            for symbol in current_weights:
                if symbol in target_weights:
                    deviation = abs(current_weights[symbol] - target_weights[symbol])
                    max_deviation = max(max_deviation, deviation)

            if max_deviation > self.rebalancing_threshold:
                triggers.append(RebalancingTrigger.THRESHOLD_BASED)

            # 2. 時間ベース判定（月次リバランシング）
            if last_rebalance_date:
                days_since_rebalance = (datetime.now() - last_rebalance_date).days
                if days_since_rebalance >= 30:  # 30日以上
                    triggers.append(RebalancingTrigger.TIME_BASED)

            # 3. ボラティリティベース判定
            recent_volatilities = []
            for _, data in price_data.items():
                if len(data) >= self.volatility_window:
                    returns = data["Close"].pct_change().dropna()
                    vol = returns.tail(self.volatility_window).std() * np.sqrt(252)
                    recent_volatilities.append(vol)

            if recent_volatilities:
                avg_volatility = np.mean(recent_volatilities)
                if avg_volatility > 0.35:  # 35%以上の高ボラティリティ
                    triggers.append(RebalancingTrigger.VOLATILITY_BASED)

            # 4. モメンタム変化判定
            momentum_signals = []
            for _, data in price_data.items():
                if len(data) >= self.momentum_window:
                    returns = data["Close"].pct_change().dropna()
                    momentum = (1 + returns.tail(21)).prod() - 1  # 月次モメンタム
                    momentum_signals.append(momentum)

            if momentum_signals:
                avg_momentum = np.mean(momentum_signals)
                if abs(avg_momentum) > 0.05:  # 5%以上の強いモメンタム
                    triggers.append(RebalancingTrigger.MOMENTUM_BASED)

            return len(triggers) > 0, triggers

        except Exception as e:
            print(f"リバランシング判定エラー: {e}")
            return False, []

    def generate_rebalancing_signal(
        self,
        current_weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        market_index_data: Optional[pd.DataFrame] = None,
    ) -> Optional[RebalancingSignal]:
        """
        リバランシングシグナル生成

        Args:
            current_weights: 現在のウェイト
            price_data: 価格データ
            market_index_data: 市場インデックスデータ

        Returns:
            リバランシングシグナル
        """
        try:
            # 市場レジーム検出
            market_regime = self.detect_market_regime(price_data, market_index_data)

            # 目標ウェイト計算
            target_weights = self.calculate_regime_based_weights(
                price_data, market_regime, current_weights
            )

            # リバランシング必要性判定
            should_rebalance, triggers = self.should_rebalance(
                current_weights, target_weights, price_data
            )

            if not should_rebalance:
                return None

            # リバランシング強度計算
            total_deviation = sum(
                abs(current_weights.get(s, 0) - target_weights.get(s, 0))
                for s in set(list(current_weights.keys()) + list(target_weights.keys()))
            )
            rebalancing_strength = min(1.0, total_deviation / 0.2)  # 20%で最大強度

            # 信頼度計算
            confidence = self._calculate_signal_confidence(
                price_data, market_regime, triggers
            )

            # 理由生成
            reason = self._generate_rebalancing_reason(market_regime, triggers)

            return RebalancingSignal(
                timestamp=datetime.now(),
                trigger_type=triggers[0]
                if triggers
                else RebalancingTrigger.THRESHOLD_BASED,
                current_weights=current_weights,
                target_weights=target_weights,
                rebalancing_strength=rebalancing_strength,
                market_regime=market_regime,
                confidence=confidence,
                reason=reason,
            )

        except Exception as e:
            print(f"リバランシングシグナル生成エラー: {e}")
            return None

    def _calculate_signal_confidence(
        self,
        price_data: Dict[str, pd.DataFrame],
        market_regime: MarketRegime,
        triggers: List[RebalancingTrigger],
    ) -> float:
        """シグナル信頼度計算"""
        base_confidence = 0.6

        # トリガー数による信頼度向上
        trigger_bonus = min(0.3, len(triggers) * 0.1)

        # データ品質による調整
        data_quality = 0
        valid_symbols = 0

        for _, data in price_data.items():
            if len(data) >= self.lookback_window:
                valid_symbols += 1
                # データの連続性をチェック
                returns = data["Close"].pct_change().dropna()
                if len(returns) > 0:
                    data_quality += 1 - (returns.isna().sum() / len(returns))

        if valid_symbols > 0:
            avg_data_quality = data_quality / valid_symbols
            quality_bonus = 0.1 * avg_data_quality
        else:
            quality_bonus = 0

        return min(1.0, base_confidence + trigger_bonus + quality_bonus)

    def _generate_rebalancing_reason(
        self, market_regime: MarketRegime, triggers: List[RebalancingTrigger]
    ) -> str:
        """リバランシング理由生成"""
        regime_descriptions = {
            MarketRegime.BULL_MARKET: "強気相場でのモメンタム重視配分",
            MarketRegime.BEAR_MARKET: "弱気相場での保守的配分",
            MarketRegime.HIGH_VOLATILITY: "高ボラティリティ環境でのリスク分散",
            MarketRegime.LOW_VOLATILITY: "低ボラティリティ環境でのリターン追求",
            MarketRegime.SIDEWAYS: "横ばい相場でのバランス配分",
        }

        trigger_descriptions = {
            RebalancingTrigger.TIME_BASED: "定期リバランシング",
            RebalancingTrigger.THRESHOLD_BASED: "ウェイト偏差閾値超過",
            RebalancingTrigger.VOLATILITY_BASED: "ボラティリティ変化",
            RebalancingTrigger.MOMENTUM_BASED: "モメンタム変化",
            RebalancingTrigger.REGIME_CHANGE: "市場レジーム変化",
        }

        regime_reason = regime_descriptions.get(market_regime, "市場環境変化")

        if triggers:
            trigger_reason = ", ".join(
                [trigger_descriptions.get(t, "その他") for t in triggers]
            )
            return f"{regime_reason} (トリガー: {trigger_reason})"
        else:
            return regime_reason

    def calculate_portfolio_risk_metrics(
        self,
        weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> RiskMetrics:
        """ポートフォリオリスクメトリクス計算"""
        try:
            # ポートフォリオリターン計算
            portfolio_returns = self._calculate_portfolio_returns(weights, price_data)

            if len(portfolio_returns) < 30:
                raise ValueError("リスク計算に十分なデータがありません")

            # 基本統計
            mean_return = portfolio_returns.mean() * 252
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0

            # 最大ドローダウン
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min()

            # VaRとExpected Shortfall
            var_95 = np.percentile(portfolio_returns, 5)
            tail_returns = portfolio_returns[portfolio_returns <= var_95]
            expected_shortfall = (
                tail_returns.mean() if len(tail_returns) > 0 else var_95
            )

            # ベータ計算
            beta = 1.0
            tracking_error = None

            if benchmark_data is not None and len(benchmark_data) > 0:
                benchmark_returns = benchmark_data["Close"].pct_change().dropna()

                # 共通期間で計算
                common_dates = portfolio_returns.index.intersection(
                    benchmark_returns.index
                )
                if len(common_dates) > 30:
                    port_common = portfolio_returns.reindex(common_dates)
                    bench_common = benchmark_returns.reindex(common_dates)

                    covariance = port_common.cov(bench_common)
                    benchmark_variance = bench_common.var()

                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance

                    # トラッキングエラー
                    active_returns = port_common - bench_common
                    tracking_error = active_returns.std() * np.sqrt(252)

            return RiskMetrics(
                portfolio_volatility=volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                value_at_risk_95=var_95,
                expected_shortfall=expected_shortfall,
                tracking_error=tracking_error,
            )

        except Exception as e:
            print(f"リスクメトリクス計算エラー: {e}")
            # デフォルト値を返す
            return RiskMetrics(
                portfolio_volatility=0.2,
                max_drawdown=-0.1,
                sharpe_ratio=0.5,
                beta=1.0,
                value_at_risk_95=-0.02,
                expected_shortfall=-0.03,
            )

    def _calculate_portfolio_returns(
        self, weights: Dict[str, float], price_data: Dict[str, pd.DataFrame]
    ) -> pd.Series:
        """ポートフォリオリターン計算"""
        all_returns = {}

        for symbol, weight in weights.items():
            if symbol in price_data and weight > 0:
                data = price_data[symbol]
                if len(data) > 1:
                    returns = data["Close"].pct_change().dropna()
                    all_returns[symbol] = returns * weight

        if not all_returns:
            return pd.Series(dtype=float)

        # 共通期間でポートフォリオリターン計算
        returns_df = pd.DataFrame(all_returns)
        portfolio_returns = returns_df.sum(axis=1)

        return portfolio_returns


if __name__ == "__main__":
    # テスト実行
    print("動的リバランシングシステムテスト")
    print("=" * 50)

    # サンプルデータ生成
    from datetime import datetime

    import yfinance as yf

    try:
        # 実データでテスト
        symbols = ["7203.T", "8306.T", "9984.T"]

        print("実データ取得中...")
        price_data = {}

        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            if not data.empty:
                price_data[symbol] = data
                print(f"取得成功: {symbol} - {len(data)}日分")

        if price_data:
            # リバランシングエンジンテスト
            engine = DynamicRebalancingEngine()

            # 現在のウェイト（等重み）
            current_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
            print(f"現在のウェイト: {current_weights}")

            # 市場レジーム検出
            market_regime = engine.detect_market_regime(price_data)
            print(f"市場レジーム: {market_regime.value}")

            # リバランシングシグナル生成
            signal = engine.generate_rebalancing_signal(current_weights, price_data)

            if signal:
                print("\n=== リバランシングシグナル ===")
                print(f"トリガー: {signal.trigger_type.value}")
                print(f"市場レジーム: {signal.market_regime.value}")
                print(f"信頼度: {signal.confidence:.2%}")
                print(f"リバランシング強度: {signal.rebalancing_strength:.2%}")
                print(f"理由: {signal.reason}")
                print("目標ウェイト:")
                for symbol, weight in signal.target_weights.items():
                    change = weight - current_weights.get(symbol, 0)
                    print(f"  {symbol}: {weight:.2%} (変更: {change:+.2%})")

                # リスクメトリクス計算
                risk_metrics = engine.calculate_portfolio_risk_metrics(
                    signal.target_weights, price_data
                )

                print("\n=== リスクメトリクス ===")
                print(
                    f"ポートフォリオボラティリティ: {risk_metrics.portfolio_volatility:.2%}"
                )
                print(f"最大ドローダウン: {risk_metrics.max_drawdown:.2%}")
                print(f"シャープレシオ: {risk_metrics.sharpe_ratio:.3f}")
                print(f"VaR(95%): {risk_metrics.value_at_risk_95:.2%}")
                print(f"Expected Shortfall: {risk_metrics.expected_shortfall:.2%}")

            else:
                print("リバランシング不要")

            print("\n動的リバランシングシステムテスト完了")

        else:
            print("データ取得に失敗しました")

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()
