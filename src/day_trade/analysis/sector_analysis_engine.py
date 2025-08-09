#!/usr/bin/env python3
"""
セクター別分析エンジン
Issue #314: TOPIX500全銘柄対応

業種別・セクター別の包括的分析システム
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from ..data.topix500_master import TOPIX500MasterManager
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # モックマネージャー
    class TOPIX500MasterManager:
        def get_sector_summary(self):
            return {
                "3700": {"sector_name": "輸送用機器", "stock_count": 5},
                "7050": {"sector_name": "銀行業", "stock_count": 3},
                "3250": {"sector_name": "医薬品", "stock_count": 4},
            }

        def get_symbols_by_sector(self, sector_code):
            return [
                {"code": "7203", "name": "トヨタ自動車", "market_cap": 30000000},
                {"code": "7267", "name": "ホンダ", "market_cap": 10000000},
            ]


logger = get_context_logger(__name__)


@dataclass
class SectorPerformance:
    """セクターパフォーマンス"""

    sector_code: str
    sector_name: str
    stock_count: int
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_with_market: float
    momentum_score: float
    value_score: float
    quality_score: float


@dataclass
class SectorRotationSignal:
    """セクターローテーションシグナル"""

    from_sector: str
    to_sector: str
    signal_strength: float
    rotation_type: str  # "defensive", "growth", "cyclical", "value"
    confidence: float
    supporting_factors: List[str]


@dataclass
class CrossSectorAnalysis:
    """クロスセクター分析"""

    correlation_matrix: pd.DataFrame
    cluster_assignments: Dict[str, int]
    sector_rankings: List[Tuple[str, float]]
    rotation_opportunities: List[SectorRotationSignal]
    market_regime: str


class SectorAnalysisEngine:
    """
    セクター別分析エンジン

    TOPIX500全銘柄のセクター分析・業種ローテーション・相関分析
    """

    def __init__(self):
        """初期化"""
        self.master_manager = TOPIX500MasterManager()
        self.sector_data = {}
        self.performance_history = {}

        # 分析パラメータ
        self.lookback_periods = [5, 20, 60, 120]  # 短期・中期・長期分析期間
        self.market_regimes = ["bull", "bear", "sideways", "volatile"]

        logger.info("セクター別分析エンジン初期化完了")

    def calculate_sector_performance(
        self, sector_data: Dict[str, pd.DataFrame], period_days: int = 20
    ) -> Dict[str, SectorPerformance]:
        """
        セクター別パフォーマンス計算

        Args:
            sector_data: セクター別価格データ辞書
            period_days: 分析期間日数

        Returns:
            セクターパフォーマンス辞書
        """
        sector_performances = {}

        try:
            sector_summary = self.master_manager.get_sector_summary()

            for sector_code, sector_info in sector_summary.items():
                if sector_code not in sector_data:
                    continue

                data = sector_data[sector_code].tail(period_days)
                if data.empty:
                    continue

                # リターン計算
                returns = data["Close"].pct_change().dropna()
                if len(returns) == 0:
                    continue

                # パフォーマンス指標計算
                avg_return = returns.mean() * 252  # 年率化
                volatility = returns.std() * np.sqrt(252)  # 年率化
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0

                # 最大ドローダウン計算
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min()

                # 市場相関（仮想的なベンチマーク）
                market_returns = returns  # 簡易版：実際はTOPIXなどを使用
                correlation_with_market = returns.corr(market_returns)

                # モメンタムスコア（過去リターンの持続性）
                momentum_score = self._calculate_momentum_score(returns)

                # バリュースコア（仮想的な指標）
                value_score = self._calculate_value_score(sector_code)

                # クオリティスコア（仮想的な指標）
                quality_score = self._calculate_quality_score(sector_code)

                sector_performances[sector_code] = SectorPerformance(
                    sector_code=sector_code,
                    sector_name=sector_info["sector_name"],
                    stock_count=sector_info["stock_count"],
                    avg_return=float(avg_return),
                    volatility=float(volatility),
                    sharpe_ratio=float(sharpe_ratio),
                    max_drawdown=float(max_drawdown),
                    correlation_with_market=float(correlation_with_market),
                    momentum_score=float(momentum_score),
                    value_score=float(value_score),
                    quality_score=float(quality_score),
                )

            logger.info(
                f"セクターパフォーマンス計算完了: {len(sector_performances)}セクター"
            )
            return sector_performances

        except Exception as e:
            logger.error(f"セクターパフォーマンス計算エラー: {e}")
            return {}

    def _calculate_momentum_score(self, returns: pd.Series) -> float:
        """モメンタムスコア計算"""
        try:
            # 複数期間のリターン加重平均
            if len(returns) < 20:
                return 0.0

            weights = [0.5, 0.3, 0.2]  # 短期・中期・長期の重み
            periods = [5, 20, min(60, len(returns))]

            momentum = 0
            for weight, period in zip(weights, periods):
                if len(returns) >= period:
                    period_return = returns.tail(period).mean()
                    momentum += weight * period_return

            # -1から1の範囲に正規化
            return max(-1, min(1, momentum * 100))

        except Exception:
            return 0.0

    def _calculate_value_score(self, sector_code: str) -> float:
        """バリュースコア計算（簡易版）"""
        try:
            # セクター別のPER・PBR特性を考慮
            sector_characteristics = {
                "7050": 0.8,  # 銀行業：低PER
                "3250": -0.2,  # 医薬品：高PER
                "3700": 0.3,  # 輸送用機器：中程度
                "5250": -0.5,  # 情報通信：高PER
                "8050": 0.6,  # 不動産：割安傾向
            }

            return sector_characteristics.get(sector_code, 0.0)

        except Exception:
            return 0.0

    def _calculate_quality_score(self, sector_code: str) -> float:
        """クオリティスコア計算（簡易版）"""
        try:
            # セクター別の財務健全性特性
            quality_characteristics = {
                "3250": 0.8,  # 医薬品：高収益性
                "7050": 0.4,  # 銀行業：中程度
                "3700": 0.6,  # 輸送用機器：安定
                "5250": 0.3,  # 情報通信：変動大
                "2050": 0.2,  # 建設：低安定性
            }

            return quality_characteristics.get(sector_code, 0.5)

        except Exception:
            return 0.5

    def analyze_sector_correlations(
        self, sector_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        セクター間相関分析

        Args:
            sector_data: セクター別価格データ辞書

        Returns:
            セクター相関行列
        """
        try:
            # 各セクターの日次リターン計算
            sector_returns = pd.DataFrame()

            for sector_code, data in sector_data.items():
                returns = data["Close"].pct_change().dropna()
                sector_returns[sector_code] = returns

            # 相関行列計算
            correlation_matrix = sector_returns.corr()

            logger.info(f"セクター相関分析完了: {len(sector_returns.columns)}セクター")
            return correlation_matrix

        except Exception as e:
            logger.error(f"セクター相関分析エラー: {e}")
            return pd.DataFrame()

    def detect_sector_rotation_signals(
        self,
        sector_performances: Dict[str, SectorPerformance],
        correlation_matrix: pd.DataFrame,
        market_regime: str = "bull",
    ) -> List[SectorRotationSignal]:
        """
        セクターローテーションシグナル検出

        Args:
            sector_performances: セクターパフォーマンス辞書
            correlation_matrix: セクター相関行列
            market_regime: 市場環境

        Returns:
            ローテーションシグナルリスト
        """
        signals = []

        try:
            # パフォーマンスランキング
            performances_list = list(sector_performances.values())

            # モメンタム基準でのローテーション検出
            momentum_sorted = sorted(
                performances_list, key=lambda x: x.momentum_score, reverse=True
            )

            # 上位・下位セクターでのローテーション機会検出
            for i, strong_sector in enumerate(momentum_sorted[:3]):  # 上位3セクター
                for weak_sector in momentum_sorted[-3:]:  # 下位3セクター
                    if strong_sector.sector_code == weak_sector.sector_code:
                        continue

                    # シグナル強度計算
                    momentum_diff = (
                        strong_sector.momentum_score - weak_sector.momentum_score
                    )
                    sharpe_diff = strong_sector.sharpe_ratio - weak_sector.sharpe_ratio

                    signal_strength = (momentum_diff + sharpe_diff) / 2

                    # 相関チェック
                    correlation = 0.0
                    if (
                        strong_sector.sector_code in correlation_matrix.columns
                        and weak_sector.sector_code in correlation_matrix.columns
                    ):
                        correlation = correlation_matrix.loc[
                            strong_sector.sector_code, weak_sector.sector_code
                        ]

                    # 低相関かつ高パフォーマンス差の場合にシグナル生成
                    if abs(correlation) < 0.7 and signal_strength > 0.1:
                        # ローテーションタイプ判定
                        rotation_type = self._determine_rotation_type(
                            strong_sector, weak_sector, market_regime
                        )

                        # 支援要因
                        supporting_factors = []
                        if strong_sector.sharpe_ratio > weak_sector.sharpe_ratio:
                            supporting_factors.append("リスク調整後リターン優位")
                        if strong_sector.momentum_score > 0.3:
                            supporting_factors.append("強いモメンタム")
                        if abs(correlation) < 0.3:
                            supporting_factors.append("低相関でリスク分散効果")

                        signals.append(
                            SectorRotationSignal(
                                from_sector=weak_sector.sector_code,
                                to_sector=strong_sector.sector_code,
                                signal_strength=float(signal_strength),
                                rotation_type=rotation_type,
                                confidence=min(0.9, abs(signal_strength)),
                                supporting_factors=supporting_factors,
                            )
                        )

            # 信頼度で並び替え
            signals.sort(key=lambda x: x.confidence, reverse=True)

            logger.info(f"セクターローテーションシグナル検出: {len(signals)}シグナル")
            return signals[:10]  # 上位10シグナルに限定

        except Exception as e:
            logger.error(f"セクターローテーション検出エラー: {e}")
            return []

    def _determine_rotation_type(
        self,
        strong_sector: SectorPerformance,
        weak_sector: SectorPerformance,
        market_regime: str,
    ) -> str:
        """ローテーションタイプ判定"""
        try:
            # セクター特性に基づく分類
            defensive_sectors = ["4050", "3050", "7150"]  # 電力・食品・保険
            growth_sectors = ["5250", "3250", "3650"]  # IT・医薬品・電気機器
            cyclical_sectors = ["3700", "3450", "2050"]  # 自動車・鉄鋼・建設
            value_sectors = ["7050", "8050", "6050"]  # 銀行・不動産・商社

            strong_type = "other"
            if strong_sector.sector_code in defensive_sectors:
                strong_type = "defensive"
            elif strong_sector.sector_code in growth_sectors:
                strong_type = "growth"
            elif strong_sector.sector_code in cyclical_sectors:
                strong_type = "cyclical"
            elif strong_sector.sector_code in value_sectors:
                strong_type = "value"

            # 市場環境との組み合わせで判定
            if market_regime == "bull" and strong_type == "growth":
                return "growth"
            elif market_regime == "bear" and strong_type == "defensive":
                return "defensive"
            elif strong_sector.value_score > weak_sector.value_score:
                return "value"
            elif strong_type == "cyclical":
                return "cyclical"
            else:
                return "momentum"

        except Exception:
            return "momentum"

    def perform_sector_clustering(
        self, sector_performances: Dict[str, SectorPerformance]
    ) -> Dict[str, int]:
        """
        セクタークラスタリング

        Args:
            sector_performances: セクターパフォーマンス辞書

        Returns:
            セクター別クラスター割当辞書
        """
        try:
            if len(sector_performances) < 3:
                return {}

            # 特徴量準備
            features = []
            sector_codes = []

            for sector_code, perf in sector_performances.items():
                features.append(
                    [
                        perf.avg_return,
                        perf.volatility,
                        perf.sharpe_ratio,
                        perf.momentum_score,
                        perf.value_score,
                        perf.quality_score,
                    ]
                )
                sector_codes.append(sector_code)

            features = np.array(features)

            # 標準化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # クラスタリング実行
            n_clusters = min(4, len(sector_performances))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)

            # 結果辞書作成
            cluster_assignments = {
                sector_codes[i]: int(cluster_labels[i])
                for i in range(len(sector_codes))
            }

            logger.info(f"セクタークラスタリング完了: {n_clusters}クラスター")
            return cluster_assignments

        except Exception as e:
            logger.error(f"セクタークラスタリングエラー: {e}")
            return {}

    def generate_sector_rankings(
        self,
        sector_performances: Dict[str, SectorPerformance],
        ranking_criteria: str = "sharpe",
    ) -> List[Tuple[str, float]]:
        """
        セクターランキング生成

        Args:
            sector_performances: セクターパフォーマンス辞書
            ranking_criteria: ランキング基準

        Returns:
            (セクターコード, スコア)のランキングリスト
        """
        try:
            rankings = []

            for sector_code, perf in sector_performances.items():
                if ranking_criteria == "sharpe":
                    score = perf.sharpe_ratio
                elif ranking_criteria == "return":
                    score = perf.avg_return
                elif ranking_criteria == "momentum":
                    score = perf.momentum_score
                elif ranking_criteria == "quality":
                    score = perf.quality_score
                elif ranking_criteria == "value":
                    score = perf.value_score
                elif ranking_criteria == "composite":
                    # 複合スコア
                    score = (
                        0.3 * perf.sharpe_ratio
                        + 0.2 * perf.momentum_score
                        + 0.2 * perf.quality_score
                        + 0.2 * perf.value_score
                        + 0.1 * min(perf.avg_return, 0.5)  # リターンは上限設定
                    )
                else:
                    score = perf.sharpe_ratio

                rankings.append((sector_code, float(score)))

            # スコア降順でソート
            rankings.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"セクターランキング生成完了: {ranking_criteria}基準")
            return rankings

        except Exception as e:
            logger.error(f"セクターランキング生成エラー: {e}")
            return []

    def analyze_comprehensive_sectors(
        self, sector_data: Dict[str, pd.DataFrame], market_regime: str = "bull"
    ) -> CrossSectorAnalysis:
        """
        包括的セクター分析

        Args:
            sector_data: セクター別価格データ辞書
            market_regime: 市場環境

        Returns:
            クロスセクター分析結果
        """
        try:
            logger.info("包括的セクター分析開始")

            # 1. セクターパフォーマンス計算
            sector_performances = self.calculate_sector_performance(sector_data)

            # 2. セクター相関分析
            correlation_matrix = self.analyze_sector_correlations(sector_data)

            # 3. ローテーションシグナル検出
            rotation_signals = self.detect_sector_rotation_signals(
                sector_performances, correlation_matrix, market_regime
            )

            # 4. セクタークラスタリング
            cluster_assignments = self.perform_sector_clustering(sector_performances)

            # 5. セクターランキング
            sector_rankings = self.generate_sector_rankings(
                sector_performances, "composite"
            )

            analysis_result = CrossSectorAnalysis(
                correlation_matrix=correlation_matrix,
                cluster_assignments=cluster_assignments,
                sector_rankings=sector_rankings,
                rotation_opportunities=rotation_signals,
                market_regime=market_regime,
            )

            logger.info("包括的セクター分析完了")
            return analysis_result

        except Exception as e:
            logger.error(f"包括的セクター分析エラー: {e}")
            return CrossSectorAnalysis(
                correlation_matrix=pd.DataFrame(),
                cluster_assignments={},
                sector_rankings=[],
                rotation_opportunities=[],
                market_regime="unknown",
            )


if __name__ == "__main__":
    print("=== セクター別分析エンジン テスト ===")

    try:
        # エンジン初期化
        analyzer = SectorAnalysisEngine()

        print("1. テスト用セクターデータ生成...")
        # モックデータ生成
        test_sectors = ["3700", "7050", "3250", "5250", "8050"]
        sector_data = {}

        for sector_code in test_sectors:
            # 各セクター特性を反映したモックデータ生成
            dates = pd.date_range(start="2023-01-01", periods=100)
            np.random.seed(int(sector_code))

            base_price = 2000 + int(sector_code) % 1000
            returns = np.random.normal(0.001, 0.02, 100)

            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))

            sector_data[sector_code] = pd.DataFrame(
                {
                    "Open": prices[:-1],
                    "High": [p * 1.02 for p in prices[:-1]],
                    "Low": [p * 0.98 for p in prices[:-1]],
                    "Close": prices[1:],
                    "Volume": np.random.randint(1000000, 5000000, 100),
                },
                index=dates,
            )

        print(f"   生成セクター数: {len(sector_data)}")

        print("2. セクターパフォーマンス分析...")
        performances = analyzer.calculate_sector_performance(sector_data)

        for sector_code, perf in performances.items():
            print(
                f"   {sector_code}: リターン{perf.avg_return:.1%}, "
                f"シャープ{perf.sharpe_ratio:.2f}, モメンタム{perf.momentum_score:.2f}"
            )

        print("3. セクター相関分析...")
        correlation_matrix = analyzer.analyze_sector_correlations(sector_data)
        print(f"   相関行列サイズ: {correlation_matrix.shape}")

        print("4. ローテーションシグナル検出...")
        rotation_signals = analyzer.detect_sector_rotation_signals(
            performances, correlation_matrix, "bull"
        )

        for i, signal in enumerate(rotation_signals[:3]):
            print(
                f"   シグナル{i+1}: {signal.from_sector}→{signal.to_sector} "
                f"(強度{signal.signal_strength:.2f}, {signal.rotation_type})"
            )

        print("5. セクタークラスタリング...")
        clusters = analyzer.perform_sector_clustering(performances)
        cluster_summary = {}
        for sector, cluster_id in clusters.items():
            if cluster_id not in cluster_summary:
                cluster_summary[cluster_id] = []
            cluster_summary[cluster_id].append(sector)

        for cluster_id, sectors in cluster_summary.items():
            print(f"   クラスター{cluster_id}: {sectors}")

        print("6. セクターランキング...")
        rankings = analyzer.generate_sector_rankings(performances, "composite")
        for i, (sector, score) in enumerate(rankings[:3]):
            print(f"   {i+1}位: {sector} (スコア: {score:.3f})")

        print("7. 包括的分析実行...")
        comprehensive = analyzer.analyze_comprehensive_sectors(sector_data, "bull")
        print(f"   ローテーション機会: {len(comprehensive.rotation_opportunities)}件")
        print(f"   セクターランキング: {len(comprehensive.sector_rankings)}セクター")
        print(f"   市場環境: {comprehensive.market_regime}")

        print("\n[OK] セクター別分析エンジン テスト完了！")

    except Exception as e:
        print(f"[NG] テストエラー: {e}")
        import traceback

        traceback.print_exc()
