"""
ポートフォリオ分析機能
保有銘柄の管理とパフォーマンス分析を行う
"""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.stock_fetcher import StockFetcher
from .trade_manager import Position, TradeManager
from ..utils.logging_config import get_context_logger, log_performance_metric, log_error_with_context

logger = get_context_logger(__name__)


@dataclass
class PortfolioMetrics:
    """ポートフォリオ指標"""

    total_value: Decimal  # 総資産額
    total_cost: Decimal  # 総取得コスト
    total_pnl: Decimal  # 総損益
    total_pnl_percent: Decimal  # 総損益率

    # リスク指標
    volatility: Optional[float] = None  # ボラティリティ
    sharpe_ratio: Optional[float] = None  # シャープレシオ
    max_drawdown: Optional[float] = None  # 最大ドローダウン

    # パフォーマンス指標
    total_return: Optional[float] = None  # トータルリターン
    annualized_return: Optional[float] = None  # 年率リターン
    win_ratio: Optional[float] = None  # 勝率

    # 多様化指標
    concentration_risk: Optional[float] = None  # 集中リスク
    sector_diversity: Optional[float] = None  # セクター多様性


@dataclass
class SectorAllocation:
    """セクター配分"""

    sector: str
    value: Decimal
    percentage: Decimal
    positions: int


@dataclass
class PerformanceReport:
    """パフォーマンスレポート"""

    date: datetime
    metrics: PortfolioMetrics
    sector_allocations: List[SectorAllocation]
    top_performers: List[Tuple[str, Decimal]]  # 上位銘柄（銘柄, 損益率）
    worst_performers: List[Tuple[str, Decimal]]  # 下位銘柄（銘柄, 損益率）

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "date": self.date.isoformat(),
            "metrics": asdict(self.metrics),
            "sector_allocations": [asdict(alloc) for alloc in self.sector_allocations],
            "top_performers": [
                (symbol, str(pnl)) for symbol, pnl in self.top_performers
            ],
            "worst_performers": [
                (symbol, str(pnl)) for symbol, pnl in self.worst_performers
            ],
        }


class PortfolioAnalyzer:
    """ポートフォリオ分析クラス"""

    def __init__(
        self,
        trade_manager: TradeManager,
        stock_fetcher: Optional[StockFetcher] = None,
        risk_free_rate: float = 0.001,  # リスクフリーレート（年率0.1%）
    ):
        """
        Args:
            trade_manager: 取引記録管理インスタンス
            stock_fetcher: 株価データ取得インスタンス
            risk_free_rate: リスクフリーレート（年率）
        """
        self.trade_manager = trade_manager
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.risk_free_rate = risk_free_rate

        # セクター情報のマッピング（簡易版）
        self.sector_mapping = {
            "7203": "Automotive",  # トヨタ
            "8306": "Financial",  # 三菱UFJ
            "9984": "Technology",  # ソフトバンク
            "6758": "Technology",  # ソニー
            "4689": "Technology",  # Z Holdings
        }

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """ポートフォリオ指標を計算"""
        summary = self.trade_manager.get_portfolio_summary()
        positions = self.trade_manager.get_all_positions()

        # 基本指標（文字列からDecimalに変換）
        total_value = Decimal(summary["total_market_value"])
        total_cost = Decimal(summary["total_cost"])
        total_pnl = Decimal(summary["total_unrealized_pnl"])
        total_pnl_percent = (
            (total_pnl / total_cost * 100) if total_cost > 0 else Decimal("0")
        )

        # リスク・パフォーマンス指標の計算
        volatility = self._calculate_portfolio_volatility(positions)
        sharpe_ratio = self._calculate_sharpe_ratio(positions, volatility)
        max_drawdown = self._calculate_max_drawdown(positions)

        # その他の指標
        total_return = float(total_pnl_percent) / 100 if total_pnl_percent else None
        win_ratio = self._calculate_win_ratio()
        concentration_risk = self._calculate_concentration_risk(positions)
        sector_diversity = self._calculate_sector_diversity(positions)

        return PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_ratio=win_ratio,
            concentration_risk=concentration_risk,
            sector_diversity=sector_diversity,
        )

    def get_sector_allocation(self) -> List[SectorAllocation]:
        """セクター別配分を取得"""
        positions = self.trade_manager.get_all_positions()
        sector_data = defaultdict(lambda: {"value": Decimal("0"), "positions": 0})

        total_value = Decimal("0")

        # セクター別集計
        for symbol, position in positions.items():
            sector = self.sector_mapping.get(symbol, "Other")
            market_value = position.market_value

            sector_data[sector]["value"] += market_value
            sector_data[sector]["positions"] += 1
            total_value += market_value

        # パーセンテージ計算
        allocations = []
        for sector, data in sector_data.items():
            percentage = (
                (data["value"] / total_value * 100) if total_value > 0 else Decimal("0")
            )

            allocations.append(
                SectorAllocation(
                    sector=sector,
                    value=data["value"],
                    percentage=percentage,
                    positions=data["positions"],
                )
            )

        # 配分順でソート
        return sorted(allocations, key=lambda x: x.percentage, reverse=True)

    def get_performance_rankings(
        self, top_n: int = 5
    ) -> Tuple[List[Tuple[str, Decimal]], List[Tuple[str, Decimal]]]:
        """パフォーマンスランキングを取得"""
        positions = self.trade_manager.get_all_positions()

        performance_list = []
        for symbol, position in positions.items():
            performance_list.append((symbol, position.unrealized_pnl_percent))

        # パフォーマンス順でソート
        performance_list.sort(key=lambda x: x[1], reverse=True)

        top_performers = performance_list[:top_n]
        worst_performers = performance_list[-top_n:][::-1]  # 下位を逆順で

        return top_performers, worst_performers

    def generate_performance_report(self) -> PerformanceReport:
        """パフォーマンスレポートを生成"""
        metrics = self.get_portfolio_metrics()
        sector_allocations = self.get_sector_allocation()
        top_performers, worst_performers = self.get_performance_rankings()

        return PerformanceReport(
            date=datetime.now(),
            metrics=metrics,
            sector_allocations=sector_allocations,
            top_performers=top_performers,
            worst_performers=worst_performers,
        )

    def _calculate_portfolio_volatility(
        self, positions: Dict[str, Position]
    ) -> Optional[float]:
        """ポートフォリオのボラティリティを計算"""
        if not positions:
            return None

        try:
            # 各銘柄の過去データを取得してボラティリティを計算
            returns_data = []
            weights = []
            total_value = sum(pos.market_value for pos in positions.values())

            for symbol, position in positions.items():
                try:
                    # 過去30日のデータを取得
                    hist_data = self.stock_fetcher.get_historical_data(
                        symbol, period="1mo", interval="1d"
                    )
                    if hist_data is not None and len(hist_data) > 1:
                        # 日次リターンを計算
                        daily_returns = hist_data["Close"].pct_change().dropna()
                        if len(daily_returns) > 0:
                            returns_data.append(daily_returns.values)
                            weights.append(float(position.market_value / total_value))

                except Exception as e:
                    logger.warning("Symbol volatility calculation failed, excluding from portfolio calculation",
                                 symbol=symbol, error=str(e))
                    continue

            if not returns_data:
                return None

            # ポートフォリオのボラティリティを計算（簡易版）
            # 実際には共分散行列を使用すべきだが、ここでは加重平均で近似
            portfolio_variance = 0
            for i, returns in enumerate(returns_data):
                portfolio_variance += (weights[i] ** 2) * (
                    np.var(returns) * 252
                )  # 年率化

            return math.sqrt(portfolio_variance)

        except Exception as e:
            log_error_with_context(e, {
                "operation": "portfolio_volatility_calculation",
                "position_count": len(positions)
            })
            return None

    def _calculate_sharpe_ratio(
        self, positions: Dict[str, Position], volatility: Optional[float]
    ) -> Optional[float]:
        """シャープレシオを計算"""
        if not positions or volatility is None or volatility == 0:
            return None

        try:
            summary = self.trade_manager.get_portfolio_summary()
            total_cost = Decimal(summary["total_cost"])
            if total_cost == 0:
                return None

            # 年率リターンを計算（簡易版）
            total_pnl = Decimal(summary["total_unrealized_pnl"])
            total_return_rate = float(total_pnl / total_cost)
            excess_return = total_return_rate - self.risk_free_rate

            return excess_return / volatility

        except Exception as e:
            log_error_with_context(e, {
                "operation": "sharpe_ratio_calculation",
                "volatility": volatility,
                "risk_free_rate": self.risk_free_rate
            })
            return None

    def _calculate_max_drawdown(
        self, positions: Dict[str, Position]
    ) -> Optional[float]:
        """最大ドローダウンを計算"""
        if not positions:
            return None

        # 簡易版：現在の含み損益から推定
        try:
            pnl_values = [
                float(pos.unrealized_pnl_percent) for pos in positions.values()
            ]
            if not pnl_values:
                return None

            # 最大の含み損を最大ドローダウンとして近似
            return abs(min(pnl_values, default=0)) / 100

        except Exception as e:
            log_error_with_context(e, {
                "operation": "max_drawdown_calculation",
                "position_count": len(positions)
            })
            return None

    def _calculate_win_ratio(self) -> Optional[float]:
        """勝率を計算"""
        try:
            realized_pnl = self.trade_manager.realized_pnl
            if not realized_pnl:
                return None

            wins = sum(1 for pnl in realized_pnl if pnl.realized_pnl > 0)
            total = len(realized_pnl)

            return wins / total if total > 0 else None

        except Exception as e:
            log_error_with_context(e, {
                "operation": "win_ratio_calculation",
                "realized_pnl_count": len(self.trade_manager.realized_pnl)
            })
            return None

    def _calculate_concentration_risk(
        self, positions: Dict[str, Position]
    ) -> Optional[float]:
        """集中リスク（ハーフィンダール指数）を計算"""
        if not positions:
            return None

        try:
            total_value = sum(pos.market_value for pos in positions.values())
            if total_value == 0:
                return None

            # 各銘柄の割合を計算
            weights = [
                float(pos.market_value / total_value) for pos in positions.values()
            ]

            # ハーフィンダール指数（集中度）
            hhi = sum(w**2 for w in weights)

            return hhi

        except Exception as e:
            log_error_with_context(e, {
                "operation": "concentration_risk_calculation",
                "position_count": len(positions)
            })
            return None

    def _calculate_sector_diversity(
        self, positions: Dict[str, Position]
    ) -> Optional[float]:
        """セクター多様性を計算"""
        if not positions:
            return None

        try:
            sector_values = defaultdict(Decimal)
            total_value = Decimal("0")

            for symbol, position in positions.items():
                sector = self.sector_mapping.get(symbol, "Other")
                sector_values[sector] += position.market_value
                total_value += position.market_value

            if total_value == 0:
                return None

            # セクター別ウェイトを計算
            sector_weights = [
                float(value / total_value) for value in sector_values.values()
            ]

            # シャノンエントロピーで多様性を測定
            entropy = -sum(w * math.log(w) for w in sector_weights if w > 0)

            # 正規化（最大エントロピーで割る）
            max_entropy = math.log(len(sector_weights))
            normalized_diversity = entropy / max_entropy if max_entropy > 0 else 0

            return normalized_diversity

        except Exception as e:
            log_error_with_context(e, {
                "operation": "sector_diversity_calculation",
                "position_count": len(positions)
            })
            return None

    def export_report_to_csv(self, filename: str) -> None:
        """レポートをCSVファイルにエクスポート"""
        try:
            report = self.generate_performance_report()

            # メトリクス情報
            metrics_data = {"metric": [], "value": []}

            metrics_dict = asdict(report.metrics)
            for key, value in metrics_dict.items():
                metrics_data["metric"].append(key)
                metrics_data["value"].append(str(value) if value is not None else "N/A")

            df_metrics = pd.DataFrame(metrics_data)

            # ファイル名にタイムスタンプを付与
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = filename.replace(".csv", "")

            # メトリクスファイル
            metrics_file = f"{base_name}_metrics_{timestamp}.csv"
            df_metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")

            # セクター配分ファイル
            if report.sector_allocations:
                sector_data = []
                for alloc in report.sector_allocations:
                    sector_data.append(
                        {
                            "sector": alloc.sector,
                            "value": str(alloc.value),
                            "percentage": str(alloc.percentage),
                            "positions": alloc.positions,
                        }
                    )

                df_sectors = pd.DataFrame(sector_data)
                sector_file = f"{base_name}_sectors_{timestamp}.csv"
                df_sectors.to_csv(sector_file, index=False, encoding="utf-8-sig")

            logger.info("Portfolio report exported to CSV",
                       metrics_file=metrics_file,
                       sector_file=sector_file if report.sector_allocations else None)

        except Exception as e:
            log_error_with_context(e, {
                "operation": "portfolio_csv_export",
                "filename": filename
            })
            raise

    def export_report_to_json(self, filename: str) -> None:
        """レポートをJSONファイルにエクスポート"""
        try:
            report = self.generate_performance_report()
            report_dict = report.to_dict()

            # Decimalを文字列に変換
            def decimal_converter(obj):
                if isinstance(obj, Decimal):
                    return str(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = filename.replace(".json", f"_{timestamp}.json")

            with open(output_file, "w", encoding="utf-8") as f:
                import json

                json.dump(
                    report_dict,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=decimal_converter,
                )

            logger.info("Portfolio report exported to JSON", output_file=output_file)

        except Exception as e:
            log_error_with_context(e, {
                "operation": "portfolio_json_export",
                "filename": filename
            })
            raise


# 使用例
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # TradeManagerとの連携例
    from decimal import Decimal

    from .trade_manager import TradeManager, TradeType

    # サンプルデータでテスト
    tm = TradeManager()

    # サンプル取引を追加
    tm.add_trade("7203", TradeType.BUY, 100, Decimal("2500"))
    tm.add_trade("8306", TradeType.BUY, 50, Decimal("800"))
    tm.add_trade("9984", TradeType.BUY, 10, Decimal("15000"))

    # 現在価格を更新
    tm.update_current_prices(
        {"7203": Decimal("2600"), "8306": Decimal("850"), "9984": Decimal("14500")}
    )

    # ポートフォリオ分析
    analyzer = PortfolioAnalyzer(tm)

    # メトリクス表示
    metrics = analyzer.get_portfolio_metrics()
    logger = get_context_logger(__name__)
    logger.info(
        "ポートフォリオメトリクス",
        section="portfolio_metrics",
        total_value=float(metrics.total_value),
        total_pnl=float(metrics.total_pnl),
        total_pnl_percent=float(metrics.total_pnl_percent),
        volatility=float(metrics.volatility) if metrics.volatility else None,
        sharpe_ratio=float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None
    )

    # セクター配分
    allocations = analyzer.get_sector_allocation()
    for alloc in allocations:
        logger.info(
            "セクター配分",
            section="sector_allocation",
            sector=alloc.sector,
            percentage=float(alloc.percentage),
            value=float(alloc.value)
        )

    # パフォーマンスランキング
    top, worst = analyzer.get_performance_rankings(3)

    logger.info("パフォーマンスランキング開始", section="performance_ranking")
    for symbol, pnl_pct in top:
        logger.info(
            "パフォーマンス上位",
            section="performance_ranking_top",
            symbol=symbol,
            pnl_percent=float(pnl_pct)
        )

    for symbol, pnl_pct in worst:
        logger.info(
            "パフォーマンス下位",
            section="performance_ranking_worst",
            symbol=symbol,
            pnl_percent=float(pnl_pct)
        )
