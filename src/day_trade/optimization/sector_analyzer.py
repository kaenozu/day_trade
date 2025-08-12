#!/usr/bin/env python3
"""
セクター分析システム

セクター別ウェイト管理、業界分析、リバランシング提案
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SectorAnalyzer:
    """
    セクター分析・管理クラス

    セクター別ポートフォリオ分析とリバランシング提案
    """

    def __init__(self, sector_limits: Optional[Dict[str, float]] = None):
        """
        初期化

        Args:
            sector_limits: セクター別上限制約 {sector: max_weight}
        """
        self.sector_limits = sector_limits or {
            "Technology": 0.30,
            "Financial": 0.25,
            "Healthcare": 0.20,
            "Consumer": 0.20,
            "Industrial": 0.25,
            "Transportation": 0.15,
            "Energy": 0.15,
            "Materials": 0.15,
            "DayTrading": 0.20,
            "BioTech": 0.15,
            "Gaming": 0.10,
            "FutureTech": 0.15,
        }

        logger.info("セクターアナライザー初期化:")
        logger.info(f"  - セクター制約数: {len(self.sector_limits)}")

    def load_sector_mapping(self) -> Dict[str, str]:
        """
        銘柄-セクター対応表を読み込み

        Returns:
            {symbol: sector} の辞書
        """
        # 設定ファイルから銘柄情報を読み込み想定
        # 実際の実装では settings.json から読み込む
        sector_mapping = {
            # Technology
            "9984": "Technology",  # ソフトバンクグループ
            "6758": "Technology",  # ソニー
            "4689": "Technology",  # Z Holdings
            "4755": "Technology",  # 楽天グループ
            "3659": "Technology",  # ネクソン
            "9613": "Technology",  # NTTデータ
            "2432": "Technology",  # DeNA
            "4385": "Technology",  # メルカリ
            "4704": "Technology",  # トレンドマイクロ
            "4751": "Technology",  # サイバーエージェント
            # Financial
            "8306": "Financial",  # 三菱UFJ銀行
            "8411": "Financial",  # みずほフィナンシャルグループ
            "8766": "Financial",  # 東京海上
            "8316": "Financial",  # 三井住友フィナンシャルグループ
            "8604": "Financial",  # 野村ホールディングス
            "7182": "Financial",  # ゆうちょ銀行
            "8795": "Financial",  # T&Dホールディングス
            # Transportation
            "7203": "Transportation",  # トヨタ自動車
            "7267": "Transportation",  # ホンダ
            "9101": "Transportation",  # 日本郵船
            "9201": "Transportation",  # 日本航空
            "9202": "Transportation",  # ANA
            # Industrial
            "6861": "Industrial",  # キーエンス
            "7011": "Industrial",  # 三菱重工業
            "6503": "Industrial",  # 三菱電機
            "6954": "Industrial",  # ファナック
            "6367": "Industrial",  # ダイキン工業
            "7751": "Industrial",  # キヤノン
            "6981": "Industrial",  # 村田製作所
            # Healthcare
            "4502": "Healthcare",  # 武田薬品工業
            "4523": "Healthcare",  # エーザイ
            # Consumer
            "2914": "Consumer",  # JT
            "7974": "Consumer",  # 任天堂
            "3382": "Consumer",  # セブン&アイ
            "2801": "Consumer",  # キッコーマン
            "2502": "Consumer",  # アサヒグループ
            "9983": "Consumer",  # ファーストリテイリング
            # Trading
            "8001": "Trading",  # 伊藤忠商事
            "8058": "Trading",  # 三菱商事
            "8031": "Trading",  # 三井物産
            "8053": "Trading",  # 住友商事
            # Materials
            "5401": "Materials",  # 新日鉄住金
            "4005": "Materials",  # 住友化学
            "4061": "Materials",  # デンカ
            # Energy
            "5020": "Energy",  # ENEOS
            # Utilities
            "9501": "Utilities",  # 東京電力
            "9502": "Utilities",  # 中部電力
            # Telecom
            "9434": "Telecom",  # ソフトバンク
            "9437": "Telecom",  # NTTドコモ
            "9432": "Telecom",  # NTT
            # RealEstate
            "8802": "RealEstate",  # 三菱地所
            # Construction
            "1801": "Construction",  # 大成建設
            "1803": "Construction",  # 清水建設
            # DayTrading セクター
            "4478": "DayTrading",  # リンクバル
            "4485": "DayTrading",  # JTOWER
            "4490": "DayTrading",  # ビザスク
            "3900": "DayTrading",  # クラウドワークス
            "3774": "DayTrading",  # インターネットイニシアティブ
            "4382": "DayTrading",  # HEROZ
            "4386": "DayTrading",  # SIG
            "4475": "DayTrading",  # HENNGE
            "4421": "DayTrading",  # DigiCert
            "3655": "DayTrading",  # ブレインパッド
            "3844": "DayTrading",  # コムチュア
            "4833": "DayTrading",  # ぐるなび
            # BioTech セクター
            "4563": "BioTech",  # アンジェス
            "4592": "BioTech",  # サンバイオ
            "4564": "BioTech",  # オンコセラピー
            "4588": "BioTech",  # オンコリスバイオファーマ
            "4596": "BioTech",  # 窪田製薬
            "4591": "BioTech",  # リボミック
            "4565": "BioTech",  # そーせいグループ
            "7707": "BioTech",  # プレシジョン・システム・サイエンス
            # Gaming セクター
            "3692": "Gaming",  # FFRI
            "3656": "Gaming",  # KLab
            "3760": "Gaming",  # ケイブ
            "9449": "Gaming",  # GMOインターネット
            "4726": "Gaming",  # ソフトバンク・テクノロジー
            # FutureTech セクター
            "7779": "FutureTech",  # CYBERDYNE
            "6178": "FutureTech",  # 日本郵政
            "4847": "FutureTech",  # インテリジェント ウェイブ
            "4598": "FutureTech",  # Delta-Fly Pharma
            "4880": "FutureTech",  # セルソース
            # その他
            "4777": "Technology",  # ガーラ
            "3776": "Technology",  # ブロードバンドタワー
        }

        logger.info(f"セクターマッピング読み込み: {len(sector_mapping)}銘柄")
        return sector_mapping

    def analyze_sector_allocation(self, portfolio_weights: Dict[str, float]) -> Dict:
        """
        セクター別配分分析

        Args:
            portfolio_weights: ポートフォリオウェイト

        Returns:
            セクター分析結果
        """
        logger.info("セクター配分分析開始")

        sector_mapping = self.load_sector_mapping()

        # セクター別ウェイト集計
        sector_weights = {}
        unmapped_symbols = []

        for symbol, weight in portfolio_weights.items():
            sector = sector_mapping.get(symbol)
            if sector:
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            else:
                unmapped_symbols.append(symbol)

        # 制約違反チェック
        constraint_violations = []
        for sector, weight in sector_weights.items():
            limit = self.sector_limits.get(sector, 0.5)  # デフォルト50%制限
            if weight > limit:
                constraint_violations.append(
                    {
                        "sector": sector,
                        "current_weight": weight,
                        "limit": limit,
                        "excess": weight - limit,
                    }
                )

        # 分散度分析
        sector_count = len(sector_weights)
        sector_hhi = sum(w**2 for w in sector_weights.values())
        effective_sectors = 1 / sector_hhi if sector_hhi > 0 else 0

        sector_analysis = {
            "sector_weights": sector_weights,
            "constraint_violations": constraint_violations,
            "diversification_metrics": {
                "n_sectors": sector_count,
                "sector_hhi": sector_hhi,
                "effective_sectors": effective_sectors,
                "max_sector_weight": max(sector_weights.values()) if sector_weights else 0,
            },
            "unmapped_symbols": unmapped_symbols,
            "compliance": {
                "is_compliant": len(constraint_violations) == 0,
                "violation_count": len(constraint_violations),
            },
        }

        logger.info("セクター分析完了:")
        logger.info(f"  - セクター数: {sector_count}")
        logger.info(f"  - 制約違反: {len(constraint_violations)}件")
        logger.info(f"  - 実効セクター数: {effective_sectors:.1f}")

        return sector_analysis

    def generate_rebalancing_proposal(
        self,
        current_weights: Dict[str, float],
        target_portfolio: Dict[str, float],
        min_trade_size: float = 0.01,
    ) -> Dict:
        """
        リバランシング提案生成

        Args:
            current_weights: 現在のウェイト
            target_portfolio: 目標ポートフォリオ
            min_trade_size: 最小取引サイズ

        Returns:
            リバランシング提案
        """
        logger.info("リバランシング提案生成開始")

        # 全銘柄リスト作成
        all_symbols = set(current_weights.keys()) | set(target_portfolio.keys())

        # 取引提案計算
        trades = {}
        total_buy_amount = 0
        total_sell_amount = 0

        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_portfolio.get(symbol, 0)
            weight_diff = target_weight - current_weight

            if abs(weight_diff) >= min_trade_size:
                trade_type = "BUY" if weight_diff > 0 else "SELL"
                trades[symbol] = {
                    "symbol": symbol,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "weight_change": weight_diff,
                    "trade_type": trade_type,
                    "trade_amount": abs(weight_diff),
                }

                if weight_diff > 0:
                    total_buy_amount += weight_diff
                else:
                    total_sell_amount += abs(weight_diff)

        # セクター別影響分析
        sector_mapping = self.load_sector_mapping()
        sector_changes = {}

        for symbol, trade in trades.items():
            sector = sector_mapping.get(symbol, "Unknown")
            if sector not in sector_changes:
                sector_changes[sector] = {"net_change": 0, "trades": []}
            sector_changes[sector]["net_change"] += trade["weight_change"]
            sector_changes[sector]["trades"].append(trade)

        # 実行優先度算出
        prioritized_trades = sorted(
            trades.values(),
            key=lambda x: abs(x["weight_change"]),
            reverse=True,
        )

        rebalancing_proposal = {
            "timestamp": pd.Timestamp.now(),
            "summary": {
                "total_trades": len(trades),
                "buy_trades": len([t for t in trades.values() if t["trade_type"] == "BUY"]),
                "sell_trades": len([t for t in trades.values() if t["trade_type"] == "SELL"]),
                "total_buy_amount": total_buy_amount,
                "total_sell_amount": total_sell_amount,
                "net_flow": total_buy_amount - total_sell_amount,
            },
            "trades": trades,
            "prioritized_trades": prioritized_trades,
            "sector_impact": sector_changes,
            "execution_plan": self._generate_execution_plan(prioritized_trades),
            "cost_estimate": self._estimate_transaction_costs(trades),
        }

        logger.info("リバランシング提案完了:")
        logger.info(f"  - 取引数: {len(trades)}")
        logger.info(f"  - 買い/売り比率: {total_buy_amount:.2%}/{total_sell_amount:.2%}")

        return rebalancing_proposal

    def calculate_sector_momentum(
        self, returns_data: pd.DataFrame, lookback_days: int = 20
    ) -> Dict:
        """
        セクター別モメンタム分析

        Args:
            returns_data: 収益率データ
            lookback_days: 過去日数

        Returns:
            セクター別モメンタム
        """
        logger.info(f"セクターモメンタム分析開始: {lookback_days}日間")

        sector_mapping = self.load_sector_mapping()

        # セクター別収益率集計
        sector_returns = {}
        for symbol, sector in sector_mapping.items():
            if symbol in returns_data.columns:
                if sector not in sector_returns:
                    sector_returns[sector] = []
                sector_returns[sector].append(returns_data[symbol])

        # セクター平均収益率計算
        sector_momentum = {}
        for sector, symbol_returns in sector_returns.items():
            if len(symbol_returns) > 0:
                # 等ウェイト平均
                sector_avg_returns = pd.concat(symbol_returns, axis=1).mean(axis=1)

                # 最近のパフォーマンス計算
                recent_returns = sector_avg_returns.tail(lookback_days)
                cumulative_return = (1 + recent_returns).prod() - 1
                volatility = recent_returns.std() * np.sqrt(252)
                sharpe_ratio = (recent_returns.mean() * 252) / volatility if volatility > 0 else 0

                sector_momentum[sector] = {
                    "cumulative_return": cumulative_return,
                    "annualized_volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "n_stocks": len(symbol_returns),
                    "momentum_score": cumulative_return / volatility if volatility > 0 else 0,
                }

        # ランキング生成
        sorted_sectors = sorted(
            sector_momentum.items(),
            key=lambda x: x[1]["cumulative_return"],
            reverse=True,
        )

        momentum_analysis = {
            "lookback_days": lookback_days,
            "sector_momentum": sector_momentum,
            "sector_ranking": [
                {
                    "rank": i + 1,
                    "sector": sector,
                    "cumulative_return": data["cumulative_return"],
                    "momentum_score": data["momentum_score"],
                }
                for i, (sector, data) in enumerate(sorted_sectors)
            ],
            "top_sectors": sorted_sectors[:3],
            "bottom_sectors": sorted_sectors[-3:] if len(sorted_sectors) >= 3 else [],
        }

        logger.info(f"セクターモメンタム分析完了: {len(sector_momentum)}セクター")
        return momentum_analysis

    def _generate_execution_plan(self, prioritized_trades: List[Dict]) -> List[Dict]:
        """実行計画生成"""
        execution_plan = []

        # 売り注文を先に実行
        sell_trades = [t for t in prioritized_trades if t["trade_type"] == "SELL"]
        buy_trades = [t for t in prioritized_trades if t["trade_type"] == "BUY"]

        phase = 1
        for trades, phase_name in [(sell_trades, "売り実行"), (buy_trades, "買い実行")]:
            if trades:
                execution_plan.append(
                    {
                        "phase": phase,
                        "phase_name": phase_name,
                        "trades": trades,
                        "trade_count": len(trades),
                        "estimated_duration_minutes": len(trades) * 2,  # 1取引2分想定
                    }
                )
                phase += 1

        return execution_plan

    def _estimate_transaction_costs(self, trades: Dict) -> Dict:
        """取引コスト推定"""
        # 簡易実装：固定手数料率
        commission_rate = 0.001  # 0.1%
        bid_ask_spread = 0.0005  # 0.05%

        total_cost = 0
        cost_breakdown = {}

        for symbol, trade in trades.items():
            trade_amount = trade["trade_amount"]
            commission = trade_amount * commission_rate
            spread_cost = trade_amount * bid_ask_spread
            total_trade_cost = commission + spread_cost

            cost_breakdown[symbol] = {
                "commission": commission,
                "spread_cost": spread_cost,
                "total_cost": total_trade_cost,
            }
            total_cost += total_trade_cost

        return {
            "total_estimated_cost": total_cost,
            "cost_percentage": total_cost * 100,  # %表示
            "cost_breakdown": cost_breakdown,
        }

    def optimize_sector_allocation(
        self,
        current_weights: Dict[str, float],
        target_return: float,
        risk_tolerance: float = 0.5,
    ) -> Dict:
        """
        セクター制約下での配分最適化

        Args:
            current_weights: 現在のウェイト
            target_return: 目標リターン
            risk_tolerance: リスク許容度

        Returns:
            最適化されたセクター配分
        """
        logger.info("セクター制約最適化開始")

        sector_mapping = self.load_sector_mapping()

        # 現在のセクター配分
        current_sector_analysis = self.analyze_sector_allocation(current_weights)

        # セクター制約違反の修正
        optimized_weights = current_weights.copy()

        for violation in current_sector_analysis["constraint_violations"]:
            sector = violation["sector"]
            limit = violation["limit"]

            # 該当セクターの銘柄を特定
            sector_symbols = [
                symbol
                for symbol, mapped_sector in sector_mapping.items()
                if mapped_sector == sector and symbol in current_weights
            ]

            if sector_symbols:
                # 超過分を比例配分で削減
                sector_total_weight = sum(current_weights.get(s, 0) for s in sector_symbols)
                reduction_factor = limit / sector_total_weight if sector_total_weight > 0 else 0

                for symbol in sector_symbols:
                    if symbol in optimized_weights:
                        optimized_weights[symbol] *= reduction_factor

        # ウェイト正規化
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}

        # 最適化後のセクター分析
        optimized_sector_analysis = self.analyze_sector_allocation(optimized_weights)

        optimization_result = {
            "original_weights": current_weights,
            "optimized_weights": optimized_weights,
            "original_sector_analysis": current_sector_analysis,
            "optimized_sector_analysis": optimized_sector_analysis,
            "optimization_summary": {
                "violations_resolved": len(current_sector_analysis["constraint_violations"]),
                "compliance_achieved": optimized_sector_analysis["compliance"]["is_compliant"],
                "weight_changes": {
                    symbol: optimized_weights.get(symbol, 0) - current_weights.get(symbol, 0)
                    for symbol in set(current_weights.keys()) | set(optimized_weights.keys())
                    if abs(optimized_weights.get(symbol, 0) - current_weights.get(symbol, 0))
                    > 0.001
                },
            },
        }

        logger.info(
            f"セクター最適化完了: 制約違反{len(current_sector_analysis['constraint_violations'])}件解消"
        )
        return optimization_result


if __name__ == "__main__":
    # 使用例
    print("セクターアナライザーテスト")

    # サンプルポートフォリオ
    sample_portfolio = {
        "7203": 0.15,  # トヨタ (Transportation)
        "8306": 0.15,  # 三菱UFJ (Financial)
        "9984": 0.20,  # ソフトバンクGP (Technology)
        "6758": 0.15,  # ソニー (Technology)
        "4563": 0.10,  # アンジェス (BioTech)
        "4755": 0.10,  # 楽天 (Technology)
        "3655": 0.05,  # ブレインパッド (DayTrading)
        "4382": 0.05,  # HEROZ (DayTrading)
        "4592": 0.05,  # サンバイオ (BioTech)
    }

    analyzer = SectorAnalyzer()

    try:
        # セクター分析
        sector_analysis = analyzer.analyze_sector_allocation(sample_portfolio)

        print("\n=== セクター配分分析 ===")
        for sector, weight in sector_analysis["sector_weights"].items():
            print(f"{sector}: {weight:.1%}")

        print(
            f"\n実効セクター数: {sector_analysis['diversification_metrics']['effective_sectors']:.1f}"
        )
        print(f"制約適合性: {sector_analysis['compliance']['is_compliant']}")

        if sector_analysis["constraint_violations"]:
            print("\n=== 制約違反 ===")
            for violation in sector_analysis["constraint_violations"]:
                print(
                    f"{violation['sector']}: {violation['current_weight']:.1%} > {violation['limit']:.1%}"
                )

        # セクター制約最適化
        optimization = analyzer.optimize_sector_allocation(sample_portfolio, target_return=0.10)

        if optimization["optimization_summary"]["weight_changes"]:
            print("\n=== 最適化後の変更 ===")
            for symbol, change in optimization["optimization_summary"]["weight_changes"].items():
                if abs(change) > 0.001:
                    print(f"{symbol}: {change:+.1%}")

    except Exception as e:
        print(f"エラー: {e}")
