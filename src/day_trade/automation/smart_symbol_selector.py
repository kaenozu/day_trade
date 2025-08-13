#!/usr/bin/env python3
"""
スマート銘柄自動選択システム

Issue #487対応: 完全自動化システム実装 - Phase 1
市場流動性・出来高・ボラティリティによる自動銘柄選定
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MarketSegment(Enum):
    """市場セグメント"""
    MEGA_CAP = "mega_cap"      # 超大型株
    LARGE_CAP = "large_cap"    # 大型株  
    MID_CAP = "mid_cap"        # 中型株
    SMALL_CAP = "small_cap"    # 小型株
    

@dataclass
class SymbolMetrics:
    """銘柄メトリクス"""
    symbol: str
    market_cap: float            # 時価総額
    avg_volume: float           # 平均出来高
    price: float                # 現在価格
    volatility: float           # ボラティリティ(%)
    liquidity_score: float      # 流動性スコア (0-100)
    volume_consistency: float   # 出来高安定性 (0-100)
    price_trend: str           # 価格トレンド ("up", "down", "sideways")
    market_segment: MarketSegment
    selection_score: float      # 選定スコア (0-100)


@dataclass 
class SelectionCriteria:
    """選定基準"""
    min_market_cap: float = 1e9     # 最小時価総額（10億円）
    min_avg_volume: float = 1e6     # 最小平均出来高（100万株）
    max_volatility: float = 8.0     # 最大ボラティリティ（8%）
    min_liquidity_score: float = 60.0  # 最小流動性スコア
    target_symbols: int = 10        # 目標銘柄数
    exclude_sectors: List[str] = None  # 除外セクター


class SmartSymbolSelector:
    """スマート銘柄自動選択システム"""
    
    def __init__(self):
        """初期化"""
        # 日本市場の代表的な銘柄プール（拡張可能）
        self.symbol_pool = {
            # 大型株
            "7203.T": "トヨタ自動車",
            "6758.T": "ソニーグループ", 
            "6861.T": "キーエンス",
            "4519.T": "中外製薬",
            "6098.T": "リクルートホールディングス",
            "8058.T": "三菱商事",
            "9984.T": "ソフトバンクグループ",
            "4063.T": "信越化学工業",
            "7741.T": "HOYA",
            "6954.T": "ファナック",
            
            # 中型株・新興株
            "4755.T": "楽天グループ",
            "3659.T": "ネクソン",
            "4689.T": "Zホールディングス",
            "2269.T": "明治ホールディングス",
            "8002.T": "丸紅",
            "4502.T": "武田薬品工業",
            "6367.T": "ダイキン工業",
            "9433.T": "KDDI",
            "2914.T": "JT",
            "8316.T": "三井住友フィナンシャルグループ",
            
            # テック・成長株
            "4385.T": "メルカリ",
            "4477.T": "BASE",
            "3092.T": "ZOZO",
            "3928.T": "マイネット",
            "4751.T": "サイバーエージェント",
            "2432.T": "ディー・エヌ・エー",
            "4704.T": "トレンドマイクロ",
            "6501.T": "日立製作所",
            "9432.T": "NTT",
            "4307.T": "野村総合研究所"
        }
        
        # Issue #487対応: スマート選定アルゴリズムのパラメータ
        self.scoring_weights = {
            'liquidity': 0.35,      # 流動性重要度
            'volatility': 0.25,     # ボラティリティ重要度
            'volume_consistency': 0.20,  # 出来高安定性重要度
            'market_cap': 0.20      # 時価総額重要度
        }
        
        # データ取得設定
        self.data_period = "60d"  # 過去60日のデータ
        self.max_concurrent = 5   # 同時データ取得数
        
    async def select_optimal_symbols(self, criteria: SelectionCriteria = None) -> List[str]:
        """
        Issue #487対応: 最適銘柄の自動選択
        
        市場流動性・出来高・ボラティリティを総合評価して
        最適な銘柄リストを自動生成
        
        Args:
            criteria: 選定基準（未指定時はデフォルト）
            
        Returns:
            選定された銘柄シンボルリスト
        """
        logger.info("🤖 スマート銘柄自動選択を開始")
        start_time = time.time()
        
        if criteria is None:
            criteria = SelectionCriteria()
            
        # Step 1: 銘柄メトリクス並列計算
        logger.info(f"📊 {len(self.symbol_pool)}銘柄のメトリクス分析中...")
        symbol_metrics = await self._calculate_symbol_metrics(list(self.symbol_pool.keys()))
        
        # Step 2: 選定基準によるフィルタリング
        logger.info("🔍 選定基準による銘柄フィルタリング中...")
        filtered_metrics = self._filter_symbols(symbol_metrics, criteria)
        
        # Step 3: スコアリングによる最適銘柄選定
        logger.info("🎯 総合スコアによる最適銘柄選定中...")
        selected_symbols = self._rank_and_select(filtered_metrics, criteria.target_symbols)
        
        # Step 4: 結果ログ出力
        selection_time = time.time() - start_time
        logger.info(f"✅ 銘柄選択完了: {len(selected_symbols)}銘柄選定 ({selection_time:.1f}秒)")
        
        # 選定結果詳細ログ
        self._log_selection_results(selected_symbols, filtered_metrics)
        
        return selected_symbols
    
    async def _calculate_symbol_metrics(self, symbols: List[str]) -> List[SymbolMetrics]:
        """
        Issue #487対応: 銘柄メトリクス並列計算
        
        Args:
            symbols: 分析対象銘柄リスト
            
        Returns:
            銘柄メトリクスリスト
        """
        # 並列処理でデータ取得・計算を最適化
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._calculate_single_metric(symbol, semaphore) for symbol in symbols]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 成功した結果のみ抽出
        valid_metrics = []
        for result in results:
            if isinstance(result, SymbolMetrics):
                valid_metrics.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"銘柄メトリクス計算エラー: {result}")
                
        logger.info(f"📈 {len(valid_metrics)}/{len(symbols)}銘柄のメトリクス計算完了")
        return valid_metrics
    
    async def _calculate_single_metric(self, symbol: str, semaphore: asyncio.Semaphore) -> SymbolMetrics:
        """
        個別銘柄のメトリクス計算
        
        Args:
            symbol: 銘柄シンボル
            semaphore: 同期セマフォ
            
        Returns:
            銘柄メトリクス
        """
        async with semaphore:
            try:
                # yfinanceからデータ取得
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period=self.data_period)
                
                if hist.empty or len(hist) < 20:
                    raise ValueError(f"データ不足: {symbol}")
                
                # 基本メトリクス計算
                current_price = hist['Close'].iloc[-1]
                market_cap = info.get('marketCap', 0)
                avg_volume = hist['Volume'].mean()
                
                # ボラティリティ計算（過去30日の日次リターン標準偏差 * sqrt(252)）
                returns = hist['Close'].pct_change().dropna()
                daily_vol = returns.std()
                annualized_vol = daily_vol * np.sqrt(252) * 100  # パーセント表示
                
                # 流動性スコア計算
                liquidity_score = self._calculate_liquidity_score(hist, market_cap)
                
                # 出来高安定性計算
                volume_consistency = self._calculate_volume_consistency(hist)
                
                # 価格トレンド判定
                price_trend = self._determine_price_trend(hist)
                
                # 市場セグメント分類
                market_segment = self._classify_market_segment(market_cap)
                
                # Issue #487対応: 総合選定スコア計算
                selection_score = self._calculate_selection_score(
                    liquidity_score, annualized_vol, volume_consistency, market_cap
                )
                
                return SymbolMetrics(
                    symbol=symbol,
                    market_cap=market_cap,
                    avg_volume=avg_volume,
                    price=current_price,
                    volatility=annualized_vol,
                    liquidity_score=liquidity_score,
                    volume_consistency=volume_consistency,
                    price_trend=price_trend,
                    market_segment=market_segment,
                    selection_score=selection_score
                )
                
            except Exception as e:
                logger.warning(f"銘柄 {symbol} メトリクス計算エラー: {e}")
                raise
    
    def _calculate_liquidity_score(self, hist: pd.DataFrame, market_cap: float) -> float:
        """
        流動性スコア計算
        
        Args:
            hist: 価格履歴データ
            market_cap: 時価総額
            
        Returns:
            流動性スコア (0-100)
        """
        try:
            # 平均売買代金
            avg_turnover = (hist['Volume'] * hist['Close']).mean()
            
            # 出来高/発行済株式（流動性比率の代理指標）
            shares_outstanding = market_cap / hist['Close'].iloc[-1] if hist['Close'].iloc[-1] > 0 else 1
            volume_ratio = (hist['Volume'].mean() / shares_outstanding) * 100
            
            # ビッド・アスク・スプレッドの代理指標（価格変動幅）
            avg_spread_proxy = (hist['High'] - hist['Low']).mean() / hist['Close'].mean() * 100
            
            # 流動性スコア計算（売買代金重視）
            turnover_score = min(avg_turnover / 1e8 * 30, 60)  # 1億円で30点、上限60点
            volume_score = min(volume_ratio * 2, 25)  # 上限25点
            spread_score = max(15 - avg_spread_proxy, 0)  # スプレッドが狭いほど高得点、上限15点
            
            total_score = turnover_score + volume_score + spread_score
            return min(total_score, 100)
            
        except Exception as e:
            logger.debug(f"流動性スコア計算エラー: {e}")
            return 50.0  # デフォルトスコア
    
    def _calculate_volume_consistency(self, hist: pd.DataFrame) -> float:
        """
        出来高安定性計算
        
        Args:
            hist: 価格履歴データ
            
        Returns:
            出来高安定性スコア (0-100)
        """
        try:
            volumes = hist['Volume']
            
            # 出来高の変動係数（標準偏差/平均）
            vol_cv = volumes.std() / volumes.mean() if volumes.mean() > 0 else float('inf')
            
            # 異常出来高日の割合
            vol_median = volumes.median()
            abnormal_days = len(volumes[volumes > vol_median * 3]) / len(volumes)
            
            # 出来高トレンドの安定性（移動平均からの乖離）
            vol_ma20 = volumes.rolling(20).mean()
            trend_stability = 1 - (volumes - vol_ma20).abs().mean() / vol_ma20.mean()
            
            # 総合安定性スコア
            cv_score = max(100 - vol_cv * 20, 0)  # 変動係数が小さいほど高得点
            abnormal_score = max(100 - abnormal_days * 100, 0)  # 異常日が少ないほど高得点
            stability_score = max(trend_stability * 100, 0) if not np.isnan(trend_stability) else 50
            
            return (cv_score + abnormal_score + stability_score) / 3
            
        except Exception as e:
            logger.debug(f"出来高安定性計算エラー: {e}")
            return 50.0
    
    def _determine_price_trend(self, hist: pd.DataFrame) -> str:
        """
        価格トレンド判定
        
        Args:
            hist: 価格履歴データ
            
        Returns:
            価格トレンド ("up", "down", "sideways")
        """
        try:
            closes = hist['Close']
            
            # 短期・長期移動平均
            ma5 = closes.rolling(5).mean().iloc[-1]
            ma20 = closes.rolling(20).mean().iloc[-1] 
            ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else ma20
            current_price = closes.iloc[-1]
            
            # トレンド判定
            if current_price > ma5 > ma20 > ma50:
                return "up"
            elif current_price < ma5 < ma20 < ma50:
                return "down"
            else:
                return "sideways"
                
        except Exception:
            return "sideways"
    
    def _classify_market_segment(self, market_cap: float) -> MarketSegment:
        """
        市場セグメント分類
        
        Args:
            market_cap: 時価総額
            
        Returns:
            市場セグメント
        """
        if market_cap >= 10e12:  # 10兆円以上
            return MarketSegment.MEGA_CAP
        elif market_cap >= 3e12:  # 3兆円以上
            return MarketSegment.LARGE_CAP
        elif market_cap >= 1e12:  # 1兆円以上
            return MarketSegment.MID_CAP
        else:
            return MarketSegment.SMALL_CAP
    
    def _calculate_selection_score(self, liquidity: float, volatility: float, 
                                 volume_consistency: float, market_cap: float) -> float:
        """
        Issue #487対応: 総合選定スコア計算
        
        Args:
            liquidity: 流動性スコア
            volatility: ボラティリティ(%)
            volume_consistency: 出来高安定性スコア
            market_cap: 時価総額
            
        Returns:
            選定スコア (0-100)
        """
        # ボラティリティスコア（適度なボラティリティを好む）
        vol_score = 100 - abs(volatility - 4.0) * 10  # 4%付近が最適
        vol_score = max(vol_score, 0)
        
        # 時価総額スコア
        cap_score = min(market_cap / 1e12 * 20, 80)  # 1兆円で20点、上限80点
        
        # 加重平均による総合スコア
        total_score = (
            liquidity * self.scoring_weights['liquidity'] +
            vol_score * self.scoring_weights['volatility'] +
            volume_consistency * self.scoring_weights['volume_consistency'] +
            cap_score * self.scoring_weights['market_cap']
        )
        
        return min(total_score, 100)
    
    def _filter_symbols(self, metrics: List[SymbolMetrics], criteria: SelectionCriteria) -> List[SymbolMetrics]:
        """
        選定基準によるフィルタリング
        
        Args:
            metrics: 銘柄メトリクスリスト
            criteria: 選定基準
            
        Returns:
            フィルタリング済み銘柄メトリクス
        """
        filtered = []
        
        for metric in metrics:
            # 基準チェック
            if (metric.market_cap >= criteria.min_market_cap and
                metric.avg_volume >= criteria.min_avg_volume and
                metric.volatility <= criteria.max_volatility and
                metric.liquidity_score >= criteria.min_liquidity_score):
                
                filtered.append(metric)
        
        logger.info(f"🔍 フィルタリング結果: {len(filtered)}/{len(metrics)}銘柄が基準を満たしています")
        return filtered
    
    def _rank_and_select(self, metrics: List[SymbolMetrics], target_count: int) -> List[str]:
        """
        ランキングによる最終選定
        
        Args:
            metrics: フィルタリング済み銘柄メトリクス
            target_count: 目標銘柄数
            
        Returns:
            選定された銘柄シンボルリスト
        """
        # 選定スコアでソート
        sorted_metrics = sorted(metrics, key=lambda x: x.selection_score, reverse=True)
        
        # 上位銘柄選定（市場セグメント分散も考慮）
        selected = []
        segment_counts = {segment: 0 for segment in MarketSegment}
        max_per_segment = max(target_count // 4, 2)  # セグメント当たり最大数
        
        for metric in sorted_metrics:
            if len(selected) >= target_count:
                break
                
            # セグメント分散確保
            if segment_counts[metric.market_segment] < max_per_segment or len(selected) >= target_count * 0.8:
                selected.append(metric.symbol)
                segment_counts[metric.market_segment] += 1
        
        return selected
    
    def _log_selection_results(self, selected_symbols: List[str], all_metrics: List[SymbolMetrics]):
        """
        選定結果の詳細ログ出力
        
        Args:
            selected_symbols: 選定された銘柄
            all_metrics: 全銘柄メトリクス
        """
        logger.info("=" * 60)
        logger.info("🎯 スマート銘柄選択結果")
        logger.info("=" * 60)
        
        # 選定銘柄の詳細
        selected_metrics = [m for m in all_metrics if m.symbol in selected_symbols]
        selected_metrics.sort(key=lambda x: x.selection_score, reverse=True)
        
        for i, metric in enumerate(selected_metrics, 1):
            name = self.symbol_pool.get(metric.symbol, metric.symbol)
            logger.info(
                f"{i:2d}. {metric.symbol} ({name})\n"
                f"    スコア: {metric.selection_score:.1f} | "
                f"流動性: {metric.liquidity_score:.1f} | "
                f"ボラ: {metric.volatility:.1f}% | "
                f"出来高安定性: {metric.volume_consistency:.1f}\n"
                f"    時価総額: {metric.market_cap/1e12:.2f}兆円 | "
                f"トレンド: {metric.price_trend} | "
                f"セグメント: {metric.market_segment.value}"
            )
        
        # 統計サマリー
        avg_score = np.mean([m.selection_score for m in selected_metrics])
        avg_vol = np.mean([m.volatility for m in selected_metrics])
        segment_dist = {}
        for metric in selected_metrics:
            segment_dist[metric.market_segment.value] = segment_dist.get(metric.market_segment.value, 0) + 1
        
        logger.info("-" * 60)
        logger.info(f"📊 選定銘柄統計:")
        logger.info(f"   平均選定スコア: {avg_score:.1f}")
        logger.info(f"   平均ボラティリティ: {avg_vol:.1f}%")
        logger.info(f"   セグメント分布: {segment_dist}")
        logger.info("=" * 60)

    def get_symbol_info(self, symbol: str) -> Optional[str]:
        """銘柄名取得"""
        return self.symbol_pool.get(symbol, None)


async def get_smart_selected_symbols(target_count: int = 10) -> List[str]:
    """
    Issue #487対応: スマート銘柄選択の外部インターフェース
    
    Args:
        target_count: 目標銘柄数
        
    Returns:
        自動選択された銘柄リスト
    """
    selector = SmartSymbolSelector()
    criteria = SelectionCriteria(target_symbols=target_count)
    
    return await selector.select_optimal_symbols(criteria)


# デバッグ用メイン関数
async def main():
    """デバッグ用メイン"""
    logger.info("🤖 スマート銘柄選択システム テスト実行")
    
    symbols = await get_smart_selected_symbols(target_count=8)
    
    logger.info(f"✅ 選定完了: {symbols}")


if __name__ == "__main__":
    asyncio.run(main())