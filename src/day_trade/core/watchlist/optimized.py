"""
ウォッチリスト管理 - 最適化処理機能
パフォーマンスを最適化したウォッチリスト処理（一括JOIN、キャッシュ利用等）を提供
"""

from typing import Any, Dict, List, Optional

from ...models import PriceData, Stock, WatchlistItem, db_manager
from ...utils.logging_config import get_context_logger, log_error_with_context

logger = get_context_logger(__name__)


class WatchlistOptimized:
    """最適化されたウォッチリスト処理機能を提供するクラス"""

    def get_watchlist_optimized(
        self, group_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        最適化されたウォッチリスト取得（一括JOIN）

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            価格情報付きウォッチリストアイテムのリスト
        """
        try:
            with db_manager.session_scope() as session:
                # WatchlistItemとStockを一括JOINで取得
                query = session.query(WatchlistItem, Stock).join(
                    Stock, WatchlistItem.stock_code == Stock.code
                )

                if group_name:
                    query = query.filter(WatchlistItem.group_name == group_name)

                items = query.all()

                # 銘柄コードを抽出して一括で価格データを取得
                stock_codes = [item.WatchlistItem.stock_code for item in items]

                if stock_codes:
                    # 最適化されたメソッドで一括取得
                    latest_prices = PriceData.get_latest_prices(session, stock_codes)
                else:
                    latest_prices = {}

                # 結果を構築
                result = []
                for item in items:
                    watchlist_item = item.WatchlistItem
                    stock = item.Stock
                    code = watchlist_item.stock_code
                    price_data = latest_prices.get(code)

                    result.append(
                        {
                            "stock_code": code,
                            "stock_name": stock.name,
                            "group_name": watchlist_item.group_name,
                            "memo": watchlist_item.memo,
                            "added_date": watchlist_item.created_at,
                            "sector": stock.sector,
                            "industry": stock.industry,
                            "current_price": price_data.close if price_data else None,
                            "volume": price_data.volume if price_data else None,
                            "last_updated": price_data.datetime if price_data else None,
                        }
                    )

                return result

        except Exception as e:
            log_error_with_context(
                e, {"operation": "get_watchlist_optimized", "group_name": group_name}
            )
            return []

    def get_watchlist_with_technical_data(
        self, group_name: Optional[str] = None, include_indicators: bool = True
    ) -> List[Dict[str, Any]]:
        """
        テクニカル指標付きの最適化されたウォッチリスト取得

        Args:
            group_name: グループ名（指定しない場合は全て）
            include_indicators: テクニカル指標を含めるかどうか

        Returns:
            テクニカル指標付きウォッチリストアイテムのリスト
        """
        try:
            # 基本的なウォッチリストデータを取得
            basic_data = self.get_watchlist_optimized(group_name)
            
            if not basic_data or not include_indicators:
                return basic_data

            # テクニカル指標を計算・追加
            enhanced_data = []
            for item in basic_data:
                code = item["stock_code"]
                
                # 過去データを取得してテクニカル指標を計算
                # （実際の実装では、キャッシュされた指標データを使用することを推奨）
                technical_data = self._get_cached_technical_indicators(code)
                
                # 基本データにテクニカル指標を追加
                enhanced_item = {**item, **technical_data}
                enhanced_data.append(enhanced_item)

            return enhanced_data

        except Exception as e:
            log_error_with_context(
                e, 
                {
                    "operation": "get_watchlist_with_technical_data", 
                    "group_name": group_name,
                    "include_indicators": include_indicators
                }
            )
            return []

    def get_watchlist_performance_summary(
        self, group_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ウォッチリストのパフォーマンスサマリーを最適化取得

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            パフォーマンスサマリー情報
        """
        try:
            with db_manager.session_scope() as session:
                # 基本統計を一括クエリで取得
                query = session.query(
                    WatchlistItem.group_name,
                    Stock.code,
                    Stock.name,
                    Stock.sector,
                    Stock.industry
                ).join(Stock, WatchlistItem.stock_code == Stock.code)

                if group_name:
                    query = query.filter(WatchlistItem.group_name == group_name)

                items = query.all()

                # 銘柄コードを抽出
                stock_codes = [item.code for item in items]
                
                if not stock_codes:
                    return {
                        "total_stocks": 0,
                        "sectors": {},
                        "industries": {},
                        "groups": {},
                        "price_summary": {}
                    }

                # 価格データを一括取得
                latest_prices = PriceData.get_latest_prices(session, stock_codes)

                # 統計情報を計算
                sectors = {}
                industries = {}
                groups = {}
                price_data = []

                for item in items:
                    # セクター統計
                    sector = item.sector or "未分類"
                    sectors[sector] = sectors.get(sector, 0) + 1

                    # 業界統計
                    industry = item.industry or "未分類"
                    industries[industry] = industries.get(industry, 0) + 1

                    # グループ統計
                    group = item.group_name
                    groups[group] = groups.get(group, 0) + 1

                    # 価格データ
                    price_info = latest_prices.get(item.code)
                    if price_info:
                        price_data.append(price_info.close)

                # 価格統計を計算
                price_summary = {}
                if price_data:
                    price_summary = {
                        "count": len(price_data),
                        "average": sum(price_data) / len(price_data),
                        "min": min(price_data),
                        "max": max(price_data)
                    }

                return {
                    "total_stocks": len(items),
                    "sectors": sectors,
                    "industries": industries,
                    "groups": groups,
                    "price_summary": price_summary
                }

        except Exception as e:
            log_error_with_context(
                e, {"operation": "get_watchlist_performance_summary", "group_name": group_name}
            )
            return {
                "total_stocks": 0,
                "sectors": {},
                "industries": {},
                "groups": {},
                "price_summary": {}
            }

    def get_watchlist_batch_by_group(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        グループごとのウォッチリストを一括取得（最適化版）

        Returns:
            グループ名をキーとしたウォッチリストの辞書
        """
        try:
            # 全データを一度に取得
            all_data = self.get_watchlist_optimized()
            
            # グループごとに分類
            grouped_data = {}
            for item in all_data:
                group = item["group_name"]
                if group not in grouped_data:
                    grouped_data[group] = []
                grouped_data[group].append(item)

            return grouped_data

        except Exception as e:
            log_error_with_context(e, {"operation": "get_watchlist_batch_by_group"})
            return {}

    def _get_cached_technical_indicators(self, stock_code: str) -> Dict[str, Any]:
        """
        キャッシュされたテクニカル指標を取得（プレースホルダー）

        Args:
            stock_code: 証券コード

        Returns:
            テクニカル指標データ
        """
        try:
            # 実際の実装では、キャッシュシステム（Redis等）から取得
            # または、別途計算済みのテクニカル指標テーブルから取得
            
            # プレースホルダー実装
            return {
                "rsi": None,  # 実際は計算済みのRSI値
                "macd": None,  # 実際は計算済みのMACD値
                "bollinger_upper": None,  # ボリンジャーバンド上限
                "bollinger_lower": None,  # ボリンジャーバンド下限
                "sma_20": None,  # 20日移動平均
                "sma_50": None,  # 50日移動平均
                "volume_sma_20": None,  # 出来高20日移動平均
            }

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "get_cached_technical_indicators",
                    "stock_code": stock_code,
                },
            )
            return {}

    def prefetch_watchlist_data(self, group_names: Optional[List[str]] = None) -> bool:
        """
        ウォッチリストデータを事前取得してキャッシュ（最適化）

        Args:
            group_names: プリフェッチ対象のグループリスト（指定しない場合は全て）

        Returns:
            プリフェッチに成功した場合True
        """
        try:
            # 実際の実装では、バックグラウンドタスクでデータを事前取得し、
            # キャッシュシステム（Redis等）に保存する

            target_groups = group_names or []
            
            if not target_groups:
                # 全グループを対象とする
                from .groups import WatchlistGroups
                groups_manager = WatchlistGroups()
                target_groups = groups_manager.get_groups()

            # 各グループのデータを事前取得
            prefetched_count = 0
            for group in target_groups:
                try:
                    data = self.get_watchlist_optimized(group)
                    # ここでキャッシュに保存
                    # cache_system.set(f"watchlist:{group}", data, expire=300)
                    prefetched_count += len(data)
                except Exception as e:
                    logger.warning(f"Failed to prefetch data for group {group}: {e}")

            logger.info(f"Prefetched watchlist data for {len(target_groups)} groups, {prefetched_count} items total")
            return True

        except Exception as e:
            log_error_with_context(
                e, {"operation": "prefetch_watchlist_data", "group_names": group_names}
            )
            return False

    def get_optimization_metrics(self) -> Dict[str, Any]:
        """
        最適化処理のメトリクス情報を取得

        Returns:
            パフォーマンスメトリクス
        """
        try:
            with db_manager.session_scope() as session:
                # データベース統計
                watchlist_count = session.query(WatchlistItem).count()
                stock_count = session.query(Stock).count()
                
                # インデックス効率性の指標（簡易版）
                indexed_queries = {
                    "watchlist_by_code_group": "WatchlistItem(stock_code, group_name)",
                    "stock_by_code": "Stock(code)",
                    "price_data_by_code": "PriceData(stock_code, datetime)"
                }

                return {
                    "total_watchlist_items": watchlist_count,
                    "total_stocks": stock_count,
                    "indexed_queries": indexed_queries,
                    "optimization_features": [
                        "bulk_join_queries",
                        "batch_price_retrieval", 
                        "cached_technical_indicators",
                        "prefetch_capabilities"
                    ],
                    "recommended_cache_duration": {
                        "price_data": "5_minutes",
                        "technical_indicators": "1_hour",
                        "watchlist_data": "15_minutes"
                    }
                }

        except Exception as e:
            log_error_with_context(e, {"operation": "get_optimization_metrics"})
            return {
                "total_watchlist_items": 0,
                "total_stocks": 0,
                "indexed_queries": {},
                "optimization_features": [],
                "recommended_cache_duration": {}
            }