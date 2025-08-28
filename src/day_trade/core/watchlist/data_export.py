"""
ウォッチリスト管理 - データエクスポート機能
ウォッチリストのCSVエクスポート、データ変換機能を提供
"""

from typing import Any, Dict, Optional

from ...utils.logging_config import get_context_logger, log_error_with_context
from .core import WatchlistCore

logger = get_context_logger(__name__)


class WatchlistDataExport:
    """ウォッチリストのデータエクスポート機能を提供するクラス"""

    def __init__(self, watchlist_core: Optional[WatchlistCore] = None):
        self.core = watchlist_core or WatchlistCore()

    def export_watchlist_to_csv(
        self, filename: str, group_name: Optional[str] = None
    ) -> bool:
        """
        ウォッチリストをCSVファイルにエクスポート

        Args:
            filename: 出力ファイル名
            group_name: グループ名（指定しない場合は全て）

        Returns:
            エクスポートに成功した場合True
        """
        try:
            import pandas as pd

            # 価格情報付きウォッチリストを取得
            watchlist_data = self.core.get_watchlist_with_prices(group_name)

            if not watchlist_data:
                logger.warning(
                    "No data available for export", extra={"group_name": group_name}
                )
                return False

            # DataFrameに変換
            df_data = []
            for code, data in watchlist_data.items():
                df_data.append(
                    {
                        "証券コード": code,
                        "銘柄名": data.get("stock_name", ""),
                        "グループ": data.get("group_name", ""),
                        "現在価格": data.get("current_price", 0),
                        "前日比": data.get("change", 0),
                        "変化率(%)": data.get("change_percent", 0),
                        "出来高": data.get("volume", 0),
                        "メモ": data.get("memo", ""),
                        "追加日": data.get("added_date", ""),
                    }
                )

            df = pd.DataFrame(df_data)
            df.to_csv(filename, index=False, encoding="utf-8-sig")

            logger.info(
                "Watchlist exported to CSV",
                filename=filename,
                group_name=group_name,
                item_count=len(watchlist_data),
            )
            return True

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "export_watchlist_to_csv",
                    "filename": filename,
                    "group_name": group_name,
                },
            )
            return False

    def export_watchlist_to_json(
        self, filename: str, group_name: Optional[str] = None
    ) -> bool:
        """
        ウォッチリストをJSONファイルにエクスポート

        Args:
            filename: 出力ファイル名
            group_name: グループ名（指定しない場合は全て）

        Returns:
            エクスポートに成功した場合True
        """
        try:
            import json
            from datetime import datetime
            
            # 価格情報付きウォッチリストを取得
            watchlist_data = self.core.get_watchlist_with_prices(group_name)

            if not watchlist_data:
                logger.warning(
                    "No data available for export", extra={"group_name": group_name}
                )
                return False

            # DateTimeオブジェクトをシリアライズ可能な形式に変換
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            # JSONファイルに出力
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(watchlist_data, f, ensure_ascii=False, indent=2, 
                         default=json_serializer)

            logger.info(
                "Watchlist exported to JSON",
                filename=filename,
                group_name=group_name,
                item_count=len(watchlist_data),
            )
            return True

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "export_watchlist_to_json",
                    "filename": filename,
                    "group_name": group_name,
                },
            )
            return False

    def export_watchlist_to_excel(
        self, filename: str, group_name: Optional[str] = None
    ) -> bool:
        """
        ウォッチリストをExcelファイルにエクスポート

        Args:
            filename: 出力ファイル名
            group_name: グループ名（指定しない場合は全て）

        Returns:
            エクスポートに成功した場合True
        """
        try:
            import pandas as pd

            # 価格情報付きウォッチリストを取得
            watchlist_data = self.core.get_watchlist_with_prices(group_name)

            if not watchlist_data:
                logger.warning(
                    "No data available for export", extra={"group_name": group_name}
                )
                return False

            # DataFrameに変換
            df_data = []
            for code, data in watchlist_data.items():
                df_data.append(
                    {
                        "証券コード": code,
                        "銘柄名": data.get("stock_name", ""),
                        "グループ": data.get("group_name", ""),
                        "現在価格": data.get("current_price", 0),
                        "前日比": data.get("change", 0),
                        "変化率(%)": data.get("change_percent", 0),
                        "出来高": data.get("volume", 0),
                        "メモ": data.get("memo", ""),
                        "追加日": data.get("added_date", ""),
                    }
                )

            df = pd.DataFrame(df_data)
            
            # Excelファイルに出力（複数シートに対応）
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                if group_name:
                    # 特定グループのみの場合
                    df.to_excel(writer, sheet_name=group_name, index=False)
                else:
                    # 全体と各グループに分けて出力
                    df.to_excel(writer, sheet_name="全体", index=False)
                    
                    # グループ別にシートを作成
                    from .groups import WatchlistGroups
                    groups_manager = WatchlistGroups()
                    groups = groups_manager.get_groups()
                    
                    for group in groups:
                        group_data = self.core.get_watchlist_with_prices(group)
                        if group_data:
                            group_df_data = []
                            for code, data in group_data.items():
                                group_df_data.append(
                                    {
                                        "証券コード": code,
                                        "銘柄名": data.get("stock_name", ""),
                                        "グループ": data.get("group_name", ""),
                                        "現在価格": data.get("current_price", 0),
                                        "前日比": data.get("change", 0),
                                        "変化率(%)": data.get("change_percent", 0),
                                        "出来高": data.get("volume", 0),
                                        "メモ": data.get("memo", ""),
                                        "追加日": data.get("added_date", ""),
                                    }
                                )
                            group_df = pd.DataFrame(group_df_data)
                            # シート名が長すぎる場合の対策
                            sheet_name = group[:31] if len(group) > 31 else group
                            group_df.to_excel(writer, sheet_name=sheet_name, index=False)

            logger.info(
                "Watchlist exported to Excel",
                filename=filename,
                group_name=group_name,
                item_count=len(watchlist_data),
            )
            return True

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "export_watchlist_to_excel",
                    "filename": filename,
                    "group_name": group_name,
                },
            )
            return False

    def get_export_summary(self, group_name: Optional[str] = None) -> Dict[str, Any]:
        """
        エクスポート可能なデータのサマリーを取得

        Args:
            group_name: グループ名（指定しない場合は全て）

        Returns:
            エクスポートサマリー情報
        """
        try:
            watchlist_data = self.core.get_watchlist_with_prices(group_name)
            
            if not watchlist_data:
                return {
                    "total_stocks": 0,
                    "groups": [],
                    "has_price_data": False,
                    "available_fields": []
                }

            # 利用可能なフィールドを抽出
            available_fields = set()
            has_price_data = False
            groups = set()
            
            for data in watchlist_data.values():
                available_fields.update(data.keys())
                if "current_price" in data and data["current_price"] is not None:
                    has_price_data = True
                if "group_name" in data:
                    groups.add(data["group_name"])

            return {
                "total_stocks": len(watchlist_data),
                "groups": sorted(list(groups)),
                "has_price_data": has_price_data,
                "available_fields": sorted(list(available_fields)),
                "supported_formats": ["csv", "json", "excel"]
            }

        except Exception as e:
            log_error_with_context(
                e, {"operation": "get_export_summary", "group_name": group_name}
            )
            return {
                "total_stocks": 0,
                "groups": [],
                "has_price_data": False,
                "available_fields": []
            }