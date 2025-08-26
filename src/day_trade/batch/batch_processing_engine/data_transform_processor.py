#!/usr/bin/env python3
"""
データ変換プロセッサー
Issue #376: バッチ処理の強化

データ変換ステージの処理ロジック
"""

import time
from typing import Any, Dict

from .core_types import StageResult, WorkflowStage
from .stage_processor import StageProcessor


class DataTransformProcessor(StageProcessor):
    """データ変換プロセッサー"""

    def __init__(self):
        super().__init__(WorkflowStage.DATA_TRANSFORM)

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ変換処理"""
        start_time = time.time()

        try:
            transform_rules = parameters.get("transform_rules", {})
            output_format = parameters.get("output_format", "dict")

            if not data or not isinstance(data, dict):
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="Invalid input data for transformation",
                )

            transformed_data = {}

            # データタイプ別変換
            for data_type, type_data in data.items():
                if data_type == "current_price":
                    # 価格データの正規化
                    normalized_prices = []
                    for symbol, price_data in type_data.items():
                        normalized = {
                            "symbol": symbol,
                            "price": float(price_data.get("price", 0)),
                            "change": float(price_data.get("change", 0)),
                            "change_percent": float(
                                price_data.get("change_percent", 0)
                            ),
                            "volume": int(price_data.get("volume", 0)),
                            "timestamp": price_data.get("timestamp", time.time()),
                        }

                        # カスタム変換ルール適用
                        if "price_scaling" in transform_rules:
                            normalized["price"] *= transform_rules["price_scaling"]

                        normalized_prices.append(normalized)

                    transformed_data[data_type] = normalized_prices

                elif data_type == "historical_data":
                    # ヒストリカルデータの変換
                    processed_historical = []
                    for symbol, hist_data in type_data.items():
                        if "data" in hist_data and hist_data["data"]:
                            for record in hist_data["data"]:
                                processed_record = {
                                    "symbol": symbol,
                                    "date": record.get("date", ""),
                                    "open": float(record.get("Open", 0)),
                                    "high": float(record.get("High", 0)),
                                    "low": float(record.get("Low", 0)),
                                    "close": float(record.get("Close", 0)),
                                    "volume": int(record.get("Volume", 0)),
                                }
                                processed_historical.append(processed_record)

                    transformed_data[data_type] = processed_historical

            # 出力形式変換
            if output_format == "flat":
                # フラット形式（全データを単一リストに）
                flat_data = []
                for type_data in transformed_data.values():
                    if isinstance(type_data, list):
                        flat_data.extend(type_data)
                final_data = flat_data
            else:
                final_data = transformed_data

            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=True,
                data=final_data,
                processing_time_ms=processing_time_ms,
                statistics={
                    "input_data_types": len(data),
                    "output_format": output_format,
                    "total_records_transformed": sum(
                        len(v) if isinstance(v, list) else 0
                        for v in transformed_data.values()
                    ),
                },
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=False,
                processing_time_ms=processing_time_ms,
                error_message=str(e),
            )