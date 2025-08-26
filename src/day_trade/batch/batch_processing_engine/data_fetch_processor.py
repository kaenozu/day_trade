#!/usr/bin/env python3
"""
データ取得プロセッサー
Issue #376: バッチ処理の強化

データ取得ステージの処理ロジック
"""

import time
from typing import Any, Dict

from .core_types import StageResult, WorkflowStage
from .stage_processor import StageProcessor

# プロジェクトモジュール
try:
    from ...data.unified_api_adapter import (
        APIProvider,
        APIRequest,
        UnifiedAPIAdapter,
    )
except ImportError:
    # モッククラス
    class APIProvider:
        YFINANCE = "yfinance"

    class APIRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class UnifiedAPIAdapter:
        def __init__(self, **kwargs):
            pass

        async def execute_multiple(self, requests):
            return []


class DataFetchProcessor(StageProcessor):
    """データ取得プロセッサー"""

    def __init__(self, api_adapter: UnifiedAPIAdapter):
        super().__init__(WorkflowStage.DATA_FETCH)
        self.api_adapter = api_adapter

    async def process(self, data: Any, parameters: Dict[str, Any]) -> StageResult:
        """データ取得処理"""
        start_time = time.time()

        try:
            # パラメータ解析
            symbols = parameters.get("symbols", [])
            data_types = parameters.get("data_types", ["current_price"])

            if not symbols:
                return StageResult(
                    stage=self.stage,
                    success=False,
                    error_message="Symbols required for data fetch",
                )

            fetch_results = {}

            # データタイプ別に取得
            for data_type in data_types:
                if data_type == "current_price":
                    requests = []
                    for symbol in symbols:
                        requests.append(
                            APIRequest(
                                provider=APIProvider.YFINANCE,
                                endpoint="current_price",
                                parameters={"symbol": symbol},
                                batch_key="price_batch",
                            )
                        )

                    responses = await self.api_adapter.execute_multiple(requests)
                    prices = {}
                    for response in responses:
                        if response.success and response.data:
                            symbol = response.data.get("symbol")
                            if symbol:
                                prices[symbol] = response.data

                    fetch_results[data_type] = prices

                elif data_type == "historical_data":
                    period = parameters.get("period", "1mo")
                    interval = parameters.get("interval", "1d")

                    requests = []
                    for symbol in symbols:
                        requests.append(
                            APIRequest(
                                provider=APIProvider.YFINANCE,
                                endpoint="historical_data",
                                parameters={
                                    "symbol": symbol,
                                    "period": period,
                                    "interval": interval,
                                },
                            )
                        )

                    responses = await self.api_adapter.execute_multiple(requests)
                    historical = {}
                    for response in responses:
                        if response.success and response.data:
                            symbol = response.data.get("symbol")
                            if symbol:
                                historical[symbol] = response.data

                    fetch_results[data_type] = historical

            processing_time_ms = (time.time() - start_time) * 1000

            return StageResult(
                stage=self.stage,
                success=True,
                data=fetch_results,
                processing_time_ms=processing_time_ms,
                statistics={
                    "symbols_requested": len(symbols),
                    "data_types_fetched": len(data_types),
                    "total_records": sum(
                        len(v) if isinstance(v, dict) else 0
                        for v in fetch_results.values()
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