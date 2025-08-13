#!/usr/bin/env python3
"""
Issue #577対応: バッチデータ取得ヘルパー関数モジュール

batch_data_fetcher.py から分離されたユーティリティ関数群
"""

from typing import Dict, List
from .batch_data_fetcher import AdvancedBatchDataFetcher, DataRequest, DataResponse


def create_data_request(symbol: str, **kwargs) -> DataRequest:
    """データリクエスト作成ヘルパー
    
    Args:
        symbol: 銘柄コード
        **kwargs: DataRequestの追加パラメータ
        
    Returns:
        DataRequest: 作成されたデータリクエスト
    """
    return DataRequest(symbol=symbol, **kwargs)


def fetch_advanced_batch(
    symbols: List[str],
    period: str = "60d",
    preprocessing: bool = True,
    priority: int = 1,
    **kwargs,
) -> Dict[str, DataResponse]:
    """高度バッチ取得（簡易インターフェース）
    
    Args:
        symbols: 銘柄コードリスト
        period: 取得期間
        preprocessing: 前処理実行フラグ
        priority: 優先度
        **kwargs: AdvancedBatchDataFetcherの追加パラメータ
        
    Returns:
        Dict[str, DataResponse]: 銘柄別データレスポンス
    """
    requests = [
        DataRequest(
            symbol=symbol, period=period, preprocessing=preprocessing, priority=priority
        )
        for symbol in symbols
    ]

    fetcher = AdvancedBatchDataFetcher(**kwargs)
    try:
        return fetcher.fetch_batch(requests)
    finally:
        fetcher.close()


def preload_advanced_cache(
    symbols: List[str], periods: List[str] = None, **kwargs
) -> Dict[str, int]:
    """高度キャッシュ事前読み込み
    
    Args:
        symbols: 銘柄コードリスト
        periods: 取得期間リスト
        **kwargs: AdvancedBatchDataFetcherの追加パラメータ
        
    Returns:
        Dict[str, int]: 期間別成功数
    """
    if periods is None:
        periods = ["5d", "60d", "1y"]

    results = {}
    fetcher = AdvancedBatchDataFetcher(**kwargs)

    try:
        for period in periods:
            requests = [
                DataRequest(symbol=symbol, period=period, preprocessing=True)
                for symbol in symbols
            ]

            responses = fetcher.fetch_batch(requests)
            success_count = len([r for r in responses.values() if r.success])
            results[period] = success_count

    finally:
        fetcher.close()

    return results