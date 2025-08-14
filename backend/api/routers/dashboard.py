# Dashboard API Router
# Day Trade ML System - Issue #803

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging

# Models & Schemas
from ..schemas.dashboard import (
    DashboardResponse,
    KPIMetricsResponse,
    ChartDataResponse,
    TradeHistoryResponse,
    AlertsResponse,
    MarketDataResponse,
    PortfolioResponse,
    SystemStatusResponse,
    DashboardFilters,
    TimeRange
)

# Services
from ..services.dashboard_service import DashboardService
from ..services.ml_service import MLService
from ..services.trading_service import TradingService
from ..services.market_service import MarketService
from ..services.portfolio_service import PortfolioService
from ..services.alert_service import AlertService
from ..services.cache_service import CacheService

# Dependencies
from ..dependencies.auth import get_current_user
from ..dependencies.pagination import get_pagination_params
from ..dependencies.rate_limit import rate_limit

# Models
from ..database.models import User

# Utils
from ..utils.metrics import track_api_call
from ..utils.cache import cache_key_builder


router = APIRouter()


@router.get(
    "/",
    response_model=DashboardResponse,
    summary="ダッシュボード全体データ取得",
    description="ダッシュボードに表示する全てのデータを一括取得"
)
@rate_limit(max_calls=60, window=60)  # 1分間に60回まで
@track_api_call("dashboard_get_all")
async def get_dashboard_data(
    time_range: TimeRange = Query(TimeRange.HOUR_24, description="時間範囲"),
    symbols: Optional[List[str]] = Query(None, description="対象銘柄"),
    current_user: User = Depends(get_current_user),
    dashboard_service: DashboardService = Depends(),
    cache_service: CacheService = Depends()
):
    """ダッシュボード全体データを取得"""
    try:
        # キャッシュキー生成
        cache_key = cache_key_builder(
            "dashboard_data",
            user_id=current_user.id,
            time_range=time_range.value,
            symbols=symbols
        )

        # キャッシュから取得を試行
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            return DashboardResponse(**cached_data)

        # データ取得（並行処理）
        results = await asyncio.gather(
            dashboard_service.get_kpi_metrics(current_user.id, time_range),
            dashboard_service.get_chart_data(current_user.id, time_range, symbols),
            dashboard_service.get_recent_trades(current_user.id, limit=20),
            dashboard_service.get_portfolio_summary(current_user.id),
            dashboard_service.get_market_overview(),
            dashboard_service.get_alerts(current_user.id, limit=10),
            dashboard_service.get_ml_performance(time_range),
            dashboard_service.get_system_status(),
            return_exceptions=True
        )

        # エラーチェック
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Dashboard data error in task {i}: {result}")
                # 部分的なエラーは無視して継続

        kpi_metrics, chart_data, trades, portfolio, market, alerts, ml_performance, system_status = results

        # レスポンス構築
        response_data = DashboardResponse(
            kpi_metrics=kpi_metrics if not isinstance(kpi_metrics, Exception) else None,
            chart_data=chart_data if not isinstance(chart_data, Exception) else None,
            recent_trades=trades if not isinstance(trades, Exception) else [],
            portfolio=portfolio if not isinstance(portfolio, Exception) else None,
            market_data=market if not isinstance(market, Exception) else None,
            alerts=alerts if not isinstance(alerts, Exception) else [],
            ml_performance=ml_performance if not isinstance(ml_performance, Exception) else None,
            system_status=system_status if not isinstance(system_status, Exception) else None,
            last_updated=datetime.utcnow(),
            cache_status="miss"
        )

        # キャッシュに保存（5分間）
        await cache_service.set(cache_key, response_data.dict(), ttl=300)

        return response_data

    except Exception as e:
        logging.error(f"Dashboard data retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="ダッシュボードデータの取得に失敗しました"
        )


@router.get(
    "/kpis",
    response_model=KPIMetricsResponse,
    summary="KPIメトリクス取得",
    description="主要パフォーマンス指標を取得"
)
@rate_limit(max_calls=120, window=60)
@track_api_call("dashboard_get_kpis")
async def get_kpi_metrics(
    time_range: TimeRange = Query(TimeRange.HOUR_24, description="時間範囲"),
    current_user: User = Depends(get_current_user),
    dashboard_service: DashboardService = Depends()
):
    """KPIメトリクスを取得"""
    try:
        metrics = await dashboard_service.get_kpi_metrics(current_user.id, time_range)
        return metrics

    except Exception as e:
        logging.error(f"KPI metrics retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="KPIメトリクスの取得に失敗しました"
        )


@router.get(
    "/charts/{chart_type}",
    response_model=ChartDataResponse,
    summary="チャートデータ取得",
    description="指定されたタイプのチャートデータを取得"
)
@rate_limit(max_calls=100, window=60)
@track_api_call("dashboard_get_chart")
async def get_chart_data(
    chart_type: str,
    time_range: TimeRange = Query(TimeRange.HOUR_24),
    symbols: Optional[List[str]] = Query(None),
    current_user: User = Depends(get_current_user),
    dashboard_service: DashboardService = Depends()
):
    """特定のチャートデータを取得"""
    try:
        chart_data = await dashboard_service.get_specific_chart_data(
            current_user.id,
            chart_type,
            time_range,
            symbols
        )
        return chart_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Chart data retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="チャートデータの取得に失敗しました"
        )


@router.get(
    "/trades",
    response_model=TradeHistoryResponse,
    summary="取引履歴取得",
    description="ユーザーの取引履歴を取得"
)
@rate_limit(max_calls=100, window=60)
@track_api_call("dashboard_get_trades")
async def get_trade_history(
    limit: int = Query(50, ge=1, le=1000, description="取得件数"),
    offset: int = Query(0, ge=0, description="オフセット"),
    symbol: Optional[str] = Query(None, description="銘柄フィルター"),
    status: Optional[str] = Query(None, description="ステータスフィルター"),
    time_range: TimeRange = Query(TimeRange.HOUR_24),
    current_user: User = Depends(get_current_user),
    trading_service: TradingService = Depends()
):
    """取引履歴を取得"""
    try:
        filters = DashboardFilters(
            time_range=time_range,
            symbols=[symbol] if symbol else None,
            status=[status] if status else None
        )

        trades = await trading_service.get_user_trades(
            user_id=current_user.id,
            filters=filters,
            limit=limit,
            offset=offset
        )

        return trades

    except Exception as e:
        logging.error(f"Trade history retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="取引履歴の取得に失敗しました"
        )


@router.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="ポートフォリオ取得",
    description="現在のポートフォリオ情報を取得"
)
@rate_limit(max_calls=60, window=60)
@track_api_call("dashboard_get_portfolio")
async def get_portfolio(
    current_user: User = Depends(get_current_user),
    portfolio_service: PortfolioService = Depends()
):
    """ポートフォリオ情報を取得"""
    try:
        portfolio = await portfolio_service.get_user_portfolio(current_user.id)
        return portfolio

    except Exception as e:
        logging.error(f"Portfolio retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="ポートフォリオの取得に失敗しました"
        )


@router.get(
    "/market",
    response_model=MarketDataResponse,
    summary="市場データ取得",
    description="市場概要データを取得"
)
@rate_limit(max_calls=120, window=60)
@track_api_call("dashboard_get_market")
async def get_market_data(
    symbols: Optional[List[str]] = Query(None, description="対象銘柄"),
    market_service: MarketService = Depends(),
    cache_service: CacheService = Depends()
):
    """市場データを取得"""
    try:
        # キャッシュキー
        cache_key = cache_key_builder("market_data", symbols=symbols)

        # キャッシュから取得
        cached_data = await cache_service.get(cache_key)
        if cached_data:
            return MarketDataResponse(**cached_data)

        # 市場データ取得
        market_data = await market_service.get_market_overview(symbols)

        # キャッシュに保存（1分間）
        await cache_service.set(cache_key, market_data.dict(), ttl=60)

        return market_data

    except Exception as e:
        logging.error(f"Market data retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="市場データの取得に失敗しました"
        )


@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="アラート取得",
    description="ユーザーのアラートを取得"
)
@rate_limit(max_calls=100, window=60)
@track_api_call("dashboard_get_alerts")
async def get_alerts(
    limit: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None, description="重要度フィルター"),
    unread_only: bool = Query(False, description="未読のみ"),
    current_user: User = Depends(get_current_user),
    alert_service: AlertService = Depends()
):
    """アラートを取得"""
    try:
        alerts = await alert_service.get_user_alerts(
            user_id=current_user.id,
            limit=limit,
            severity=severity,
            unread_only=unread_only
        )

        return alerts

    except Exception as e:
        logging.error(f"Alerts retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="アラートの取得に失敗しました"
        )


@router.get(
    "/ml-performance",
    response_model=Dict[str, Any],
    summary="ML性能データ取得",
    description="機械学習モデルの性能データを取得"
)
@rate_limit(max_calls=60, window=60)
@track_api_call("dashboard_get_ml_performance")
async def get_ml_performance(
    time_range: TimeRange = Query(TimeRange.HOUR_24),
    model_name: Optional[str] = Query(None, description="モデル名"),
    ml_service: MLService = Depends()
):
    """ML性能データを取得"""
    try:
        performance = await ml_service.get_model_performance(
            time_range=time_range,
            model_name=model_name
        )

        return performance

    except Exception as e:
        logging.error(f"ML performance retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="ML性能データの取得に失敗しました"
        )


@router.get(
    "/system-status",
    response_model=SystemStatusResponse,
    summary="システム状態取得",
    description="システム全体の状態を取得"
)
@rate_limit(max_calls=30, window=60)
@track_api_call("dashboard_get_system_status")
async def get_system_status(
    dashboard_service: DashboardService = Depends()
):
    """システム状態を取得"""
    try:
        status = await dashboard_service.get_system_status()
        return status

    except Exception as e:
        logging.error(f"System status retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail="システム状態の取得に失敗しました"
        )


@router.post(
    "/refresh",
    summary="ダッシュボードデータ更新",
    description="ダッシュボードデータを強制的に更新"
)
@rate_limit(max_calls=10, window=60)  # 制限的
@track_api_call("dashboard_refresh")
async def refresh_dashboard(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    cache_service: CacheService = Depends(),
    dashboard_service: DashboardService = Depends()
):
    """ダッシュボードデータを強制更新"""
    try:
        # キャッシュクリア
        await cache_service.clear_pattern(f"dashboard_*_{current_user.id}_*")

        # バックグラウンドでデータ再取得
        background_tasks.add_task(
            dashboard_service.preload_user_data,
            current_user.id
        )

        return {
            "message": "ダッシュボードデータの更新を開始しました",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logging.error(f"Dashboard refresh error: {e}")
        raise HTTPException(
            status_code=500,
            detail="ダッシュボードの更新に失敗しました"
        )


@router.get(
    "/export",
    summary="ダッシュボードデータエクスポート",
    description="ダッシュボードデータをエクスポート"
)
@rate_limit(max_calls=5, window=300)  # 5分間に5回まで
@track_api_call("dashboard_export")
async def export_dashboard_data(
    format: str = Query("json", regex="^(json|csv|xlsx)$"),
    time_range: TimeRange = Query(TimeRange.HOUR_24),
    current_user: User = Depends(get_current_user),
    dashboard_service: DashboardService = Depends()
):
    """ダッシュボードデータをエクスポート"""
    try:
        export_data = await dashboard_service.export_user_data(
            user_id=current_user.id,
            format=format,
            time_range=time_range
        )

        if format == "json":
            return JSONResponse(content=export_data)
        else:
            # CSV/XLSX の場合はファイルレスポンス
            from fastapi.responses import FileResponse
            return FileResponse(
                export_data["file_path"],
                media_type=export_data["media_type"],
                filename=export_data["filename"]
            )

    except Exception as e:
        logging.error(f"Dashboard export error: {e}")
        raise HTTPException(
            status_code=500,
            detail="データのエクスポートに失敗しました"
        )