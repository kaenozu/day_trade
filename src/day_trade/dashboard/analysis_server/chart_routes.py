"""
インタラクティブチャートAPIルーティング

価格チャート、パフォーマンスチャート、リスク分析チャートAPI
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ...utils.logging_config import get_context_logger
from .app_config import get_chart_manager, get_analysis_engine

logger = get_context_logger(__name__)

router = APIRouter(prefix="/api/charts", tags=["charts"])


@router.post("/price")
async def create_price_chart(request: dict):
    """価格チャート作成"""
    chart_manager = get_chart_manager()
    analysis_engine = get_analysis_engine()

    if not chart_manager or not analysis_engine:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        symbols = request.get("symbols", ["7203.T"])
        chart_type = request.get("chartType", "candlestick")
        indicators = request.get("indicators", [])
        show_volume = request.get("showVolume", True)

        # モックデータ生成（実データに置き換え可能）
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        data = {}

        for symbol in symbols:
            base_price = 2000 + hash(symbol) % 1000
            prices = base_price + np.cumsum(np.random.normal(0, 20, len(dates)))

            data[symbol] = pd.DataFrame(
                {
                    "Open": prices + np.random.normal(0, 10, len(dates)),
                    "High": prices + np.abs(np.random.normal(10, 5, len(dates))),
                    "Low": prices - np.abs(np.random.normal(10, 5, len(dates))),
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

        chart_data = chart_manager.create_price_chart(
            data=data,
            symbols=symbols,
            chart_type=chart_type,
            show_volume=show_volume,
            indicators=indicators,
        )

        return chart_data

    except Exception as e:
        logger.error(f"価格チャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@router.post("/performance")
async def create_performance_chart(request: dict):
    """パフォーマンスチャート作成"""
    chart_manager = get_chart_manager()

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        symbols = request.get("symbols", ["7203.T"])

        # モックパフォーマンスデータ
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        performance_data = {}

        for symbol in symbols:
            # ランダムリターンデータ生成
            returns = np.random.normal(0.001, 0.02, len(dates))
            performance_data[f"{symbol} 戦略"] = returns.tolist()

        # ベンチマークデータ
        benchmark = {"TOPIX": np.random.normal(0.0005, 0.018, len(dates)).tolist()}

        chart_data = chart_manager.create_performance_chart(
            performance_data=performance_data,
            dates=dates.to_pydatetime().tolist(),
            benchmark=benchmark,
            cumulative=True,
        )

        return chart_data

    except Exception as e:
        logger.error(f"パフォーマンスチャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@router.post("/risk-scatter")
async def create_risk_scatter_chart(risk_data: dict):
    """リスク散布図チャート作成"""
    chart_manager = get_chart_manager()

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        chart_data = chart_manager.create_risk_analysis_chart(
            risk_data=risk_data, chart_type="scatter"
        )
        return chart_data

    except Exception as e:
        logger.error(f"リスク散布図作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@router.post("/risk-radar")
async def create_risk_radar_chart(risk_data: dict):
    """リスクレーダーチャート作成"""
    chart_manager = get_chart_manager()

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        chart_data = chart_manager.create_risk_analysis_chart(
            risk_data=risk_data, chart_type="radar"
        )
        return chart_data

    except Exception as e:
        logger.error(f"リスクレーダーチャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@router.get("/market-overview")
async def get_market_overview():
    """市場概要ダッシュボード取得"""
    chart_manager = get_chart_manager()

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        # モック市場データ
        market_data = {
            "sector_performance": {
                "テクノロジー": 0.15,
                "金融": 0.08,
                "ヘルスケア": 0.12,
                "消費財": 0.06,
                "エネルギー": -0.03,
            },
            "volume_trend": {
                "dates": pd.date_range(start="2024-12-01", end="2024-12-31", freq="D")
                .strftime("%Y-%m-%d")
                .tolist(),
                "volumes": np.random.randint(1000000, 5000000, 31).tolist(),
            },
            "price_distribution": np.random.normal(2500, 500, 1000).tolist(),
            "correlation_matrix": {
                "symbols": ["7203.T", "8306.T", "9984.T"],
                "values": [[1.0, 0.3, 0.1], [0.3, 1.0, 0.6], [0.1, 0.6, 1.0]],
            },
        }

        chart_data = chart_manager.create_market_overview_dashboard(
            market_data=market_data, layout="grid"
        )

        return chart_data

    except Exception as e:
        logger.error(f"市場概要ダッシュボード作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ダッシュボード作成エラー: {str(e)}"
        ) from e


@router.get("/enhanced", response_class=HTMLResponse)
async def enhanced_dashboard():
    """強化版ダッシュボードページ"""
    template_path = "src/day_trade/dashboard/templates/enhanced_dashboard.html"

    try:
        with open(template_path, encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>強化版ダッシュボード</h1><p>テンプレートファイルが見つかりません</p>",
            status_code=404,
        )


@router.post("/technical-indicators")
async def create_technical_indicators_chart(request: dict):
    """テクニカル指標チャート作成"""
    chart_manager = get_chart_manager()

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        symbol = request.get("symbol", "7203.T")
        indicators = request.get("indicators", ["RSI", "MACD", "Bollinger"])
        period = request.get("period", 252)

        # モックデータ生成
        dates = pd.date_range(end=pd.Timestamp.now(), periods=period, freq="D")
        base_price = 2500
        prices = base_price + np.cumsum(np.random.normal(0, 20, period))
        
        # テクニカル指標の計算（簡略版）
        indicator_data = {}
        
        if "RSI" in indicators:
            # RSI（14日移動平均）
            rsi_values = np.random.uniform(20, 80, period)
            indicator_data["RSI"] = rsi_values.tolist()
            
        if "MACD" in indicators:
            # MACD
            macd_line = np.random.normal(0, 5, period)
            signal_line = np.random.normal(0, 3, period)
            indicator_data["MACD"] = {
                "macd": macd_line.tolist(),
                "signal": signal_line.tolist(),
                "histogram": (macd_line - signal_line).tolist()
            }
            
        if "Bollinger" in indicators:
            # ボリンジャーバンド
            middle = np.convolve(prices, np.ones(20)/20, mode='same')
            std = np.std(prices)
            indicator_data["Bollinger"] = {
                "upper": (middle + 2 * std).tolist(),
                "middle": middle.tolist(),
                "lower": (middle - 2 * std).tolist()
            }

        return {
            "symbol": symbol,
            "dates": dates.strftime("%Y-%m-%d").tolist(),
            "prices": prices.tolist(),
            "indicators": indicator_data,
            "generated_at": pd.Timestamp.now().isoformat()
        }

    except Exception as e:
        logger.error(f"テクニカル指標チャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@router.get("/chart-templates")
async def get_chart_templates():
    """利用可能チャートテンプレート一覧取得"""
    chart_manager = get_chart_manager()

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    templates = [
        {
            "id": "price_basic",
            "name": "基本価格チャート",
            "description": "ローソク足と出来高",
            "type": "price"
        },
        {
            "id": "price_with_indicators",
            "name": "テクニカル指標付き価格チャート",
            "description": "価格チャート + RSI, MACD等",
            "type": "price"
        },
        {
            "id": "performance_comparison",
            "name": "パフォーマンス比較",
            "description": "複数戦略のパフォーマンス比較",
            "type": "performance"
        },
        {
            "id": "risk_scatter",
            "name": "リスク・リターン散布図",
            "description": "リスクとリターンの関係性",
            "type": "risk"
        },
        {
            "id": "market_overview",
            "name": "市場概要ダッシュボード",
            "description": "セクター別パフォーマンス等",
            "type": "market"
        }
    ]

    return {
        "success": True,
        "templates": templates,
        "count": len(templates)
    }