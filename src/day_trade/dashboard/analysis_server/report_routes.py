"""
カスタムレポートAPIルーティング

レポート作成、エクスポート、テンプレート管理、定期レポート設定API
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from ...utils.logging_config import get_context_logger
from .app_config import get_report_manager, get_analysis_engine

logger = get_context_logger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/create")
async def create_custom_report(request: dict):
    """カスタムレポート作成"""
    report_manager = get_report_manager()
    analysis_engine = get_analysis_engine()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        template = request.get("template", "standard")
        custom_metrics = request.get("metrics", [])
        title = request.get("title")

        # サンプルデータ（実際の実装では分析エンジンからデータ取得）
        sample_data = {
            "portfolio_returns": np.random.normal(0.001, 0.02, 252).tolist(),
            "benchmark_returns": np.random.normal(0.0008, 0.018, 252).tolist(),
            "symbols": ["7203.T", "8306.T", "9984.T", "6758.T", "6861.T"],
            "sector_weights": {
                "technology": 0.35,
                "finance": 0.25,
                "healthcare": 0.20,
                "consumer": 0.15,
                "energy": 0.05,
            },
        }

        report_result = report_manager.create_custom_report(
            data=sample_data,
            template=template,
            custom_metrics=custom_metrics,
            title=title,
        )

        return report_result

    except Exception as e:
        logger.error(f"カスタムレポート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"レポート作成エラー: {str(e)}"
        ) from e


@router.post("/export")
async def export_report(request: dict):
    """レポートエクスポート"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        report_data = request.get("report_data")
        export_format = request.get("format", "json")  # json, excel, pdf
        filename = request.get("filename")

        if not report_data:
            raise HTTPException(
                status_code=400, detail="レポートデータが提供されていません"
            )

        if export_format == "json":
            filepath = report_manager.export_to_json(report_data, filename)
        elif export_format == "excel":
            filepath = report_manager.export_to_excel(report_data, filename)
        elif export_format == "pdf":
            try:
                filepath = report_manager.export_to_pdf(report_data, filename)
            except ImportError as import_error:
                raise HTTPException(
                    status_code=501, detail="PDFエクスポートライブラリが利用できません"
                ) from import_error
        else:
            raise HTTPException(
                status_code=400, detail="サポートされていないエクスポート形式"
            )

        return {
            "success": True,
            "filepath": filepath,
            "format": export_format,
            "message": f"{export_format.upper()}ファイルが正常にエクスポートされました",
        }

    except Exception as e:
        logger.error(f"レポートエクスポートエラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"エクスポートエラー: {str(e)}"
        ) from e


@router.get("/templates")
async def get_report_templates():
    """利用可能レポートテンプレート一覧取得"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        templates = report_manager.get_available_templates()
        return {"success": True, "templates": templates}

    except Exception as e:
        logger.error(f"テンプレート取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"テンプレート取得エラー: {str(e)}"
        ) from e


@router.get("/metrics")
async def get_available_metrics():
    """利用可能指標一覧取得"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        metrics = report_manager.get_available_metrics()
        return {"success": True, "metrics": metrics}

    except Exception as e:
        logger.error(f"指標取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"指標取得エラー: {str(e)}"
        ) from e


@router.post("/schedule")
async def schedule_periodic_report(request: dict):
    """定期レポートスケジュール設定"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        template = request.get("template", "standard")
        frequency = request.get("frequency", "monthly")  # daily, weekly, monthly
        recipients = request.get("recipients", [])

        if frequency not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="無効な頻度が指定されました")

        schedule_result = report_manager.schedule_periodic_report(
            template=template, frequency=frequency, recipients=recipients
        )

        return schedule_result

    except Exception as e:
        logger.error(f"定期レポートスケジュール設定エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"スケジュール設定エラー: {str(e)}"
        ) from e


@router.get("/history")
async def get_report_history(limit: int = 10, offset: int = 0):
    """レポート履歴取得"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        # モック履歴データ
        history = []
        for i in range(limit):
            history.append({
                "id": f"report_{offset + i + 1:03d}",
                "title": f"分析レポート #{offset + i + 1}",
                "template": "standard",
                "created_at": "2024-12-01T10:00:00",
                "status": "completed",
                "file_size": f"{np.random.randint(500, 5000)}KB",
                "format": "pdf"
            })

        return {
            "success": True,
            "history": history,
            "total": 100,  # モック総数
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"レポート履歴取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"履歴取得エラー: {str(e)}"
        ) from e


@router.delete("/history/{report_id}")
async def delete_report(report_id: str):
    """レポート削除"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        # モック削除処理
        success = True  # 実際の実装では削除処理を実行

        if success:
            return {
                "success": True,
                "message": f"レポート {report_id} が削除されました"
            }
        else:
            raise HTTPException(
                status_code=404, detail="指定されたレポートが見つかりません"
            )

    except Exception as e:
        logger.error(f"レポート削除エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"削除エラー: {str(e)}"
        ) from e


@router.post("/batch-export")
async def batch_export_reports(request: dict):
    """複数レポート一括エクスポート"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        report_ids = request.get("report_ids", [])
        export_format = request.get("format", "zip")
        
        if not report_ids:
            raise HTTPException(
                status_code=400, detail="エクスポートするレポートが指定されていません"
            )

        # モック一括エクスポート処理
        exported_files = []
        for report_id in report_ids:
            exported_files.append({
                "id": report_id,
                "filename": f"{report_id}.pdf",
                "size": f"{np.random.randint(500, 5000)}KB"
            })

        return {
            "success": True,
            "message": f"{len(report_ids)}件のレポートが正常にエクスポートされました",
            "archive_path": f"reports_export_{len(report_ids)}.zip",
            "exported_files": exported_files,
            "total_size": f"{sum(int(f['size'].replace('KB', '')) for f in exported_files)}KB"
        }

    except Exception as e:
        logger.error(f"一括エクスポートエラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"一括エクスポートエラー: {str(e)}"
        ) from e


@router.get("/scheduled")
async def get_scheduled_reports():
    """スケジュール済みレポート一覧取得"""
    report_manager = get_report_manager()

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        # モックスケジュールデータ
        scheduled_reports = [
            {
                "id": "schedule_001",
                "name": "日次市場分析レポート",
                "template": "daily_market",
                "frequency": "daily",
                "next_run": "2024-12-02T09:00:00",
                "recipients": ["admin@example.com"],
                "status": "active"
            },
            {
                "id": "schedule_002", 
                "name": "週次パフォーマンスレポート",
                "template": "weekly_performance",
                "frequency": "weekly",
                "next_run": "2024-12-08T18:00:00",
                "recipients": ["team@example.com"],
                "status": "active"
            }
        ]

        return {
            "success": True,
            "scheduled_reports": scheduled_reports,
            "count": len(scheduled_reports)
        }

    except Exception as e:
        logger.error(f"スケジュール済みレポート取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"スケジュール取得エラー: {str(e)}"
        ) from e