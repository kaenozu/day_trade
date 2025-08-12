"""
エクスポート管理

画像・PDF・データ出力を統合管理
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未インストール")

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly未インストール")

warnings.filterwarnings("ignore", category=UserWarning)


class ExportManager:
    """
    統合エクスポート管理

    複数形式での出力・レポート生成を統一管理
    """

    def __init__(self, base_output_dir: str = "output"):
        """
        初期化

        Args:
            base_output_dir: 基本出力ディレクトリ
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        self.charts_dir = self.base_output_dir / "charts"
        self.reports_dir = self.base_output_dir / "reports"
        self.data_dir = self.base_output_dir / "data"

        for dir_path in [self.charts_dir, self.reports_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"エクスポートマネージャー初期化完了 - 出力先: {self.base_output_dir}")

    def save_chart(self, fig, filename: str, format: str = "png", **kwargs) -> str:
        """
        チャート保存

        Args:
            fig: 図オブジェクト (matplotlib or plotly)
            filename: ファイル名
            format: 保存形式
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        # ファイル名に拡張子がない場合は追加
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        filepath = self.charts_dir / filename

        # matplotlib図の場合
        if MATPLOTLIB_AVAILABLE and hasattr(fig, "savefig"):
            fig.savefig(
                filepath,
                dpi=kwargs.get("dpi", 300),
                bbox_inches="tight",
                facecolor="white",
                **kwargs,
            )
            plt.close(fig)

        # plotly図の場合
        elif PLOTLY_AVAILABLE and isinstance(fig, go.Figure):
            if format.lower() == "html":
                fig.write_html(str(filepath))
            elif format.lower() in ["png", "jpg", "pdf", "svg"]:
                fig.write_image(str(filepath), format=format, **kwargs)
            else:
                logger.error(f"未対応の形式: {format}")
                return ""

        else:
            logger.error("未対応の図オブジェクト")
            return ""

        logger.info(f"チャート保存完了: {filepath}")
        return str(filepath)

    def save_data(self, data: pd.DataFrame, filename: str, format: str = "csv", **kwargs) -> str:
        """
        データ保存

        Args:
            data: データフレーム
            filename: ファイル名
            format: 保存形式 ("csv", "excel", "json", "parquet")
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"

        filepath = self.data_dir / filename

        try:
            if format.lower() == "csv":
                data.to_csv(filepath, **kwargs)
            elif format.lower() == "excel":
                data.to_excel(filepath, **kwargs)
            elif format.lower() == "json":
                data.to_json(filepath, **kwargs)
            elif format.lower() == "parquet":
                data.to_parquet(filepath, **kwargs)
            else:
                logger.error(f"未対応のデータ形式: {format}")
                return ""

            logger.info(f"データ保存完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"データ保存エラー: {e}")
            return ""

    def save_analysis_report(self, analysis_results: Dict[str, Any], filename: str = None) -> str:
        """
        分析レポート保存

        Args:
            analysis_results: 分析結果辞書
            filename: ファイル名（指定しない場合は自動生成）

        Returns:
            保存されたファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}.json"

        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        filepath = self.reports_dir / filename

        # レポートにメタデータ追加
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_type": "analysis_report",
                "version": "1.0",
            },
            "analysis_results": analysis_results,
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"分析レポート保存完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"レポート保存エラー: {e}")
            return ""

    def create_pdf_report(self, figures: List, filename: str = None, **kwargs) -> str:
        """
        PDF複合レポート作成

        Args:
            figures: 図のリスト
            filename: ファイル名
            **kwargs: 追加パラメータ

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib未インストール - PDF作成不可")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{timestamp}.pdf"

        if not filename.endswith(".pdf"):
            filename = f"{filename}.pdf"

        filepath = self.reports_dir / filename

        try:
            with PdfPages(filepath) as pdf:
                for fig in figures:
                    if hasattr(fig, "savefig"):
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)

            logger.info(f"PDFレポート保存完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"PDF作成エラー: {e}")
            return ""

    def save_interactive_dashboard(self, fig: go.Figure, filename: str = None) -> str:
        """
        インタラクティブダッシュボード保存

        Args:
            fig: Plotly図オブジェクト
            filename: ファイル名

        Returns:
            保存されたファイルパス
        """
        if not PLOTLY_AVAILABLE:
            logger.error("plotly未インストール - インタラクティブダッシュボード保存不可")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interactive_dashboard_{timestamp}.html"

        if not filename.endswith(".html"):
            filename = f"{filename}.html"

        filepath = self.reports_dir / filename

        try:
            # カスタムテンプレート付きで保存
            config = {
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "chart",
                    "height": 800,
                    "width": 1200,
                    "scale": 1,
                },
            }

            fig.write_html(
                str(filepath),
                config=config,
                include_plotlyjs=True,
                div_id="analysis-dashboard",
            )

            logger.info(f"インタラクティブダッシュボード保存完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"ダッシュボード保存エラー: {e}")
            return ""

    def export_summary(self, exported_files: List[str], filename: str = None) -> str:
        """
        エクスポートサマリー作成

        Args:
            exported_files: エクスポートされたファイルリスト
            filename: サマリーファイル名

        Returns:
            保存されたサマリーファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_summary_{timestamp}.json"

        if not filename.endswith(".json"):
            filename = f"{filename}.json"

        filepath = self.reports_dir / filename

        summary_data = {
            "export_summary": {
                "generated_at": datetime.now().isoformat(),
                "total_files": len(exported_files),
                "exported_files": exported_files,
                "output_directories": {
                    "charts": str(self.charts_dir),
                    "reports": str(self.reports_dir),
                    "data": str(self.data_dir),
                },
            }
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)

            logger.info(f"エクスポートサマリー保存完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"サマリー保存エラー: {e}")
            return ""

    def cleanup_old_exports(self, days_to_keep: int = 7) -> None:
        """
        古いエクスポートファイルのクリーンアップ

        Args:
            days_to_keep: 保持日数
        """
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleanup_count = 0

        for directory in [self.charts_dir, self.reports_dir, self.data_dir]:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    file_modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_modified_time < cutoff_date:
                        try:
                            file_path.unlink()
                            cleanup_count += 1
                        except Exception as e:
                            logger.warning(f"ファイル削除失敗: {file_path} - {e}")

        if cleanup_count > 0:
            logger.info(f"古いエクスポートファイル {cleanup_count} 個を削除しました")

    def get_export_statistics(self) -> Dict[str, Any]:
        """
        エクスポート統計情報取得

        Returns:
            統計情報辞書
        """
        stats = {
            "directories": {
                "base": str(self.base_output_dir),
                "charts": str(self.charts_dir),
                "reports": str(self.reports_dir),
                "data": str(self.data_dir),
            },
            "file_counts": {},
            "disk_usage": {},
            "recent_exports": [],
        }

        for name, directory in stats["directories"].items():
            if name == "base":
                continue

            dir_path = Path(directory)
            if dir_path.exists():
                files = list(dir_path.rglob("*"))
                file_count = len([f for f in files if f.is_file()])

                # ディスク使用量計算
                total_size = sum(f.stat().st_size for f in files if f.is_file())

                stats["file_counts"][name] = file_count
                stats["disk_usage"][name] = f"{total_size / 1024 / 1024:.2f} MB"

        return stats
