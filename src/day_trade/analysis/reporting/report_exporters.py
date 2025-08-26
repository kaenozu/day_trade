"""
レポート管理システム - レポートエクスポートモジュール

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import DetailedMarketReport, ReportFormat
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ReportExporters:
    """レポートエクスポート機能を提供するクラス"""

    def __init__(self, export_directory: Path):
        self.export_directory = export_directory

    def export_report(
        self,
        report: DetailedMarketReport,
        format: ReportFormat = ReportFormat.JSON,
        filename: Optional[str] = None,
    ) -> Path:
        """レポートエクスポート"""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_analysis_report_{timestamp}.{format.value}"

        filepath = self.export_directory / filename

        try:
            if format == ReportFormat.JSON:
                self._export_as_json(report, filepath)
            elif format == ReportFormat.MARKDOWN:
                self._export_as_markdown(report, filepath)
            elif format == ReportFormat.HTML:
                self._export_as_html(report, filepath)
            elif format == ReportFormat.CSV:
                self._export_as_csv(report, filepath)

            logger.info(f"レポートエクスポート完了: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"レポートエクスポートエラー: {e}")
            raise

    def _export_as_json(self, report: DetailedMarketReport, filepath: Path) -> None:
        """JSON形式でエクスポート"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                asdict(report), f, ensure_ascii=False, indent=2, default=str
            )

    def _export_as_markdown(self, report: DetailedMarketReport, filepath: Path) -> None:
        """Markdown形式でエクスポート"""
        md_content = [
            "# 市場分析レポート",
            f"**レポートID**: {report.report_id}",
            f"**生成日時**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**レポートタイプ**: {report.report_type.value}",
            f"**分析銘柄数**: {len(report.symbols_analyzed)}",
            "",
            "## 📊 分析専用システム",
            "**このシステムは分析・監視専用です**",
            "- 市場データ分析機能",
            "- ML予測システム",
            "- リスク評価機能",
            "",
            "## 📊 市場概要",
        ]

        # 市場サマリー追加
        if report.market_summary:
            for key, value in report.market_summary.items():
                if key != "注意":
                    md_content.append(f"- **{key}**: {value}")

        # トレンド分析追加
        md_content.extend(
            [
                "",
                "## 📈 トレンド分析",
                f"**全体的センチメント**: {report.trend_analysis.get('overall_sentiment', 'N/A')}",
            ]
        )

        if "key_observations" in report.trend_analysis:
            md_content.append("\n### 主要観察事項")
            for obs in report.trend_analysis["key_observations"]:
                md_content.append(f"- {obs}")

        # 教育的洞察追加
        md_content.extend(["", "## 🎓 教育的洞察"])
        for insight in report.educational_insights:
            md_content.append(insight)

        # 推奨事項追加
        md_content.extend(["", "## 💡 推奨事項"])
        for rec in report.recommendations:
            md_content.append(rec)

        # ファイル書き込み
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

    def _export_as_html(self, report: DetailedMarketReport, filepath: Path) -> None:
        """HTML形式でエクスポート"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>市場分析レポート - {report.report_id}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .safe-mode {{
                    background: #28a745;
                    color: white;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .section {{
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .insight-item {{
                    background: #e3f2fd;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                }}
                .recommendation {{
                    background: #f3e5f5;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 市場分析レポート</h1>
                <p>レポートID: {report.report_id}</p>
                <p>生成日時: {report.generated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>

            <div class="safe-mode">
                <h3>🔒 セーフモード動作中</h3>
                <p>このシステムは分析・教育・研究専用です。自動取引は完全に無効化されています。</p>
            </div>

            <div class="section">
                <h2>📈 トレンド分析</h2>
                <p><strong>全体的センチメント:</strong> {report.trend_analysis.get("overall_sentiment", "N/A")}</p>
            </div>

            <div class="section">
                <h2>🎓 教育的洞察</h2>
        """

        for insight in report.educational_insights:
            html_content += f'<div class="insight-item">{insight}</div>'

        html_content += """
            </div>

            <div class="section">
                <h2>💡 推奨事項</h2>
        """

        for rec in report.recommendations:
            html_content += f'<div class="recommendation">{rec}</div>'

        html_content += f"""
            </div>

            <div class="section">
                <h2>ℹ️ メタデータ</h2>
                <p><strong>分析銘柄数:</strong> {len(report.symbols_analyzed)}</p>
                <p><strong>セーフモード:</strong> 有効</p>
                <p><strong>免責事項:</strong> {report.metadata.get("disclaimer", "N/A")}</p>
            </div>
        </body>
        </html>
        """

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _export_as_csv(self, report: DetailedMarketReport, filepath: Path) -> None:
        """CSV形式でエクスポート（簡易版）"""
        try:
            # 個別銘柄データをCSV形式で出力
            csv_data = []

            for symbol, analysis in report.individual_analyses.items():
                row = {
                    "レポートID": report.report_id,
                    "生成日時": report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "銘柄": symbol,
                    "現在価格": analysis["basic_info"]["current_price"],
                    "価格トレンド": analysis["trend_analysis"]["price_trend"],
                    "出来高トレンド": analysis["trend_analysis"]["volume_trend"],
                    "ボラティリティ": analysis["trend_analysis"]["volatility"],
                    "リスクレベル": analysis["risk_assessment"]["risk_level"],
                    "リスクスコア": analysis["risk_assessment"]["score"],
                }

                if "signal_analysis" in analysis and analysis["signal_analysis"]:
                    row.update(
                        {
                            "シグナルタイプ": analysis["signal_analysis"].get(
                                "signal_type", "N/A"
                            ),
                            "シグナル信頼度": analysis["signal_analysis"].get(
                                "confidence", "N/A"
                            ),
                        }
                    )

                csv_data.append(row)

            # pandas使用してCSV出力
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

        except Exception as e:
            logger.error(f"CSV出力エラー: {e}")
            # フォールバック: 基本的なCSV出力
            with open(filepath, "w", encoding="utf-8-sig") as f:
                f.write("レポートID,生成日時,銘柄数,セーフモード\n")
                f.write(
                    f"{report.report_id},{report.generated_at},{len(report.symbols_analyzed)},有効\n"
                )