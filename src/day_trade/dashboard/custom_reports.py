#!/usr/bin/env python3
"""
カスタマイズレポート生成システム
Issue #319: 分析ダッシュボード強化

ユーザー定義指標によるレポート生成・エクスポート機能
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from matplotlib import rcParams

    # 日本語フォント設定
    rcParams["font.family"] = ["DejaVu Sans", "Arial"]

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class CustomReportManager:
    """カスタマイズレポート管理システム"""

    def __init__(self, output_dir: str = "reports"):
        """
        初期化

        Args:
            output_dir: レポート出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # テンプレート管理
        self.templates = self._load_report_templates()

        # メトリクス定義
        self.available_metrics = {
            "基本指標": [
                {"key": "total_return", "name": "累積リターン", "format": ".2%"},
                {"key": "volatility", "name": "ボラティリティ", "format": ".2%"},
                {"key": "sharpe_ratio", "name": "シャープレシオ", "format": ".3f"},
                {"key": "max_drawdown", "name": "最大ドローダウン", "format": ".2%"},
                {"key": "win_rate", "name": "勝率", "format": ".1%"},
            ],
            "リスク指標": [
                {"key": "var_95", "name": "VaR (95%)", "format": ".2%"},
                {"key": "cvar_95", "name": "CVaR (95%)", "format": ".2%"},
                {"key": "sortino_ratio", "name": "ソルティノレシオ", "format": ".3f"},
                {"key": "calmar_ratio", "name": "カルマーレシオ", "format": ".3f"},
                {"key": "beta", "name": "ベータ", "format": ".3f"},
            ],
            "パフォーマンス指標": [
                {"key": "alpha", "name": "アルファ", "format": ".2%"},
                {"key": "information_ratio", "name": "情報レシオ", "format": ".3f"},
                {
                    "key": "tracking_error",
                    "name": "トラッキングエラー",
                    "format": ".2%",
                },
                {
                    "key": "upside_capture",
                    "name": "アップサイドキャプチャ",
                    "format": ".2%",
                },
                {
                    "key": "downside_capture",
                    "name": "ダウンサイドキャプチャ",
                    "format": ".2%",
                },
            ],
        }

        logger.info("カスタマイズレポートマネージャー初期化完了")

    def _load_report_templates(self) -> Dict[str, Dict]:
        """レポートテンプレート読み込み"""
        return {
            "standard": {
                "name": "標準レポート",
                "description": "基本的なパフォーマンス指標を含むレポート",
                "sections": [
                    {"type": "summary", "title": "エグゼクティブサマリー"},
                    {"type": "performance", "title": "パフォーマンス分析"},
                    {"type": "risk", "title": "リスク分析"},
                    {"type": "charts", "title": "チャート分析"},
                ],
            },
            "detailed": {
                "name": "詳細レポート",
                "description": "包括的な分析結果を含む詳細レポート",
                "sections": [
                    {"type": "summary", "title": "エグゼクティブサマリー"},
                    {"type": "performance", "title": "パフォーマンス分析"},
                    {"type": "risk", "title": "リスク分析"},
                    {"type": "attribution", "title": "パフォーマンス寄与度分析"},
                    {"type": "sector", "title": "セクター分析"},
                    {"type": "charts", "title": "チャート分析"},
                    {"type": "appendix", "title": "補足資料"},
                ],
            },
            "risk_focused": {
                "name": "リスク重点レポート",
                "description": "リスク管理に焦点を当てたレポート",
                "sections": [
                    {"type": "risk_summary", "title": "リスクサマリー"},
                    {"type": "var_analysis", "title": "VaR分析"},
                    {"type": "stress_test", "title": "ストレステスト"},
                    {"type": "correlation", "title": "相関分析"},
                    {"type": "scenario", "title": "シナリオ分析"},
                ],
            },
        }

    def create_custom_report(
        self,
        data: Dict[str, Any],
        template: str = "standard",
        custom_metrics: List[str] = None,
        title: str = None,
        period: Tuple[datetime, datetime] = None,
    ) -> Dict[str, Any]:
        """
        カスタムレポート作成

        Args:
            data: 分析データ
            template: テンプレート名
            custom_metrics: カスタム指標リスト
            title: レポートタイトル
            period: 分析期間

        Returns:
            レポート生成結果
        """
        try:
            # テンプレート取得
            if template not in self.templates:
                template = "standard"

            template_config = self.templates[template]

            # レポート基本情報
            report_info = {
                "title": title
                or f"{template_config['name']} - {datetime.now().strftime('%Y年%m月%d日')}",
                "template": template,
                "generated_at": datetime.now().isoformat(),
                "period": {
                    "start": period[0].isoformat() if period else None,
                    "end": period[1].isoformat() if period else None,
                },
                "metrics_used": custom_metrics or [],
            }

            # セクション生成
            sections = []
            for section_config in template_config["sections"]:
                section_data = self._generate_section(
                    section_config, data, custom_metrics
                )
                if section_data:
                    sections.append(section_data)

            report_content = {"info": report_info, "sections": sections}

            logger.info(f"カスタムレポート作成完了: {template}")
            return {
                "success": True,
                "report": report_content,
                "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            }

        except Exception as e:
            logger.error(f"カスタムレポート作成エラー: {e}")
            return {"success": False, "error": str(e), "report": None}

    def _generate_section(
        self,
        section_config: Dict[str, str],
        data: Dict[str, Any],
        custom_metrics: List[str],
    ) -> Dict[str, Any]:
        """レポートセクション生成"""
        section_type = section_config["type"]
        section_title = section_config["title"]

        try:
            if section_type == "summary":
                return self._generate_summary_section(
                    data, custom_metrics, section_title
                )
            elif section_type == "performance":
                return self._generate_performance_section(
                    data, custom_metrics, section_title
                )
            elif section_type == "risk":
                return self._generate_risk_section(data, custom_metrics, section_title)
            elif section_type == "charts":
                return self._generate_charts_section(data, section_title)
            elif section_type == "attribution":
                return self._generate_attribution_section(data, section_title)
            else:
                return {
                    "type": section_type,
                    "title": section_title,
                    "content": f"{section_title}の内容（実装予定）",
                }

        except Exception as e:
            logger.warning(f"セクション生成エラー ({section_type}): {e}")
            return None

    def _generate_summary_section(
        self, data: Dict[str, Any], custom_metrics: List[str], title: str
    ) -> Dict[str, Any]:
        """サマリーセクション生成"""
        # モックサマリーデータ
        summary_metrics = {
            "期間": "2024年1月1日 - 2024年12月31日",
            "累積リターン": "+12.3%",
            "年率リターン": "+12.3%",
            "ボラティリティ": "18.5%",
            "シャープレシオ": "0.67",
            "最大ドローダウン": "-8.1%",
            "勝率": "58.3%",
        }

        key_findings = [
            "ポートフォリオは市場を2.1%上回るパフォーマンスを達成",
            "リスク調整後リターン（シャープレシオ）は0.67と良好",
            "最大ドローダウンは-8.1%に制限され、リスク管理が機能",
            "テクノロジーセクターの好調が主要な収益源",
        ]

        return {
            "type": "summary",
            "title": title,
            "content": {
                "metrics": summary_metrics,
                "key_findings": key_findings,
                "risk_level": "中リスク",
                "recommendation": "現在のポートフォリオ構成を維持し、定期的なリバランシングを実施",
            },
        }

    def _generate_performance_section(
        self, data: Dict[str, Any], custom_metrics: List[str], title: str
    ) -> Dict[str, Any]:
        """パフォーマンスセクション生成"""
        # 月次パフォーマンスデータ（モック）
        monthly_returns = {
            month: np.random.normal(0.01, 0.03)
            for month in pd.date_range("2024-01", "2024-12", freq="M").strftime("%Y-%m")
        }

        # 指標計算
        returns_array = np.array(list(monthly_returns.values()))
        performance_metrics = {
            "月次リターン平均": f"{np.mean(returns_array):.2%}",
            "月次リターン標準偏差": f"{np.std(returns_array):.2%}",
            "ベストパフォーマンス月": f"{max(monthly_returns, key=monthly_returns.get)}: {max(monthly_returns.values()):.2%}",
            "ワーストパフォーマンス月": f"{min(monthly_returns, key=monthly_returns.get)}: {min(monthly_returns.values()):.2%}",
            "正の月数": f"{sum(1 for r in returns_array if r > 0)}ヶ月",
        }

        return {
            "type": "performance",
            "title": title,
            "content": {
                "monthly_returns": monthly_returns,
                "metrics": performance_metrics,
                "benchmark_comparison": {
                    "vs_topix": "+2.1%",
                    "vs_nikkei": "+1.8%",
                    "tracking_error": "4.2%",
                },
            },
        }

    def _generate_risk_section(
        self, data: Dict[str, Any], custom_metrics: List[str], title: str
    ) -> Dict[str, Any]:
        """リスクセクション生成"""
        risk_metrics = {
            "VaR (95%, 日次)": "-2.1%",
            "CVaR (95%, 日次)": "-3.2%",
            "ソルティノレシオ": "0.89",
            "カルマーレシオ": "1.52",
            "ベータ (vs TOPIX)": "0.91",
            "相関 (vs TOPIX)": "0.78",
            "アップサイドキャプチャ": "95.2%",
            "ダウンサイドキャプチャ": "87.1%",
        }

        risk_assessment = {
            "総合リスクレベル": "中リスク",
            "市場リスク": "中",
            "集中リスク": "低",
            "流動性リスク": "低",
            "主要リスク要因": [
                "テクノロジーセクターへの集中",
                "大型株への偏重",
                "為替リスク（限定的）",
            ],
        }

        return {
            "type": "risk",
            "title": title,
            "content": {
                "metrics": risk_metrics,
                "assessment": risk_assessment,
                "recommendations": [
                    "セクター分散の改善",
                    "中小型株への配分検討",
                    "定期的なストレステスト実施",
                ],
            },
        }

    def _generate_charts_section(
        self, data: Dict[str, Any], title: str
    ) -> Dict[str, Any]:
        """チャートセクション生成"""
        charts_info = {
            "cumulative_return": {
                "title": "累積リターン推移",
                "description": "ポートフォリオとベンチマークの累積リターン比較",
                "type": "line_chart",
            },
            "drawdown": {
                "title": "ドローダウン分析",
                "description": "期間中のドローダウン推移",
                "type": "area_chart",
            },
            "sector_allocation": {
                "title": "セクター配分",
                "description": "現在のセクター別配分状況",
                "type": "pie_chart",
            },
            "risk_return": {
                "title": "リスク・リターン散布図",
                "description": "保有銘柄のリスク・リターン分析",
                "type": "scatter_plot",
            },
        }

        return {
            "type": "charts",
            "title": title,
            "content": {
                "charts": charts_info,
                "note": "チャートは別途生成され、レポートに挿入されます",
            },
        }

    def _generate_attribution_section(
        self, data: Dict[str, Any], title: str
    ) -> Dict[str, Any]:
        """パフォーマンス寄与度セクション生成"""
        # 寄与度分析（モック）
        attribution_data = {
            "セクター寄与度": {
                "テクノロジー": "+3.2%",
                "金融": "+1.1%",
                "ヘルスケア": "+0.8%",
                "消費財": "-0.3%",
                "エネルギー": "-0.2%",
            },
            "銘柄寄与度（上位5位）": {
                "7203.T (トヨタ)": "+1.8%",
                "6758.T (ソニー)": "+1.5%",
                "9984.T (SBG)": "+1.2%",
                "8306.T (MUFG)": "+0.9%",
                "6861.T (キーエンス)": "+0.7%",
            },
            "アクティブリターン": {
                "選択効果": "+1.2%",
                "配分効果": "+0.9%",
                "交互作用": "+0.1%",
                "合計": "+2.2%",
            },
        }

        return {"type": "attribution", "title": title, "content": attribution_data}

    def export_to_json(self, report_data: Dict[str, Any], filename: str = None) -> str:
        """JSONエクスポート"""
        try:
            if not filename:
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            logger.info(f"JSONレポートエクスポート完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"JSONエクスポートエラー: {e}")
            raise

    def export_to_excel(self, report_data: Dict[str, Any], filename: str = None) -> str:
        """Excelエクスポート"""
        try:
            if not filename:
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            filepath = self.output_dir / filename

            # ExcelWriterを使用してマルチシート作成
            with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                # サマリーシート
                summary_data = self._extract_summary_for_excel(report_data)
                summary_df = pd.DataFrame.from_dict(
                    summary_data, orient="index", columns=["値"]
                )
                summary_df.to_excel(writer, sheet_name="サマリー")

                # パフォーマンスシート
                perf_data = self._extract_performance_for_excel(report_data)
                if perf_data:
                    perf_df = pd.DataFrame(perf_data)
                    perf_df.to_excel(writer, sheet_name="パフォーマンス", index=False)

                # リスクシート
                risk_data = self._extract_risk_for_excel(report_data)
                if risk_data:
                    risk_df = pd.DataFrame.from_dict(
                        risk_data, orient="index", columns=["値"]
                    )
                    risk_df.to_excel(writer, sheet_name="リスク分析")

            logger.info(f"Excelレポートエクスポート完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Excelエクスポートエラー: {e}")
            raise

    def export_to_pdf(self, report_data: Dict[str, Any], filename: str = None) -> str:
        """PDFエクスポート"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("PDFエクスポートにはreportlabライブラリが必要です")

        try:
            if not filename:
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            filepath = self.output_dir / filename

            # PDF文書作成
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()

            # タイトル
            title = report_data.get("info", {}).get("title", "カスタムレポート")
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=16,
                spaceAfter=20,
                alignment=1,  # 中央揃え
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))

            # 各セクション追加
            for section in report_data.get("sections", []):
                self._add_section_to_pdf(story, section, styles)

            # PDF生成
            doc.build(story)

            logger.info(f"PDFレポートエクスポート完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"PDFエクスポートエラー: {e}")
            raise

    def _extract_summary_for_excel(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Excel用サマリーデータ抽出"""
        for section in report_data.get("sections", []):
            if section.get("type") == "summary":
                return section.get("content", {}).get("metrics", {})
        return {}

    def _extract_performance_for_excel(
        self, report_data: Dict[str, Any]
    ) -> Optional[Dict]:
        """Excel用パフォーマンスデータ抽出"""
        for section in report_data.get("sections", []):
            if section.get("type") == "performance":
                monthly_returns = section.get("content", {}).get("monthly_returns", {})
                if monthly_returns:
                    return {
                        "月": list(monthly_returns.keys()),
                        "リターン": [f"{v:.2%}" for v in monthly_returns.values()],
                    }
        return None

    def _extract_risk_for_excel(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Excel用リスクデータ抽出"""
        for section in report_data.get("sections", []):
            if section.get("type") == "risk":
                return section.get("content", {}).get("metrics", {})
        return {}

    def _add_section_to_pdf(self, story: List, section: Dict[str, Any], styles) -> None:
        """PDFセクション追加"""
        # セクションタイトル
        story.append(Paragraph(section.get("title", ""), styles["Heading2"]))
        story.append(Spacer(1, 12))

        # セクション内容
        content = section.get("content", {})

        if section.get("type") == "summary":
            # サマリーテーブル
            metrics = content.get("metrics", {})
            if metrics:
                table_data = [["項目", "値"]]
                table_data.extend([[k, v] for k, v in metrics.items()])

                table = Table(table_data)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                story.append(table)

        story.append(Spacer(1, 20))

    def get_available_templates(self) -> Dict[str, Dict]:
        """利用可能テンプレート一覧取得"""
        return self.templates

    def get_available_metrics(self) -> Dict[str, List]:
        """利用可能指標一覧取得"""
        return self.available_metrics

    def schedule_periodic_report(
        self,
        template: str,
        frequency: str,  # 'daily', 'weekly', 'monthly'
        recipients: List[str] = None,
    ) -> Dict[str, Any]:
        """定期レポートスケジュール設定"""
        # 定期レポート設定（実装簡略版）
        schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        schedule_config = {
            "id": schedule_id,
            "template": template,
            "frequency": frequency,
            "recipients": recipients or [],
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        # スケジュール保存（実装では永続化）
        schedule_file = self.output_dir / "schedules.json"
        schedules = {}

        if schedule_file.exists():
            with open(schedule_file, encoding="utf-8") as f:
                schedules = json.load(f)

        schedules[schedule_id] = schedule_config

        with open(schedule_file, "w", encoding="utf-8") as f:
            json.dump(schedules, f, ensure_ascii=False, indent=2)

        logger.info(f"定期レポートスケジュール設定完了: {schedule_id}")

        return {"success": True, "schedule_id": schedule_id, "config": schedule_config}


if __name__ == "__main__":
    # テスト実行
    print("カスタマイズレポートマネージャー テスト")
    print("=" * 60)

    try:
        report_manager = CustomReportManager()

        # サンプルデータ
        sample_data = {
            "portfolio_returns": np.random.normal(0.001, 0.02, 252).tolist(),
            "benchmark_returns": np.random.normal(0.0008, 0.018, 252).tolist(),
            "symbols": ["7203.T", "8306.T", "9984.T"],
            "sector_weights": {"tech": 0.4, "finance": 0.3, "healthcare": 0.3},
        }

        # 標準レポート作成
        print("\n1. 標準レポート作成テスト")
        standard_report = report_manager.create_custom_report(
            data=sample_data, template="standard", title="テスト用標準レポート"
        )

        if standard_report["success"]:
            print(f"✅ レポート作成成功: {standard_report['report_id']}")
            print(f"   セクション数: {len(standard_report['report']['sections'])}")

        # JSONエクスポートテスト
        print("\n2. JSONエクスポートテスト")
        if standard_report["success"]:
            json_path = report_manager.export_to_json(
                standard_report["report"], "test_report.json"
            )
            print(f"✅ JSONエクスポート完了: {json_path}")

        # Excelエクスポートテスト
        print("\n3. Excelエクスポートテスト")
        try:
            if standard_report["success"]:
                excel_path = report_manager.export_to_excel(
                    standard_report["report"], "test_report.xlsx"
                )
                print(f"✅ Excelエクスポート完了: {excel_path}")
        except ImportError:
            print("⚠️  Excelエクスポートにはopenpyxlが必要です")

        # 利用可能テンプレート確認
        print("\n4. 利用可能テンプレート")
        templates = report_manager.get_available_templates()
        for template_id, template_info in templates.items():
            print(f"   - {template_id}: {template_info['name']}")

        # 利用可能指標確認
        print("\n5. 利用可能指標")
        metrics = report_manager.get_available_metrics()
        for category, metric_list in metrics.items():
            print(f"   {category}: {len(metric_list)}個の指標")

        print("\n✅ カスタマイズレポートマネージャー テスト完了！")

    except Exception as e:
        print(f"❌ テストエラー: {e}")
        import traceback

        traceback.print_exc()
