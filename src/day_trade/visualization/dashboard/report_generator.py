"""
レポート自動生成

統合レポート・PDF生成・サマリー作成の自動化機能
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

from ...utils.logging_config import get_context_logger
from ..base.export_manager import ExportManager

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
    from jinja2 import Template, Environment, FileSystemLoader

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logger.warning("jinja2未インストール - テンプレートレポート生成不可")


class ReportGenerator:
    """
    レポート自動生成クラス

    統合レポート・PDF生成・サマリー作成の自動化機能を提供
    """

    def __init__(self, output_dir: str = "output/reports", template_dir: str = "templates"):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
            template_dir: テンプレートディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.export_manager = ExportManager(str(self.output_dir.parent))

        # Jinja2環境設定
        if JINJA2_AVAILABLE:
            try:
                self.jinja_env = Environment(
                    loader=FileSystemLoader(str(self.template_dir))
                )
            except Exception as e:
                logger.warning(f"テンプレートディレクトリ設定失敗: {e}")
                self.jinja_env = None
        else:
            self.jinja_env = None

        logger.info("レポート自動生成システム初期化完了")

    def generate_comprehensive_report(
        self,
        data: pd.DataFrame,
        analysis_results: Dict,
        visualization_results: Dict,
        symbol: str = "STOCK",
        **kwargs,
    ) -> Dict[str, str]:
        """
        総合レポート生成

        Args:
            data: 価格データ
            analysis_results: 分析結果
            visualization_results: 可視化結果
            symbol: 銘柄シンボル
            **kwargs: 追加パラメータ

        Returns:
            生成ファイルパス辞書
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"{symbol}_{timestamp}"

        generated_files = {}

        try:
            # 1. サマリーレポート（JSON）
            summary_report = self._generate_summary_report(
                data, analysis_results, symbol, report_id
            )
            if summary_report:
                generated_files["summary_json"] = summary_report

            # 2. 詳細レポート（HTML）
            html_report = self._generate_html_report(
                data, analysis_results, visualization_results, symbol, report_id
            )
            if html_report:
                generated_files["detailed_html"] = html_report

            # 3. PDFレポート
            pdf_report = self._generate_pdf_report(
                data, analysis_results, visualization_results, symbol, report_id
            )
            if pdf_report:
                generated_files["comprehensive_pdf"] = pdf_report

            # 4. エグゼクティブサマリー
            executive_summary = self._generate_executive_summary(
                analysis_results, symbol, report_id
            )
            if executive_summary:
                generated_files["executive_summary"] = executive_summary

            # 5. データエクスポート
            data_exports = self._export_analysis_data(
                data, analysis_results, symbol, report_id
            )
            generated_files.update(data_exports)

            # 6. レポートインデックス生成
            index_file = self._generate_report_index(
                generated_files, symbol, report_id
            )
            if index_file:
                generated_files["report_index"] = index_file

            logger.info(f"総合レポート生成完了: {len(generated_files)}ファイル")
            return generated_files

        except Exception as e:
            logger.error(f"総合レポート生成エラー: {e}")
            return {}

    def _generate_summary_report(
        self, data: pd.DataFrame, analysis_results: Dict, symbol: str, report_id: str
    ) -> Optional[str]:
        """
        サマリーレポート生成

        Args:
            data: 価格データ
            analysis_results: 分析結果
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            保存されたファイルパス
        """
        try:
            summary_data = {
                "report_metadata": {
                    "report_id": report_id,
                    "symbol": symbol,
                    "generated_at": datetime.now().isoformat(),
                    "data_period": {
                        "start": str(data.index[0]) if len(data) > 0 else None,
                        "end": str(data.index[-1]) if len(data) > 0 else None,
                        "total_days": len(data),
                    },
                },
                "price_summary": self._extract_price_summary(data),
                "prediction_summary": self._extract_prediction_summary(analysis_results),
                "technical_summary": self._extract_technical_summary(analysis_results),
                "risk_summary": self._extract_risk_summary(analysis_results),
                "model_performance": self._extract_model_performance(analysis_results),
                "key_insights": self._generate_key_insights(data, analysis_results),
            }

            filename = f"summary_report_{report_id}.json"
            return self.export_manager.save_analysis_report(summary_data, filename)

        except Exception as e:
            logger.error(f"サマリーレポート生成エラー: {e}")
            return None

    def _generate_html_report(
        self,
        data: pd.DataFrame,
        analysis_results: Dict,
        visualization_results: Dict,
        symbol: str,
        report_id: str,
    ) -> Optional[str]:
        """
        HTMLレポート生成

        Args:
            data: 価格データ
            analysis_results: 分析結果
            visualization_results: 可視化結果
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            保存されたファイルパス
        """
        if not self.jinja_env:
            # テンプレートなしでシンプルなHTMLを生成
            return self._generate_simple_html_report(
                data, analysis_results, symbol, report_id
            )

        try:
            template = self.jinja_env.get_template("comprehensive_report.html")

            context = {
                "symbol": symbol,
                "report_id": report_id,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "price_summary": self._extract_price_summary(data),
                "prediction_summary": self._extract_prediction_summary(analysis_results),
                "technical_summary": self._extract_technical_summary(analysis_results),
                "risk_summary": self._extract_risk_summary(analysis_results),
                "visualization_results": visualization_results,
                "key_insights": self._generate_key_insights(data, analysis_results),
            }

            html_content = template.render(context)

            filename = f"detailed_report_{report_id}.html"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTMLレポート生成完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"HTMLレポート生成エラー: {e}")
            return self._generate_simple_html_report(
                data, analysis_results, symbol, report_id
            )

    def _generate_simple_html_report(
        self, data: pd.DataFrame, analysis_results: Dict, symbol: str, report_id: str
    ) -> Optional[str]:
        """
        シンプルHTMLレポート生成（テンプレートなし）

        Args:
            data: 価格データ
            analysis_results: 分析結果
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            保存されたファイルパス
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{symbol} 分析レポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; }}
        .insight {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-left: 4px solid #007acc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{symbol} 総合分析レポート</h1>
        <p><strong>レポートID:</strong> {report_id}</p>
        <p><strong>生成日時:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>価格サマリー</h2>
        {self._format_price_summary_html(data)}
    </div>

    <div class="section">
        <h2>予測サマリー</h2>
        {self._format_prediction_summary_html(analysis_results)}
    </div>

    <div class="section">
        <h2>リスク分析</h2>
        {self._format_risk_summary_html(analysis_results)}
    </div>

    <div class="section">
        <h2>主要インサイト</h2>
        {self._format_insights_html(data, analysis_results)}
    </div>
</body>
</html>
"""

            filename = f"detailed_report_{report_id}.html"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"シンプルHTMLレポート生成完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"シンプルHTMLレポート生成エラー: {e}")
            return None

    def _generate_pdf_report(
        self,
        data: pd.DataFrame,
        analysis_results: Dict,
        visualization_results: Dict,
        symbol: str,
        report_id: str,
    ) -> Optional[str]:
        """
        PDFレポート生成

        Args:
            data: 価格データ
            analysis_results: 分析結果
            visualization_results: 可視化結果
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            保存されたファイルパス
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未インストール - PDF生成不可")
            return None

        try:
            filename = f"comprehensive_report_{report_id}.pdf"
            filepath = self.output_dir / filename

            # 可視化図の収集
            figures = []

            # サマリーページ作成
            summary_fig = self._create_summary_page(data, analysis_results, symbol)
            if summary_fig:
                figures.append(summary_fig)

            # 可視化結果から図を収集
            for category, files in visualization_results.items():
                if isinstance(files, dict):
                    for chart_type, chart_path in files.items():
                        if isinstance(chart_path, str) and chart_path.endswith(
                            (".png", ".jpg", ".jpeg")
                        ):
                            # 画像ファイルをmatplotlib図として読み込み
                            fig = self._load_image_as_figure(chart_path)
                            if fig:
                                figures.append(fig)

            # PDF作成
            if figures:
                return self.export_manager.create_pdf_report(figures, filename)
            else:
                logger.warning("PDF生成用の図が見つかりませんでした")
                return None

        except Exception as e:
            logger.error(f"PDFレポート生成エラー: {e}")
            return None

    def _generate_executive_summary(
        self, analysis_results: Dict, symbol: str, report_id: str
    ) -> Optional[str]:
        """
        エグゼクティブサマリー生成

        Args:
            analysis_results: 分析結果
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            保存されたファイルパス
        """
        try:
            # 主要指標の抽出
            key_metrics = self._extract_key_metrics(analysis_results)

            # 投資判断の生成
            investment_recommendation = self._generate_investment_recommendation(
                analysis_results
            )

            # リスク評価
            risk_assessment = self._generate_risk_assessment(analysis_results)

            executive_summary = {
                "executive_summary": {
                    "symbol": symbol,
                    "report_id": report_id,
                    "generated_at": datetime.now().isoformat(),
                    "investment_recommendation": investment_recommendation,
                    "key_metrics": key_metrics,
                    "risk_assessment": risk_assessment,
                    "summary_points": self._generate_summary_points(analysis_results),
                    "next_actions": self._generate_next_actions(analysis_results),
                }
            }

            filename = f"executive_summary_{report_id}.json"
            return self.export_manager.save_analysis_report(executive_summary, filename)

        except Exception as e:
            logger.error(f"エグゼクティブサマリー生成エラー: {e}")
            return None

    def _export_analysis_data(
        self, data: pd.DataFrame, analysis_results: Dict, symbol: str, report_id: str
    ) -> Dict[str, str]:
        """
        分析データエクスポート

        Args:
            data: 価格データ
            analysis_results: 分析結果
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            エクスポートファイルパス辞書
        """
        exported_files = {}

        try:
            # 価格データ（CSV）
            price_file = f"price_data_{report_id}.csv"
            price_path = self.export_manager.save_data(data, price_file, format="csv")
            if price_path:
                exported_files["price_data_csv"] = price_path

            # 価格データ（Excel）
            excel_file = f"price_data_{report_id}.xlsx"
            excel_path = self.export_manager.save_data(data, excel_file, format="excel")
            if excel_path:
                exported_files["price_data_excel"] = excel_path

            # 予測結果データ
            predictions_data = self._compile_predictions_data(analysis_results)
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                pred_file = f"predictions_{report_id}.csv"
                pred_path = self.export_manager.save_data(
                    predictions_df, pred_file, format="csv"
                )
                if pred_path:
                    exported_files["predictions_csv"] = pred_path

            # 分析結果（JSON）
            analysis_file = f"analysis_results_{report_id}.json"
            analysis_path = self.export_manager.save_analysis_report(
                analysis_results, analysis_file
            )
            if analysis_path:
                exported_files["analysis_json"] = analysis_path

            return exported_files

        except Exception as e:
            logger.error(f"分析データエクスポートエラー: {e}")
            return {}

    def _generate_report_index(
        self, generated_files: Dict, symbol: str, report_id: str
    ) -> Optional[str]:
        """
        レポートインデックス生成

        Args:
            generated_files: 生成ファイル辞書
            symbol: 銘柄シンボル
            report_id: レポートID

        Returns:
            インデックスファイルパス
        """
        try:
            index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{symbol} 分析レポートインデックス</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .file-list {{ margin: 20px 0; }}
        .file-item {{
            display: block;
            padding: 10px;
            margin: 5px 0;
            background-color: #f8f9fa;
            text-decoration: none;
            color: #333;
            border-left: 4px solid #007acc;
            border-radius: 3px;
        }}
        .file-item:hover {{ background-color: #e9ecef; }}
        .category {{
            font-size: 18px;
            font-weight: bold;
            margin: 20px 0 10px 0;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{symbol} 分析レポートインデックス</h1>
        <p><strong>レポートID:</strong> {report_id}</p>
        <p><strong>生成日時:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>総ファイル数:</strong> {len(generated_files)}</p>
    </div>

    <div class="file-list">
        {self._format_file_list_html(generated_files)}
    </div>
</body>
</html>
"""

            filename = f"report_index_{report_id}.html"
            filepath = self.output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(index_html)

            logger.info(f"レポートインデックス生成完了: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"レポートインデックス生成エラー: {e}")
            return None

    # ヘルパーメソッド群
    def _extract_price_summary(self, data: pd.DataFrame) -> Dict:
        """価格サマリー抽出"""
        if "Close" in data.columns and len(data) > 0:
            close_prices = data["Close"]
            return {
                "current_price": float(close_prices.iloc[-1]),
                "period_return": float((close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100),
                "volatility": float(close_prices.pct_change().std() * 100),
                "max_price": float(close_prices.max()),
                "min_price": float(close_prices.min()),
                "avg_price": float(close_prices.mean()),
            }
        return {}

    def _extract_prediction_summary(self, analysis_results: Dict) -> Dict:
        """予測サマリー抽出"""
        summary = {}

        if "lstm_prediction" in analysis_results:
            lstm_data = analysis_results["lstm_prediction"]
            if "predictions" in lstm_data:
                predictions = lstm_data["predictions"]
                summary["lstm_next_price"] = float(predictions[-1]) if predictions else None
                summary["lstm_confidence"] = lstm_data.get("confidence", 0.5)

        if "ensemble_prediction" in analysis_results:
            ensemble_data = analysis_results["ensemble_prediction"]
            if "ensemble_prediction" in ensemble_data:
                predictions = ensemble_data["ensemble_prediction"]
                summary["ensemble_next_price"] = float(predictions[-1]) if predictions else None
                summary["ensemble_confidence"] = ensemble_data.get("confidence", 0.5)

        return summary

    def _extract_technical_summary(self, analysis_results: Dict) -> Dict:
        """テクニカルサマリー抽出"""
        summary = {}

        if "technical_analysis" in analysis_results:
            tech_data = analysis_results["technical_analysis"]

            if "RSI" in tech_data and "rsi" in tech_data["RSI"]:
                rsi_values = tech_data["RSI"]["rsi"]
                if rsi_values:
                    current_rsi = rsi_values[-1]
                    summary["current_rsi"] = float(current_rsi)
                    if current_rsi > 70:
                        summary["rsi_signal"] = "過買い"
                    elif current_rsi < 30:
                        summary["rsi_signal"] = "過売り"
                    else:
                        summary["rsi_signal"] = "中立"

        return summary

    def _extract_risk_summary(self, analysis_results: Dict) -> Dict:
        """リスクサマリー抽出"""
        summary = {}

        if "garch_prediction" in analysis_results:
            garch_data = analysis_results["garch_prediction"]
            if "risk_metrics" in garch_data:
                risk_metrics = garch_data["risk_metrics"]
                if "var_95" in risk_metrics:
                    var_95 = risk_metrics["var_95"]
                    if isinstance(var_95, list) and var_95:
                        summary["var_95"] = float(var_95[-1])

        return summary

    def _extract_model_performance(self, analysis_results: Dict) -> Dict:
        """モデルパフォーマンス抽出"""
        performance = {}

        models = ["lstm_prediction", "garch_prediction", "ensemble_prediction"]
        for model in models:
            if model in analysis_results:
                model_data = analysis_results[model]
                if "accuracy_metrics" in model_data:
                    metrics = model_data["accuracy_metrics"]
                    performance[model] = {
                        "mse": metrics.get("mse"),
                        "mae": metrics.get("mae"),
                        "r2_score": metrics.get("r2_score"),
                    }

        return performance

    def _generate_key_insights(self, data: pd.DataFrame, analysis_results: Dict) -> List[str]:
        """主要インサイト生成"""
        insights = []

        # 価格トレンドインサイト
        if "Close" in data.columns and len(data) >= 2:
            recent_return = (data["Close"].iloc[-1] / data["Close"].iloc[-2] - 1) * 100
            if abs(recent_return) > 2:
                direction = "上昇" if recent_return > 0 else "下落"
                insights.append(f"直近の価格は{recent_return:.1f}%の{direction}を示しています")

        # 予測インサイト
        prediction_summary = self._extract_prediction_summary(analysis_results)
        if "ensemble_confidence" in prediction_summary:
            confidence = prediction_summary["ensemble_confidence"]
            if confidence > 0.8:
                insights.append("アンサンブル予測の信頼度が高く、予測精度が期待できます")
            elif confidence < 0.5:
                insights.append("予測信頼度が低く、市場環境が不安定な可能性があります")

        # テクニカルインサイト
        technical_summary = self._extract_technical_summary(analysis_results)
        if "rsi_signal" in technical_summary:
            rsi_signal = technical_summary["rsi_signal"]
            if rsi_signal != "中立":
                insights.append(f"RSI指標は{rsi_signal}を示しており、注意が必要です")

        return insights

    def _extract_key_metrics(self, analysis_results: Dict) -> Dict:
        """主要指標抽出"""
        key_metrics = {}

        # 予測精度
        if "ensemble_prediction" in analysis_results:
            ensemble_data = analysis_results["ensemble_prediction"]
            key_metrics["prediction_accuracy"] = ensemble_data.get("confidence", 0.5)

        # リスク指標
        if "garch_prediction" in analysis_results:
            garch_data = analysis_results["garch_prediction"]
            if "risk_metrics" in garch_data:
                risk_metrics = garch_data["risk_metrics"]
                key_metrics["risk_level"] = risk_metrics.get("var_95", 0)

        return key_metrics

    def _generate_investment_recommendation(self, analysis_results: Dict) -> str:
        """投資推奨生成"""
        prediction_summary = self._extract_prediction_summary(analysis_results)

        ensemble_confidence = prediction_summary.get("ensemble_confidence", 0.5)
        technical_summary = self._extract_technical_summary(analysis_results)

        if ensemble_confidence > 0.8:
            if technical_summary.get("rsi_signal") == "過売り":
                return "買い推奨"
            elif technical_summary.get("rsi_signal") == "過買い":
                return "売り検討"
            else:
                return "ホールド"
        elif ensemble_confidence < 0.5:
            return "様子見"
        else:
            return "中立"

    def _generate_risk_assessment(self, analysis_results: Dict) -> str:
        """リスク評価生成"""
        risk_summary = self._extract_risk_summary(analysis_results)

        if "var_95" in risk_summary:
            var_95 = abs(risk_summary["var_95"])
            if var_95 > 0.05:
                return "高リスク"
            elif var_95 > 0.02:
                return "中リスク"
            else:
                return "低リスク"

        return "リスク評価不可"

    def _generate_summary_points(self, analysis_results: Dict) -> List[str]:
        """サマリーポイント生成"""
        points = []

        prediction_summary = self._extract_prediction_summary(analysis_results)
        if prediction_summary:
            if "ensemble_confidence" in prediction_summary:
                confidence = prediction_summary["ensemble_confidence"]
                points.append(f"アンサンブル予測信頼度: {confidence:.1%}")

        technical_summary = self._extract_technical_summary(analysis_results)
        if "rsi_signal" in technical_summary:
            points.append(f"RSIシグナル: {technical_summary['rsi_signal']}")

        return points

    def _generate_next_actions(self, analysis_results: Dict) -> List[str]:
        """次のアクション生成"""
        actions = []

        prediction_summary = self._extract_prediction_summary(analysis_results)
        ensemble_confidence = prediction_summary.get("ensemble_confidence", 0.5)

        if ensemble_confidence < 0.5:
            actions.append("追加データの収集を検討してください")
            actions.append("モデルパラメータの調整が必要です")

        technical_summary = self._extract_technical_summary(analysis_results)
        if technical_summary.get("rsi_signal") in ["過買い", "過売り"]:
            actions.append("テクニカル指標による売買タイミングを監視してください")

        return actions

    def _compile_predictions_data(self, analysis_results: Dict) -> Dict:
        """予測データコンパイル"""
        compiled_data = {}

        models = ["lstm_prediction", "garch_prediction", "ensemble_prediction"]
        for model in models:
            if model in analysis_results and "predictions" in analysis_results[model]:
                predictions = analysis_results[model]["predictions"]
                compiled_data[model] = predictions

        return compiled_data

    def _format_price_summary_html(self, data: pd.DataFrame) -> str:
        """価格サマリーHTML形式"""
        price_summary = self._extract_price_summary(data)
        if not price_summary:
            return "<p>価格データが利用できません</p>"

        return f"""
        <div class="metric">現在価格: {price_summary.get('current_price', 'N/A'):.2f}</div>
        <div class="metric">期間リターン: {price_summary.get('period_return', 'N/A'):.1f}%</div>
        <div class="metric">ボラティリティ: {price_summary.get('volatility', 'N/A'):.1f}%</div>
        """

    def _format_prediction_summary_html(self, analysis_results: Dict) -> str:
        """予測サマリーHTML形式"""
        pred_summary = self._extract_prediction_summary(analysis_results)
        if not pred_summary:
            return "<p>予測データが利用できません</p>"

        html_content = ""
        if "ensemble_next_price" in pred_summary:
            html_content += f'<div class="metric">アンサンブル予測価格: {pred_summary["ensemble_next_price"]:.2f}</div>'
        if "ensemble_confidence" in pred_summary:
            html_content += f'<div class="metric">予測信頼度: {pred_summary["ensemble_confidence"]:.1%}</div>'

        return html_content or "<p>予測データが利用できません</p>"

    def _format_risk_summary_html(self, analysis_results: Dict) -> str:
        """リスクサマリーHTML形式"""
        risk_summary = self._extract_risk_summary(analysis_results)
        if not risk_summary:
            return "<p>リスクデータが利用できません</p>"

        html_content = ""
        if "var_95" in risk_summary:
            html_content += f'<div class="metric">VaR(95%): {risk_summary["var_95"]:.4f}</div>'

        return html_content or "<p>リスクデータが利用できません</p>"

    def _format_insights_html(self, data: pd.DataFrame, analysis_results: Dict) -> str:
        """インサイトHTML形式"""
        insights = self._generate_key_insights(data, analysis_results)
        if not insights:
            return "<p>主要インサイトが生成できませんでした</p>"

        html_content = ""
        for insight in insights:
            html_content += f'<div class="insight">{insight}</div>'

        return html_content

    def _format_file_list_html(self, generated_files: Dict) -> str:
        """ファイルリストHTML形式"""
        categories = {
            "summary": "サマリーレポート",
            "detailed": "詳細レポート",
            "comprehensive": "総合レポート",
            "executive": "エグゼクティブサマリー",
            "data": "データファイル",
            "analysis": "分析結果",
        }

        html_content = ""
        for category, title in categories.items():
            category_files = {k: v for k, v in generated_files.items() if category in k}
            if category_files:
                html_content += f'<div class="category">{title}</div>'
                for file_key, file_path in category_files.items():
                    filename = Path(file_path).name
                    html_content += f'<a href="{filename}" class="file-item">{filename} ({file_key})</a>'

        # その他のファイル
        other_files = {k: v for k, v in generated_files.items()
                      if not any(cat in k for cat in categories.keys())}
        if other_files:
            html_content += '<div class="category">その他のファイル</div>'
            for file_key, file_path in other_files.items():
                filename = Path(file_path).name
                html_content += f'<a href="{filename}" class="file-item">{filename} ({file_key})</a>'

        return html_content

    def _create_summary_page(self, data: pd.DataFrame, analysis_results: Dict, symbol: str):
        """サマリーページ作成（matplotlib）"""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4サイズ
            ax.axis('off')

            # タイトル
            fig.suptitle(f'{symbol} 分析サマリー', fontsize=20, fontweight='bold')

            # サマリー情報をテキストで表示
            summary_text = self._generate_summary_text(data, analysis_results)
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace')

            return fig

        except Exception as e:
            logger.error(f"サマリーページ作成エラー: {e}")
            return None

    def _generate_summary_text(self, data: pd.DataFrame, analysis_results: Dict) -> str:
        """サマリーテキスト生成"""
        price_summary = self._extract_price_summary(data)
        pred_summary = self._extract_prediction_summary(analysis_results)
        tech_summary = self._extract_technical_summary(analysis_results)

        summary_lines = [
            "=" * 50,
            "価格サマリー",
            "=" * 50,
            f"現在価格: {price_summary.get('current_price', 'N/A'):.2f}",
            f"期間リターン: {price_summary.get('period_return', 'N/A'):.1f}%",
            f"ボラティリティ: {price_summary.get('volatility', 'N/A'):.1f}%",
            "",
            "予測サマリー",
            "=" * 50,
        ]

        if "ensemble_next_price" in pred_summary:
            summary_lines.append(f"アンサンブル予測: {pred_summary['ensemble_next_price']:.2f}")
        if "ensemble_confidence" in pred_summary:
            summary_lines.append(f"予測信頼度: {pred_summary['ensemble_confidence']:.1%}")

        summary_lines.extend(["", "テクニカル分析", "=" * 50])
        if "current_rsi" in tech_summary:
            summary_lines.append(f"現在RSI: {tech_summary['current_rsi']:.1f}")
        if "rsi_signal" in tech_summary:
            summary_lines.append(f"RSIシグナル: {tech_summary['rsi_signal']}")

        return "\n".join(summary_lines)

    def _load_image_as_figure(self, image_path: str):
        """画像ファイルをmatplotlib図として読み込み"""
        try:
            if Path(image_path).exists():
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                import matplotlib.image as mpimg
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.axis('off')
                return fig
        except Exception as e:
            logger.warning(f"画像読み込みエラー {image_path}: {e}")
        return None
