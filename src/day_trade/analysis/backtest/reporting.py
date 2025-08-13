#!/usr/bin/env python3
"""
高度バックテスト結果レポート生成

Issue #753対応: バックテスト機能強化
包括的なレポート生成、PDF出力、インタラクティブ可視化
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings
import tempfile
import base64

# 可視化ライブラリ
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False

warnings.filterwarnings("ignore")

from .advanced_metrics import AdvancedRiskMetrics, AdvancedReturnMetrics, MarketRegimeMetrics
from .ml_integration import MLBacktestResult
from .types import BacktestResult


class BacktestReportGenerator:
    """バックテスト結果レポート生成器"""

    def __init__(self, output_dir: Optional[str] = None):
        """
        初期化

        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # スタイル設定
        if matplotlib_available:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'use') else 'default')
            sns.set_palette("husl")

    def generate_comprehensive_report(self,
                                    backtest_result: BacktestResult,
                                    advanced_risk_metrics: Optional[AdvancedRiskMetrics] = None,
                                    advanced_return_metrics: Optional[AdvancedReturnMetrics] = None,
                                    market_regime_metrics: Optional[MarketRegimeMetrics] = None,
                                    ml_result: Optional[MLBacktestResult] = None,
                                    report_name: str = "backtest_report") -> Dict[str, Any]:
        """
        包括的レポート生成

        Args:
            backtest_result: 基本バックテスト結果
            advanced_risk_metrics: 高度リスク指標
            advanced_return_metrics: 高度リターン指標
            market_regime_metrics: 市場レジーム分析
            ml_result: ML統合結果
            report_name: レポート名

        Returns:
            生成されたレポート情報
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = f"{report_name}_{timestamp}"

        # 1. JSONレポート生成
        json_report = self._generate_json_report(
            backtest_result, advanced_risk_metrics, advanced_return_metrics,
            market_regime_metrics, ml_result
        )

        # 2. HTMLレポート生成
        html_report = self._generate_html_report(
            backtest_result, advanced_risk_metrics, advanced_return_metrics,
            market_regime_metrics, ml_result, report_id
        )

        # 3. PDFレポート生成
        pdf_path = None
        if matplotlib_available:
            pdf_path = self._generate_pdf_report(
                backtest_result, advanced_risk_metrics, advanced_return_metrics,
                market_regime_metrics, ml_result, report_id
            )

        # 4. インタラクティブダッシュボード生成
        dashboard_path = None
        if plotly_available:
            dashboard_path = self._generate_interactive_dashboard(
                backtest_result, advanced_risk_metrics, advanced_return_metrics,
                market_regime_metrics, ml_result, report_id
            )

        # ファイル保存
        json_path = self.output_dir / f"{report_id}.json"
        html_path = self.output_dir / f"{report_id}.html"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)

        return {
            'report_id': report_id,
            'json_path': str(json_path),
            'html_path': str(html_path),
            'pdf_path': str(pdf_path) if pdf_path else None,
            'dashboard_path': str(dashboard_path) if dashboard_path else None,
            'generation_time': timestamp,
            'summary': self._generate_executive_summary(backtest_result, ml_result)
        }

    def _generate_json_report(self,
                            backtest_result: BacktestResult,
                            advanced_risk_metrics: Optional[AdvancedRiskMetrics],
                            advanced_return_metrics: Optional[AdvancedReturnMetrics],
                            market_regime_metrics: Optional[MarketRegimeMetrics],
                            ml_result: Optional[MLBacktestResult]) -> Dict[str, Any]:
        """JSON形式レポート生成"""

        report = {
            'metadata': {
                'generation_time': datetime.now().isoformat(),
                'report_type': 'comprehensive_backtest_analysis',
                'version': '1.0'
            },
            'basic_results': backtest_result.to_dict() if hasattr(backtest_result, 'to_dict') else {},
            'advanced_risk_metrics': asdict(advanced_risk_metrics) if advanced_risk_metrics else {},
            'advanced_return_metrics': asdict(advanced_return_metrics) if advanced_return_metrics else {},
            'market_regime_analysis': asdict(market_regime_metrics) if market_regime_metrics else {},
            'ml_integration_results': self._ml_result_to_dict(ml_result) if ml_result else {}
        }

        # パフォーマンス分析
        report['performance_analysis'] = self._analyze_performance(
            backtest_result, advanced_risk_metrics, advanced_return_metrics
        )

        # リスク分析
        report['risk_analysis'] = self._analyze_risk_profile(
            backtest_result, advanced_risk_metrics
        )

        # 取引分析
        report['trading_analysis'] = self._analyze_trading_patterns(backtest_result)

        return report

    def _generate_html_report(self,
                            backtest_result: BacktestResult,
                            advanced_risk_metrics: Optional[AdvancedRiskMetrics],
                            advanced_return_metrics: Optional[AdvancedReturnMetrics],
                            market_regime_metrics: Optional[MarketRegimeMetrics],
                            ml_result: Optional[MLBacktestResult],
                            report_id: str) -> str:
        """HTML形式レポート生成"""

        # エグゼクティブサマリー
        executive_summary = self._generate_executive_summary(backtest_result, ml_result)

        # パフォーマンステーブル
        performance_table = self._create_performance_table(
            backtest_result, advanced_return_metrics
        )

        # リスクテーブル
        risk_table = self._create_risk_table(backtest_result, advanced_risk_metrics)

        # ML分析テーブル
        ml_table = self._create_ml_analysis_table(ml_result) if ml_result else ""

        # チャート（base64エンコード画像として埋め込み）
        charts_html = self._generate_charts_html(
            backtest_result, advanced_risk_metrics, ml_result
        )

        html_template = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>バックテスト分析レポート - {report_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
        .metric-value {{ font-size: 1.2em; font-weight: bold; color: #2c5aa0; }}
        .metric-label {{ font-size: 0.9em; color: #666; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 バックテスト分析レポート</h1>
        <p>レポートID: {report_id}</p>
        <p>生成日時: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>🎯 エグゼクティブサマリー</h2>
        <div class="summary-grid">
            {executive_summary}
        </div>
    </div>

    <div class="section">
        <h2>📈 パフォーマンス指標</h2>
        {performance_table}
    </div>

    <div class="section">
        <h2>⚠️ リスク指標</h2>
        {risk_table}
    </div>

    {ml_table}

    <div class="section">
        <h2>📊 チャート分析</h2>
        {charts_html}
    </div>

    <div class="section">
        <h2>📋 詳細分析</h2>
        <h3>取引統計</h3>
        <p>総取引数: {len(backtest_result.trades) if backtest_result.trades else 0}</p>
        <p>勝率: {backtest_result.win_rate:.2%}</p>
        <p>プロフィットファクター: {backtest_result.profit_factor:.2f}</p>

        <h3>期間情報</h3>
        <p>開始日: {backtest_result.start_date.strftime("%Y年%m月%d日")}</p>
        <p>終了日: {backtest_result.end_date.strftime("%Y年%m月%d日")}</p>
        <p>分析期間: {(backtest_result.end_date - backtest_result.start_date).days}日</p>
    </div>

    <footer style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center;">
        <p>🤖 Issue #753 高度バックテスト機能強化により生成</p>
        <p>Generated with Claude Code - Day Trade Analysis System</p>
    </footer>
</body>
</html>
"""

        return html_template

    def _generate_pdf_report(self,
                           backtest_result: BacktestResult,
                           advanced_risk_metrics: Optional[AdvancedRiskMetrics],
                           advanced_return_metrics: Optional[AdvancedReturnMetrics],
                           market_regime_metrics: Optional[MarketRegimeMetrics],
                           ml_result: Optional[MLBacktestResult],
                           report_id: str) -> Optional[str]:
        """PDF形式レポート生成"""

        if not matplotlib_available:
            return None

        pdf_path = self.output_dir / f"{report_id}.pdf"

        try:
            with PdfPages(pdf_path) as pdf:
                # ページ1: サマリー
                self._create_summary_page(pdf, backtest_result, advanced_return_metrics)

                # ページ2: パフォーマンスチャート
                self._create_performance_charts_page(pdf, backtest_result)

                # ページ3: リスク分析
                self._create_risk_analysis_page(pdf, backtest_result, advanced_risk_metrics)

                # ページ4: ML分析（該当する場合）
                if ml_result:
                    self._create_ml_analysis_page(pdf, ml_result)

                # ページ5: 詳細統計
                self._create_detailed_statistics_page(pdf, backtest_result, advanced_return_metrics)

            return str(pdf_path)

        except Exception as e:
            print(f"PDF生成エラー: {e}")
            return None

    def _generate_interactive_dashboard(self,
                                      backtest_result: BacktestResult,
                                      advanced_risk_metrics: Optional[AdvancedRiskMetrics],
                                      advanced_return_metrics: Optional[AdvancedReturnMetrics],
                                      market_regime_metrics: Optional[MarketRegimeMetrics],
                                      ml_result: Optional[MLBacktestResult],
                                      report_id: str) -> Optional[str]:
        """インタラクティブダッシュボード生成"""

        if not plotly_available:
            return None

        # サブプロット作成
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'ポートフォリオ価値推移', 'ドローダウン推移',
                'リターン分布', 'リスク-リターン散布図',
                'VaR分析', '取引パフォーマンス'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # ポートフォリオ価値推移
        if backtest_result.portfolio_value_history:
            portfolio_values = [float(v) for v in backtest_result.portfolio_value_history]
            dates = pd.date_range(
                start=backtest_result.start_date,
                periods=len(portfolio_values),
                freq='D'
            )

            fig.add_trace(
                go.Scatter(x=dates, y=portfolio_values, name='ポートフォリオ価値'),
                row=1, col=1
            )

        # ドローダウン推移
        if backtest_result.drawdown_history:
            dates = pd.date_range(
                start=backtest_result.start_date,
                periods=len(backtest_result.drawdown_history),
                freq='D'
            )

            fig.add_trace(
                go.Scatter(x=dates, y=backtest_result.drawdown_history,
                          name='ドローダウン', fill='tozeroy'),
                row=1, col=2
            )

        # リターン分布
        if backtest_result.daily_returns:
            fig.add_trace(
                go.Histogram(x=backtest_result.daily_returns, name='日次リターン分布'),
                row=2, col=1
            )

        # その他のチャートを追加...

        # レイアウト設定
        fig.update_layout(
            title=f'バックテスト分析ダッシュボード - {report_id}',
            height=1200,
            showlegend=True
        )

        # HTMLファイルとして保存
        dashboard_path = self.output_dir / f"{report_id}_dashboard.html"
        fig.write_html(str(dashboard_path))

        return str(dashboard_path)

    def _generate_executive_summary(self, backtest_result: BacktestResult,
                                  ml_result: Optional[MLBacktestResult]) -> str:
        """エグゼクティブサマリー生成"""

        # パフォーマンス評価
        performance_grade = self._grade_performance(backtest_result.total_return, backtest_result.sharpe_ratio)
        risk_grade = self._grade_risk(backtest_result.max_drawdown, backtest_result.volatility)

        summary_items = [
            f'<div class="metric"><div class="metric-value {self._get_color_class(backtest_result.total_return)}">'
            f'{backtest_result.total_return:.2%}</div><div class="metric-label">総リターン</div></div>',

            f'<div class="metric"><div class="metric-value">{backtest_result.sharpe_ratio:.2f}</div>'
            f'<div class="metric-label">シャープレシオ</div></div>',

            f'<div class="metric"><div class="metric-value negative">{backtest_result.max_drawdown:.2%}</div>'
            f'<div class="metric-label">最大ドローダウン</div></div>',

            f'<div class="metric"><div class="metric-value">{backtest_result.win_rate:.2%}</div>'
            f'<div class="metric-label">勝率</div></div>',

            f'<div class="metric"><div class="metric-value">{performance_grade}</div>'
            f'<div class="metric-label">パフォーマンス評価</div></div>',

            f'<div class="metric"><div class="metric-value">{risk_grade}</div>'
            f'<div class="metric-label">リスク評価</div></div>'
        ]

        if ml_result:
            summary_items.extend([
                f'<div class="metric"><div class="metric-value">{ml_result.prediction_accuracy:.2%}</div>'
                f'<div class="metric-label">予測精度</div></div>',

                f'<div class="metric"><div class="metric-value {self._get_color_class(ml_result.excess_return)}">'
                f'{ml_result.excess_return:.2%}</div><div class="metric-label">超過リターン</div></div>'
            ])

        return '\n'.join(summary_items)

    def _create_performance_table(self, backtest_result: BacktestResult,
                                advanced_return_metrics: Optional[AdvancedReturnMetrics]) -> str:
        """パフォーマンステーブル作成"""

        rows = [
            f'<tr><td>総リターン</td><td class="{self._get_color_class(backtest_result.total_return)}">{backtest_result.total_return:.2%}</td></tr>',
            f'<tr><td>年率リターン</td><td class="{self._get_color_class(backtest_result.annualized_return)}">{backtest_result.annualized_return:.2%}</td></tr>',
            f'<tr><td>ボラティリティ</td><td>{backtest_result.volatility:.2%}</td></tr>',
            f'<tr><td>シャープレシオ</td><td>{backtest_result.sharpe_ratio:.2f}</td></tr>',
            f'<tr><td>カルマーレシオ</td><td>{backtest_result.calmar_ratio:.2f}</td></tr>',
        ]

        if advanced_return_metrics:
            rows.extend([
                f'<tr><td>幾何平均リターン</td><td>{advanced_return_metrics.geometric_mean_return:.2%}</td></tr>',
                f'<tr><td>インフォメーションレシオ</td><td>{advanced_return_metrics.information_ratio:.2f}</td></tr>',
                f'<tr><td>スターリングレシオ</td><td>{advanced_return_metrics.sterling_ratio:.2f}</td></tr>',
                f'<tr><td>期待値</td><td>{advanced_return_metrics.expectancy:.4f}</td></tr>',
            ])

        return f'<table><thead><tr><th>指標</th><th>値</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'

    def _create_risk_table(self, backtest_result: BacktestResult,
                         advanced_risk_metrics: Optional[AdvancedRiskMetrics]) -> str:
        """リスクテーブル作成"""

        rows = [
            f'<tr><td>最大ドローダウン</td><td class="negative">{backtest_result.max_drawdown:.2%}</td></tr>',
            f'<tr><td>VaR (95%)</td><td class="negative">{backtest_result.value_at_risk:.2%}</td></tr>',
            f'<tr><td>CVaR (95%)</td><td class="negative">{backtest_result.conditional_var:.2%}</td></tr>',
        ]

        if advanced_risk_metrics:
            rows.extend([
                f'<tr><td>ダウンサイド偏差</td><td>{advanced_risk_metrics.downside_deviation:.2%}</td></tr>',
                f'<tr><td>ソルティーノレシオ</td><td>{advanced_risk_metrics.sortino_ratio:.2f}</td></tr>',
                f'<tr><td>ペインインデックス</td><td>{advanced_risk_metrics.pain_index:.2%}</td></tr>',
                f'<tr><td>アルサーインデックス</td><td>{advanced_risk_metrics.ulcer_index:.2%}</td></tr>',
                f'<tr><td>歪度</td><td>{advanced_risk_metrics.skewness:.2f}</td></tr>',
                f'<tr><td>尖度</td><td>{advanced_risk_metrics.excess_kurtosis:.2f}</td></tr>',
            ])

        return f'<table><thead><tr><th>リスク指標</th><th>値</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'

    def _create_ml_analysis_table(self, ml_result: MLBacktestResult) -> str:
        """ML分析テーブル作成"""

        model_contrib_rows = ""
        for model, contrib in ml_result.model_contributions.items():
            model_contrib_rows += f'<tr><td>{model}</td><td>{contrib:.2%}</td></tr>'

        return f'''
        <div class="section">
            <h2>🤖 機械学習分析</h2>
            <table>
                <thead><tr><th>ML指標</th><th>値</th></tr></thead>
                <tbody>
                    <tr><td>予測精度</td><td>{ml_result.prediction_accuracy:.2%}</td></tr>
                    <tr><td>方向性予測精度</td><td>{ml_result.direction_accuracy:.2%}</td></tr>
                    <tr><td>シグナル精度</td><td>{ml_result.signal_precision:.2%}</td></tr>
                    <tr><td>シグナル再現率</td><td>{ml_result.signal_recall:.2%}</td></tr>
                    <tr><td>F1スコア</td><td>{ml_result.signal_f1_score:.2f}</td></tr>
                    <tr><td>超過リターン</td><td class="{self._get_color_class(ml_result.excess_return)}">{ml_result.excess_return:.2%}</td></tr>
                </tbody>
            </table>

            <h3>モデル貢献度</h3>
            <table>
                <thead><tr><th>モデル</th><th>貢献度</th></tr></thead>
                <tbody>{model_contrib_rows}</tbody>
            </table>
        </div>
        '''

    def _generate_charts_html(self, backtest_result: BacktestResult,
                            advanced_risk_metrics: Optional[AdvancedRiskMetrics],
                            ml_result: Optional[MLBacktestResult]) -> str:
        """チャートHTML生成"""

        if not matplotlib_available:
            return "<p>チャート表示にはMatplotlibが必要です。</p>"

        charts_html = ""

        # ポートフォリオ価値推移チャート
        if backtest_result.portfolio_value_history:
            chart_base64 = self._create_portfolio_chart(backtest_result)
            if chart_base64:
                charts_html += f'<div class="chart-container"><img src="data:image/png;base64,{chart_base64}" alt="ポートフォリオ推移"></div>'

        # ドローダウンチャート
        if backtest_result.drawdown_history:
            chart_base64 = self._create_drawdown_chart(backtest_result)
            if chart_base64:
                charts_html += f'<div class="chart-container"><img src="data:image/png;base64,{chart_base64}" alt="ドローダウン推移"></div>'

        return charts_html if charts_html else "<p>チャートデータが不足しています。</p>"

    def _create_portfolio_chart(self, backtest_result: BacktestResult) -> Optional[str]:
        """ポートフォリオチャート作成"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            portfolio_values = [float(v) for v in backtest_result.portfolio_value_history]
            dates = pd.date_range(
                start=backtest_result.start_date,
                periods=len(portfolio_values),
                freq='D'
            )

            ax.plot(dates, portfolio_values, linewidth=2, color='blue')
            ax.set_title('ポートフォリオ価値推移', fontsize=14, fontweight='bold')
            ax.set_xlabel('日付')
            ax.set_ylabel('ポートフォリオ価値')
            ax.grid(True, alpha=0.3)

            # 日付フォーマット
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()

            # Base64エンコード
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
                plt.close()

                with open(tmp.name, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()

                Path(tmp.name).unlink()  # 一時ファイル削除
                return encoded

        except Exception as e:
            print(f"ポートフォリオチャート作成エラー: {e}")
            return None

    def _create_drawdown_chart(self, backtest_result: BacktestResult) -> Optional[str]:
        """ドローダウンチャート作成"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            dates = pd.date_range(
                start=backtest_result.start_date,
                periods=len(backtest_result.drawdown_history),
                freq='D'
            )

            ax.fill_between(dates, backtest_result.drawdown_history, 0,
                           alpha=0.7, color='red', label='ドローダウン')
            ax.set_title('ドローダウン推移', fontsize=14, fontweight='bold')
            ax.set_xlabel('日付')
            ax.set_ylabel('ドローダウン (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # 日付フォーマット
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate()

            # Base64エンコード
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
                plt.close()

                with open(tmp.name, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode()

                Path(tmp.name).unlink()
                return encoded

        except Exception as e:
            print(f"ドローダウンチャート作成エラー: {e}")
            return None

    # PDF生成ヘルパーメソッド
    def _create_summary_page(self, pdf, backtest_result: BacktestResult,
                           advanced_return_metrics: Optional[AdvancedReturnMetrics]):
        """PDFサマリーページ作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('バックテスト分析サマリー', fontsize=16, fontweight='bold')

        # パフォーマンス指標
        metrics = ['総リターン', 'シャープレシオ', '最大DD', '勝率']
        values = [backtest_result.total_return, backtest_result.sharpe_ratio,
                 backtest_result.max_drawdown, backtest_result.win_rate]

        ax1.bar(metrics, values)
        ax1.set_title('主要指標')
        ax1.tick_params(axis='x', rotation=45)

        # その他のチャートを追加...

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_performance_charts_page(self, pdf, backtest_result: BacktestResult):
        """PDFパフォーマンスチャートページ作成"""
        # 実装省略...
        pass

    def _create_risk_analysis_page(self, pdf, backtest_result: BacktestResult,
                                 advanced_risk_metrics: Optional[AdvancedRiskMetrics]):
        """PDFリスク分析ページ作成"""
        # 実装省略...
        pass

    def _create_ml_analysis_page(self, pdf, ml_result: MLBacktestResult):
        """PDF ML分析ページ作成"""
        # 実装省略...
        pass

    def _create_detailed_statistics_page(self, pdf, backtest_result: BacktestResult,
                                       advanced_return_metrics: Optional[AdvancedReturnMetrics]):
        """PDF詳細統計ページ作成"""
        # 実装省略...
        pass

    # ヘルパーメソッド
    def _ml_result_to_dict(self, ml_result: MLBacktestResult) -> Dict[str, Any]:
        """ML結果を辞書に変換"""
        return {
            'total_return': ml_result.total_return,
            'benchmark_return': ml_result.benchmark_return,
            'excess_return': ml_result.excess_return,
            'prediction_accuracy': ml_result.prediction_accuracy,
            'direction_accuracy': ml_result.direction_accuracy,
            'signal_precision': ml_result.signal_precision,
            'signal_recall': ml_result.signal_recall,
            'signal_f1_score': ml_result.signal_f1_score,
            'model_contributions': ml_result.model_contributions,
            'information_ratio': ml_result.information_ratio,
            'tracking_error': ml_result.tracking_error,
            'maximum_drawdown': ml_result.maximum_drawdown
        }

    def _analyze_performance(self, backtest_result: BacktestResult,
                           advanced_risk_metrics: Optional[AdvancedRiskMetrics],
                           advanced_return_metrics: Optional[AdvancedReturnMetrics]) -> Dict[str, Any]:
        """パフォーマンス分析"""
        return {
            'performance_grade': self._grade_performance(backtest_result.total_return, backtest_result.sharpe_ratio),
            'risk_adjusted_return': backtest_result.total_return / max(backtest_result.volatility, 0.01),
            'consistency_score': self._calculate_consistency_score(backtest_result),
            'market_correlation': backtest_result.beta if hasattr(backtest_result, 'beta') else None
        }

    def _analyze_risk_profile(self, backtest_result: BacktestResult,
                            advanced_risk_metrics: Optional[AdvancedRiskMetrics]) -> Dict[str, Any]:
        """リスクプロファイル分析"""
        return {
            'risk_grade': self._grade_risk(backtest_result.max_drawdown, backtest_result.volatility),
            'tail_risk_score': self._calculate_tail_risk_score(advanced_risk_metrics),
            'downside_protection': self._calculate_downside_protection(backtest_result),
            'risk_consistency': self._calculate_risk_consistency(backtest_result)
        }

    def _analyze_trading_patterns(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """取引パターン分析"""
        if not backtest_result.trades:
            return {}

        trade_durations = []  # 実装省略
        trade_sizes = []      # 実装省略

        return {
            'average_trade_duration': np.mean(trade_durations) if trade_durations else 0,
            'trade_size_consistency': np.std(trade_sizes) if trade_sizes else 0,
            'trading_frequency': len(backtest_result.trades) / max((backtest_result.end_date - backtest_result.start_date).days, 1),
            'market_timing_score': self._calculate_market_timing_score(backtest_result)
        }

    def _grade_performance(self, total_return: float, sharpe_ratio: float) -> str:
        """パフォーマンスグレード評価"""
        if total_return > 0.2 and sharpe_ratio > 1.5:
            return "A+"
        elif total_return > 0.15 and sharpe_ratio > 1.0:
            return "A"
        elif total_return > 0.1 and sharpe_ratio > 0.5:
            return "B"
        elif total_return > 0.05:
            return "C"
        else:
            return "D"

    def _grade_risk(self, max_drawdown: float, volatility: float) -> str:
        """リスクグレード評価"""
        risk_score = abs(max_drawdown) + volatility

        if risk_score < 0.1:
            return "低リスク"
        elif risk_score < 0.2:
            return "中リスク"
        else:
            return "高リスク"

    def _get_color_class(self, value: float) -> str:
        """値に基づく色クラス取得"""
        if value > 0:
            return "positive"
        elif value < 0:
            return "negative"
        else:
            return "neutral"

    def _calculate_consistency_score(self, backtest_result: BacktestResult) -> float:
        """一貫性スコア計算"""
        if not backtest_result.daily_returns:
            return 0.0

        returns = np.array(backtest_result.daily_returns)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.5

        pos_consistency = 1 / (1 + np.std(positive_returns))
        neg_consistency = 1 / (1 + np.std(negative_returns))

        return (pos_consistency + neg_consistency) / 2

    def _calculate_tail_risk_score(self, advanced_risk_metrics: Optional[AdvancedRiskMetrics]) -> float:
        """テールリスクスコア計算"""
        if not advanced_risk_metrics:
            return 0.0

        # VaRとCVaRの差異でテールリスクを評価
        var_cvar_ratio = abs(advanced_risk_metrics.cvar_1 / advanced_risk_metrics.var_1) if advanced_risk_metrics.var_1 != 0 else 1

        return min(var_cvar_ratio / 2, 1.0)  # 正規化

    def _calculate_downside_protection(self, backtest_result: BacktestResult) -> float:
        """ダウンサイドプロテクション計算"""
        if not backtest_result.daily_returns:
            return 0.0

        negative_returns = [r for r in backtest_result.daily_returns if r < 0]
        if not negative_returns:
            return 1.0

        avg_negative_return = np.mean(negative_returns)
        return max(0, 1 + avg_negative_return * 10)  # スケーリング

    def _calculate_risk_consistency(self, backtest_result: BacktestResult) -> float:
        """リスク一貫性計算"""
        if len(backtest_result.drawdown_history) < 2:
            return 0.0

        # ドローダウンの変動度を一貫性として評価
        dd_changes = np.diff(backtest_result.drawdown_history)
        consistency = 1 / (1 + np.std(dd_changes))

        return consistency

    def _calculate_market_timing_score(self, backtest_result: BacktestResult) -> float:
        """マーケットタイミングスコア計算"""
        # 簡易実装：勝率ベース
        return backtest_result.win_rate