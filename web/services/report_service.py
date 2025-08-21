#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Service - レポート生成サービス
包括的なレポート生成とエクスポート機能
"""

import io
import json
import logging
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import base64
from enum import Enum

class ReportType(Enum):
    """レポートタイプ"""
    PORTFOLIO_SUMMARY = "PORTFOLIO_SUMMARY"
    PERFORMANCE_ANALYSIS = "PERFORMANCE_ANALYSIS"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    TRADING_JOURNAL = "TRADING_JOURNAL"
    MARKET_OVERVIEW = "MARKET_OVERVIEW"
    BACKTEST_REPORT = "BACKTEST_REPORT"

class ReportFormat(Enum):
    """レポートフォーマット"""
    HTML = "HTML"
    PDF = "PDF"
    JSON = "JSON"
    CSV = "CSV"
    EXCEL = "EXCEL"

@dataclass
class ReportMetadata:
    """レポートメタデータ"""
    report_id: str
    report_type: ReportType
    title: str
    description: str
    generated_at: str
    period_start: str
    period_end: str
    data_points: int
    file_size: int = 0

@dataclass
class ChartConfig:
    """チャート設定"""
    chart_type: str  # line, bar, pie, candlestick
    title: str
    x_label: str = ""
    y_label: str = ""
    width: int = 12
    height: int = 8
    color_scheme: str = "default"

class ChartGenerator:
    """チャート生成器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_portfolio_chart(self, portfolio_data: Dict[str, Any],
                                config: ChartConfig) -> str:
        """ポートフォリオチャート生成"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(config.width, config.height))
            fig.suptitle(config.title, fontsize=16, fontweight='bold')

            # 1. ポートフォリオ価値推移
            if 'value_history' in portfolio_data:
                dates = [datetime.fromisoformat(d['date']) for d in portfolio_data['value_history']]
                values = [d['value'] for d in portfolio_data['value_history']]

                ax1.plot(dates, values, linewidth=2, color='#2E86AB')
                ax1.set_title('ポートフォリオ価値推移')
                ax1.set_ylabel('価値 (円)')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

            # 2. セクター分散
            if 'sector_allocation' in portfolio_data:
                sectors = list(portfolio_data['sector_allocation'].keys())
                sizes = list(portfolio_data['sector_allocation'].values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(sectors)))

                ax2.pie(sizes, labels=sectors, autopct='%1.1f%%', colors=colors)
                ax2.set_title('セクター分散')

            # 3. 月次リターン
            if 'monthly_returns' in portfolio_data:
                months = list(portfolio_data['monthly_returns'].keys())
                returns = list(portfolio_data['monthly_returns'].values())
                colors = ['green' if r >= 0 else 'red' for r in returns]

                ax3.bar(months, returns, color=colors, alpha=0.7)
                ax3.set_title('月次リターン')
                ax3.set_ylabel('リターン (%)')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

            # 4. トップ銘柄
            if 'top_holdings' in portfolio_data:
                symbols = [h['symbol'] for h in portfolio_data['top_holdings'][:5]]
                weights = [h['weight'] for h in portfolio_data['top_holdings'][:5]]

                ax4.barh(symbols, weights, color='#A23B72')
                ax4.set_title('トップ5銘柄')
                ax4.set_xlabel('構成比 (%)')

            plt.tight_layout()

            # Base64エンコード
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            self.logger.error(f"ポートフォリオチャート生成エラー: {e}")
            return ""

    def generate_performance_chart(self, performance_data: Dict[str, Any],
                                 config: ChartConfig) -> str:
        """パフォーマンスチャート生成"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(config.width, config.height))
            fig.suptitle(config.title, fontsize=16, fontweight='bold')

            # 1. 累積リターン比較
            if 'cumulative_returns' in performance_data:
                dates = [datetime.fromisoformat(d) for d in performance_data['cumulative_returns']['dates']]
                portfolio_returns = performance_data['cumulative_returns']['portfolio']
                benchmark_returns = performance_data['cumulative_returns'].get('benchmark', [])

                ax1.plot(dates, portfolio_returns, label='ポートフォリオ', linewidth=2, color='#2E86AB')
                if benchmark_returns:
                    ax1.plot(dates, benchmark_returns, label='ベンチマーク', linewidth=2, color='#F24236', linestyle='--')
                ax1.set_title('累積リターン比較')
                ax1.set_ylabel('累積リターン (%)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. リスクリターン散布図
            if 'risk_return_data' in performance_data:
                risk_data = performance_data['risk_return_data']['risk']
                return_data = performance_data['risk_return_data']['return']
                symbols = performance_data['risk_return_data']['symbols']

                scatter = ax2.scatter(risk_data, return_data, alpha=0.6, s=100, c=range(len(symbols)), cmap='viridis')
                for i, symbol in enumerate(symbols):
                    ax2.annotate(symbol, (risk_data[i], return_data[i]), xytext=(5, 5), textcoords='offset points')
                ax2.set_title('リスク・リターン分析')
                ax2.set_xlabel('リスク (標準偏差 %)')
                ax2.set_ylabel('リターン (%)')
                ax2.grid(True, alpha=0.3)

            # 3. ドローダウン
            if 'drawdown_data' in performance_data:
                dates = [datetime.fromisoformat(d) for d in performance_data['drawdown_data']['dates']]
                drawdowns = performance_data['drawdown_data']['values']

                ax3.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
                ax3.plot(dates, drawdowns, color='red', linewidth=1)
                ax3.set_title('ドローダウン分析')
                ax3.set_ylabel('ドローダウン (%)')
                ax3.grid(True, alpha=0.3)

            # 4. 月次統計
            if 'monthly_stats' in performance_data:
                metrics = list(performance_data['monthly_stats'].keys())
                values = list(performance_data['monthly_stats'].values())

                ax4.barh(metrics, values, color=['green' if v >= 0 else 'red' for v in values])
                ax4.set_title('パフォーマンス指標')
                ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            self.logger.error(f"パフォーマンスチャート生成エラー: {e}")
            return ""

    def generate_risk_chart(self, risk_data: Dict[str, Any], config: ChartConfig) -> str:
        """リスクチャート生成"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(config.width, config.height))
            fig.suptitle(config.title, fontsize=16, fontweight='bold')

            # 1. VaRヒストリー
            if 'var_history' in risk_data:
                dates = [datetime.fromisoformat(d) for d in risk_data['var_history']['dates']]
                var_95 = risk_data['var_history']['var_95']
                var_99 = risk_data['var_history']['var_99']

                ax1.plot(dates, var_95, label='VaR 95%', linewidth=2, color='orange')
                ax1.plot(dates, var_99, label='VaR 99%', linewidth=2, color='red')
                ax1.set_title('VaR推移')
                ax1.set_ylabel('VaR (円)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. セクター集中度
            if 'sector_risk' in risk_data:
                sectors = list(risk_data['sector_risk'].keys())
                concentrations = list(risk_data['sector_risk'].values())

                bars = ax2.bar(sectors, concentrations, color='lightcoral', alpha=0.7)
                ax2.axhline(y=20, color='red', linestyle='--', label='リスク上限')
                ax2.set_title('セクター集中度リスク')
                ax2.set_ylabel('集中度 (%)')
                ax2.legend()
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # 3. ポジション別リスク
            if 'position_risks' in risk_data:
                symbols = [p['symbol'] for p in risk_data['position_risks']]
                risks = [p['risk_amount'] for p in risk_data['position_risks']]

                ax3.barh(symbols, risks, color='steelblue', alpha=0.7)
                ax3.set_title('ポジション別リスク')
                ax3.set_xlabel('リスク金額 (円)')

            # 4. リスク指標レーダーチャート
            if 'risk_metrics' in risk_data:
                metrics = list(risk_data['risk_metrics'].keys())
                values = list(risk_data['risk_metrics'].values())

                # 値を0-10スケールに正規化
                normalized_values = [(v / max(values)) * 10 for v in values]

                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))
                normalized_values = normalized_values + [normalized_values[0]]

                ax4 = plt.subplot(2, 2, 4, polar=True)
                ax4.plot(angles, normalized_values, 'o-', linewidth=2, color='red', alpha=0.7)
                ax4.fill(angles, normalized_values, alpha=0.25, color='red')
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(metrics)
                ax4.set_title('リスク指標')

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            self.logger.error(f"リスクチャート生成エラー: {e}")
            return ""

class HTMLReportGenerator:
    """HTMLレポート生成器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_portfolio_report(self, portfolio_data: Dict[str, Any],
                                performance_data: Dict[str, Any],
                                risk_data: Dict[str, Any]) -> str:
        """ポートフォリオレポート生成"""
        try:
            chart_generator = ChartGenerator()

            # チャート生成
            portfolio_chart = chart_generator.generate_portfolio_chart(
                portfolio_data, ChartConfig("portfolio", "ポートフォリオ概要")
            )

            performance_chart = chart_generator.generate_performance_chart(
                performance_data, ChartConfig("performance", "パフォーマンス分析")
            )

            risk_chart = chart_generator.generate_risk_chart(
                risk_data, ChartConfig("risk", "リスク分析")
            )

            # HTMLテンプレート
            html_template = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ポートフォリオレポート - {datetime.now().strftime('%Y年%m月%d日')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #2E86AB; padding-bottom: 20px; margin-bottom: 30px; }}
        .title {{ font-size: 28px; color: #2E86AB; margin: 0; }}
        .subtitle {{ color: #666; margin: 10px 0 0 0; }}
        .section {{ margin: 30px 0; }}
        .section-title {{ font-size: 20px; color: #333; border-left: 4px solid #2E86AB; padding-left: 15px; margin-bottom: 15px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .summary-table th, .summary-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .summary-table th {{ background-color: #2E86AB; color: white; }}
        .positive {{ color: #27AE60; }}
        .negative {{ color: #E74C3C; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">ポートフォリオレポート</h1>
            <p class="subtitle">生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
        </div>

        <div class="section">
            <h2 class="section-title">概要</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">¥{portfolio_data.get('total_value', 0):,.0f}</div>
                    <div class="metric-label">総資産</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if portfolio_data.get('total_pnl', 0) >= 0 else 'negative'}">
                        ¥{portfolio_data.get('total_pnl', 0):,.0f}
                    </div>
                    <div class="metric-label">未実現損益</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if portfolio_data.get('total_pnl_pct', 0) >= 0 else 'negative'}">
                        {portfolio_data.get('total_pnl_pct', 0):+.2f}%
                    </div>
                    <div class="metric-label">総リターン</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{portfolio_data.get('positions_count', 0)}</div>
                    <div class="metric-label">保有銘柄数</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">ポートフォリオ分析</h2>
            <div class="chart-container">
                <img src="{portfolio_chart}" alt="ポートフォリオチャート">
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">パフォーマンス分析</h2>
            <div class="chart-container">
                <img src="{performance_chart}" alt="パフォーマンスチャート">
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{performance_data.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">シャープレシオ</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_data.get('max_drawdown', 0):.2f}%</div>
                    <div class="metric-label">最大ドローダウン</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_data.get('volatility', 0):.2f}%</div>
                    <div class="metric-label">ボラティリティ</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_data.get('win_rate', 0):.1f}%</div>
                    <div class="metric-label">勝率</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">リスク分析</h2>
            <div class="chart-container">
                <img src="{risk_chart}" alt="リスクチャート">
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">¥{risk_data.get('var_95', 0):,.0f}</div>
                    <div class="metric-label">VaR (95%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">¥{risk_data.get('var_99', 0):,.0f}</div>
                    <div class="metric-label">VaR (99%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_data.get('total_risk_pct', 0):.2f}%</div>
                    <div class="metric-label">ポートフォリオリスク</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_data.get('concentration_risk', 'N/A')}</div>
                    <div class="metric-label">集中リスク</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">保有銘柄</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>銘柄</th>
                        <th>数量</th>
                        <th>現在価格</th>
                        <th>評価額</th>
                        <th>損益</th>
                        <th>損益率</th>
                        <th>構成比</th>
                    </tr>
                </thead>
                <tbody>
            """

            # 保有銘柄テーブル
            for position in portfolio_data.get('positions', []):
                pnl_class = 'positive' if position.get('unrealized_pnl', 0) >= 0 else 'negative'
                html_template += f"""
                    <tr>
                        <td>{position.get('symbol', 'N/A')}</td>
                        <td>{position.get('quantity', 0):,}</td>
                        <td>¥{position.get('current_price', 0):,.0f}</td>
                        <td>¥{position.get('market_value', 0):,.0f}</td>
                        <td class="{pnl_class}">¥{position.get('unrealized_pnl', 0):,.0f}</td>
                        <td class="{pnl_class}">{position.get('unrealized_pnl_pct', 0):+.2f}%</td>
                        <td>{position.get('weight', 0):.1f}%</td>
                    </tr>
                """

            html_template += f"""
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Day Trade Portfolio Management System</p>
            <p>このレポートは自動生成されました</p>
        </div>
    </div>
</body>
</html>
            """

            return html_template

        except Exception as e:
            self.logger.error(f"HTMLレポート生成エラー: {e}")
            return f"<html><body><h1>レポート生成エラー</h1><p>{str(e)}</p></body></html>"

class ReportService:
    """レポートサービス統合"""

    def __init__(self, reports_dir: str = "data/reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.chart_generator = ChartGenerator()
        self.html_generator = HTMLReportGenerator()
        self.logger = logging.getLogger(__name__)

    def generate_portfolio_report(self, portfolio_data: Dict[str, Any],
                                performance_data: Dict[str, Any] = None,
                                risk_data: Dict[str, Any] = None,
                                format: ReportFormat = ReportFormat.HTML) -> Dict[str, Any]:
        """ポートフォリオレポート生成"""
        try:
            report_id = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # デフォルトデータ補完
            if not performance_data:
                performance_data = self._generate_sample_performance_data()
            if not risk_data:
                risk_data = self._generate_sample_risk_data()

            if format == ReportFormat.HTML:
                content = self.html_generator.generate_portfolio_report(
                    portfolio_data, performance_data, risk_data
                )
                file_extension = ".html"
            elif format == ReportFormat.JSON:
                content = json.dumps({
                    'portfolio': portfolio_data,
                    'performance': performance_data,
                    'risk': risk_data,
                    'metadata': {
                        'report_id': report_id,
                        'generated_at': datetime.now().isoformat()
                    }
                }, ensure_ascii=False, indent=2)
                file_extension = ".json"
            else:
                raise ValueError(f"未対応のフォーマット: {format}")

            # ファイル保存
            file_path = self.reports_dir / f"{report_id}{file_extension}"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # メタデータ
            metadata = ReportMetadata(
                report_id=report_id,
                report_type=ReportType.PORTFOLIO_SUMMARY,
                title="ポートフォリオレポート",
                description="包括的なポートフォリオ分析レポート",
                generated_at=datetime.now().isoformat(),
                period_start=(datetime.now() - timedelta(days=30)).isoformat(),
                period_end=datetime.now().isoformat(),
                data_points=len(portfolio_data.get('positions', [])),
                file_size=len(content)
            )

            # Enumを文字列に変換してJSONシリアライズ可能にする
            metadata_dict = asdict(metadata)
            metadata_dict['report_type'] = metadata.report_type.value

            return {
                'success': True,
                'report_id': report_id,
                'file_path': str(file_path),
                'metadata': metadata_dict,
                'content': content if format == ReportFormat.JSON else None
            }

        except Exception as e:
            self.logger.error(f"ポートフォリオレポート生成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def generate_trading_journal(self, transactions: List[Dict[str, Any]],
                               period_days: int = 30) -> Dict[str, Any]:
        """取引ジャーナルレポート生成"""
        try:
            report_id = f"journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 期間フィルタリング
            cutoff_date = datetime.now() - timedelta(days=period_days)
            filtered_transactions = [
                t for t in transactions
                if datetime.fromisoformat(t.get('date', '2024-01-01')) >= cutoff_date
            ]

            # 統計計算
            total_trades = len(filtered_transactions)
            winning_trades = len([t for t in filtered_transactions if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in filtered_transactions if t.get('pnl', 0) < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(t.get('pnl', 0) for t in filtered_transactions)
            gross_profit = sum(t.get('pnl', 0) for t in filtered_transactions if t.get('pnl', 0) > 0)
            gross_loss = sum(t.get('pnl', 0) for t in filtered_transactions if t.get('pnl', 0) < 0)

            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

            # レポート内容
            journal_data = {
                'period': f"{cutoff_date.strftime('%Y-%m-%d')} - {datetime.now().strftime('%Y-%m-%d')}",
                'summary': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'gross_profit': gross_profit,
                    'gross_loss': gross_loss,
                    'profit_factor': profit_factor
                },
                'transactions': filtered_transactions,
                'generated_at': datetime.now().isoformat()
            }

            # JSON保存
            file_path = self.reports_dir / f"{report_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(journal_data, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'report_id': report_id,
                'file_path': str(file_path),
                'summary': journal_data['summary']
            }

        except Exception as e:
            self.logger.error(f"取引ジャーナル生成エラー: {e}")
            return {'success': False, 'error': str(e)}

    def _generate_sample_performance_data(self) -> Dict[str, Any]:
        """サンプルパフォーマンスデータ生成"""
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]

        return {
            'cumulative_returns': {
                'dates': dates,
                'portfolio': [round(np.random.normal(0, 1) * i * 0.1, 2) for i in range(len(dates))],
                'benchmark': [round(np.random.normal(0, 0.8) * i * 0.1, 2) for i in range(len(dates))]
            },
            'risk_return_data': {
                'symbols': ['7203', '9984', '8306'],
                'risk': [15.2, 22.1, 18.5],
                'return': [8.3, 12.1, 6.7]
            },
            'drawdown_data': {
                'dates': dates,
                'values': [round(min(0, np.random.normal(-1, 2)), 2) for _ in dates]
            },
            'monthly_stats': {
                'シャープレシオ': 1.23,
                'ソルティノレシオ': 1.45,
                'カルマーレシオ': 0.89,
                'トレイナーレシオ': 0.67
            },
            'sharpe_ratio': 1.23,
            'max_drawdown': -8.5,
            'volatility': 16.7,
            'win_rate': 67.3
        }

    def _generate_sample_risk_data(self) -> Dict[str, Any]:
        """サンプルリスクデータ生成"""
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10, 0, -1)]

        return {
            'var_history': {
                'dates': dates,
                'var_95': [round(np.random.uniform(10000, 50000), 0) for _ in dates],
                'var_99': [round(np.random.uniform(15000, 70000), 0) for _ in dates]
            },
            'sector_risk': {
                'Technology': 25.3,
                'Financial': 18.7,
                'Automotive': 15.2,
                'Healthcare': 12.1
            },
            'position_risks': [
                {'symbol': '7203', 'risk_amount': 45000},
                {'symbol': '9984', 'risk_amount': 38000},
                {'symbol': '8306', 'risk_amount': 27000}
            ],
            'risk_metrics': {
                'ボラティリティ': 16.7,
                '集中度': 8.3,
                '流動性': 7.2,
                '相関': 6.8,
                'レバレッジ': 4.1
            },
            'var_95': 42000,
            'var_99': 58000,
            'total_risk_pct': 2.1,
            'concentration_risk': 'Medium'
        }

    def list_reports(self, report_type: ReportType = None) -> List[Dict[str, Any]]:
        """レポート一覧取得"""
        try:
            reports = []

            for file_path in self.reports_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()

                    # ファイル名からレポートタイプ推定
                    if file_path.stem.startswith('portfolio'):
                        file_report_type = ReportType.PORTFOLIO_SUMMARY
                    elif file_path.stem.startswith('journal'):
                        file_report_type = ReportType.TRADING_JOURNAL
                    else:
                        file_report_type = ReportType.MARKET_OVERVIEW

                    if report_type is None or file_report_type == report_type:
                        reports.append({
                            'report_id': file_path.stem,
                            'report_type': file_report_type.value,
                            'file_path': str(file_path),
                            'file_size': stat.st_size,
                            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })

            return sorted(reports, key=lambda x: x['modified_at'], reverse=True)

        except Exception as e:
            self.logger.error(f"レポート一覧取得エラー: {e}")
            return []

    def get_report_content(self, report_id: str) -> Optional[str]:
        """レポート内容取得"""
        try:
            for file_path in self.reports_dir.glob(f"{report_id}.*"):
                if file_path.is_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()

            return None

        except Exception as e:
            self.logger.error(f"レポート内容取得エラー: {e}")
            return None