# Automated SLA Reporting System
# Day Trade ML System - Issue #802

import asyncio
import aiohttp
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from pathlib import Path
import jinja2


@dataclass
class SLIMetric:
    """SLI測定結果"""
    name: str
    value: float
    target: float
    unit: str
    timestamp: datetime
    status: str  # "ok", "warning", "critical"

    @property
    def compliance_percentage(self) -> float:
        """コンプライアンス率計算"""
        if self.unit == "percent":
            return min(100.0, (self.value / self.target) * 100)
        elif "latency" in self.name.lower() or "duration" in self.name.lower():
            # レイテンシの場合は逆転（低いほど良い）
            return min(100.0, (self.target / max(self.value, 0.01)) * 100)
        else:
            return min(100.0, (self.value / self.target) * 100)


@dataclass
class SLAReport:
    """SLAレポート"""
    period_start: datetime
    period_end: datetime
    report_type: str  # "daily", "weekly", "monthly", "quarterly"
    service_name: str
    sli_metrics: List[SLIMetric]
    overall_compliance: float
    error_budget_consumed: float
    incidents: List[Dict]
    recommendations: List[str]
    generated_at: datetime


class PrometheusClient:
    """Prometheus APIクライアント"""

    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url.rstrip('/')
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def query_range(self, query: str, start: datetime, end: datetime, step: str = "5m") -> Dict:
        """範囲クエリ実行"""
        params = {
            'query': query,
            'start': start.timestamp(),
            'end': end.timestamp(),
            'step': step
        }

        async with self.session.get(f"{self.prometheus_url}/api/v1/query_range", params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def query(self, query: str) -> Dict:
        """瞬時クエリ実行"""
        params = {'query': query}

        async with self.session.get(f"{self.prometheus_url}/api/v1/query", params=params) as response:
            response.raise_for_status()
            return await response.json()


class SLACalculator:
    """SLA計算エンジン"""

    def __init__(self, prometheus_client: PrometheusClient, slo_config: Dict):
        self.prometheus = prometheus_client
        self.slo_config = slo_config
        self.logger = logging.getLogger(__name__)

    async def calculate_sli_metrics(self, service: str, start: datetime, end: datetime) -> List[SLIMetric]:
        """SLIメトリクス計算"""
        metrics = []
        service_config = self.slo_config['services'].get(service, {})
        objectives = service_config.get('objectives', {})

        for objective_name, objective_config in objectives.items():
            try:
                query = objective_config['sli_query']
                target = objective_config['target']

                # Prometheusクエリ実行
                result = await self.prometheus.query_range(query, start, end)

                if result['status'] == 'success' and result['data']['result']:
                    values = []
                    for series in result['data']['result']:
                        for timestamp, value in series['values']:
                            try:
                                values.append(float(value))
                            except (ValueError, TypeError):
                                continue

                    if values:
                        # 平均値を計算（目的に応じて変更可能）
                        avg_value = np.mean(values)
                        min_value = np.min(values)
                        max_value = np.max(values)

                        # ステータス判定
                        alerting_threshold = objective_config.get('alerting_threshold', target * 0.95)

                        if "error_rate" in objective_name or "latency" in objective_name:
                            # エラー率・レイテンシは低いほど良い
                            if avg_value <= alerting_threshold:
                                status = "ok"
                            elif avg_value <= target:
                                status = "warning"
                            else:
                                status = "critical"
                        else:
                            # 可用性・精度は高いほど良い
                            if avg_value >= target:
                                status = "ok"
                            elif avg_value >= alerting_threshold:
                                status = "warning"
                            else:
                                status = "critical"

                        metric = SLIMetric(
                            name=f"{service}_{objective_name}",
                            value=avg_value,
                            target=target,
                            unit=self._get_unit(objective_name),
                            timestamp=end,
                            status=status
                        )
                        metrics.append(metric)

                        self.logger.info(f"Calculated SLI: {metric.name} = {metric.value:.2f} (target: {metric.target})")

            except Exception as e:
                self.logger.error(f"Error calculating SLI {objective_name} for {service}: {e}")

        return metrics

    def _get_unit(self, objective_name: str) -> str:
        """メトリクス単位推定"""
        if "rate" in objective_name or "accuracy" in objective_name or "availability" in objective_name:
            return "percent"
        elif "latency" in objective_name or "duration" in objective_name:
            return "seconds"
        elif "throughput" in objective_name:
            return "requests/second"
        else:
            return "count"

    async def calculate_error_budget(self, service: str, start: datetime, end: datetime) -> float:
        """エラーバジェット消費率計算"""
        try:
            service_config = self.slo_config['services'].get(service, {})
            tier = service_config.get('tier', 'standard')
            tier_config = self.slo_config['service_tiers'][tier]

            # 可用性ベースのエラーバジェット計算
            availability_target = tier_config['availability']
            error_budget_percentage = 100 - availability_target  # 例: 99.9% -> 0.1%

            # 実際の可用性取得
            availability_query = f"avg_over_time(sli:availability:{service.replace('-', '_')}:5m[{self._format_duration(end - start)}])"
            result = await self.prometheus.query(availability_query)

            if result['status'] == 'success' and result['data']['result']:
                actual_availability = float(result['data']['result'][0]['value'][1])
                actual_error_rate = 100 - actual_availability

                # エラーバジェット消費率
                consumed_percentage = (actual_error_rate / error_budget_percentage) * 100
                return min(100.0, consumed_percentage)

        except Exception as e:
            self.logger.error(f"Error calculating error budget for {service}: {e}")

        return 0.0

    def _format_duration(self, delta: timedelta) -> str:
        """期間をPrometheus形式に変換"""
        total_seconds = int(delta.total_seconds())

        if total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h"
        else:
            return f"{total_seconds // 86400}d"


class IncidentDetector:
    """インシデント検出"""

    def __init__(self, prometheus_client: PrometheusClient):
        self.prometheus = prometheus_client
        self.logger = logging.getLogger(__name__)

    async def detect_incidents(self, start: datetime, end: datetime) -> List[Dict]:
        """インシデント検出"""
        incidents = []

        # アラート履歴から重要インシデントを抽出
        alert_queries = [
            'ALERTS{severity="critical"}',
            'ALERTS{severity="high"}',
            'ALERTS{alert_type="business_critical"}'
        ]

        for query in alert_queries:
            try:
                result = await self.prometheus.query_range(query, start, end)

                if result['status'] == 'success':
                    for series in result['data']['result']:
                        labels = series['metric']

                        # アラート期間の計算
                        alert_periods = []
                        current_period = None

                        for timestamp, value in series['values']:
                            ts = datetime.fromtimestamp(float(timestamp))

                            if float(value) == 1:  # アラート発火中
                                if current_period is None:
                                    current_period = {'start': ts, 'end': ts}
                                else:
                                    current_period['end'] = ts
                            else:  # アラート解決
                                if current_period:
                                    alert_periods.append(current_period)
                                    current_period = None

                        # 未解決アラートの処理
                        if current_period:
                            current_period['end'] = end
                            alert_periods.append(current_period)

                        # インシデント記録
                        for period in alert_periods:
                            duration = period['end'] - period['start']

                            incident = {
                                'alert_name': labels.get('alertname', 'Unknown'),
                                'service': labels.get('service', 'Unknown'),
                                'severity': labels.get('severity', 'unknown'),
                                'start_time': period['start'],
                                'end_time': period['end'],
                                'duration_minutes': int(duration.total_seconds() / 60),
                                'labels': labels
                            }
                            incidents.append(incident)

            except Exception as e:
                self.logger.error(f"Error detecting incidents with query {query}: {e}")

        # 重複除去と重要度順ソート
        unique_incidents = []
        seen = set()

        for incident in incidents:
            key = (incident['alert_name'], incident['service'], incident['start_time'])
            if key not in seen:
                seen.add(key)
                unique_incidents.append(incident)

        # 重要度とインパクトでソート
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        unique_incidents.sort(
            key=lambda x: (
                severity_order.get(x['severity'], 999),
                -x['duration_minutes']
            )
        )

        return unique_incidents[:10]  # 上位10件


class RecommendationEngine:
    """改善推奨エンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_recommendations(self, sli_metrics: List[SLIMetric], incidents: List[Dict], error_budget_consumed: float) -> List[str]:
        """改善推奨生成"""
        recommendations = []

        # SLI違反ベースの推奨
        critical_metrics = [m for m in sli_metrics if m.status == "critical"]
        warning_metrics = [m for m in sli_metrics if m.status == "warning"]

        if critical_metrics:
            recommendations.append(
                f"🚨 {len(critical_metrics)}個の重要SLI違反が検出されました。即座の対応が必要です。"
            )

            for metric in critical_metrics[:3]:  # 上位3件
                if "accuracy" in metric.name:
                    recommendations.append(
                        f"• ML精度が{metric.value:.1f}%まで低下。モデル再訓練またはロールバックを検討してください。"
                    )
                elif "availability" in metric.name:
                    recommendations.append(
                        f"• {metric.name}の可用性が{metric.value:.2f}%。インフラ調査とスケーリング対応が必要です。"
                    )
                elif "latency" in metric.name:
                    recommendations.append(
                        f"• {metric.name}のレイテンシが{metric.value:.1f}秒。パフォーマンス最適化が必要です。"
                    )

        # エラーバジェット消費ベースの推奨
        if error_budget_consumed > 75:
            recommendations.append(
                f"⚠️ エラーバジェットの{error_budget_consumed:.1f}%を消費。新機能リリースを一時停止し、安定性を優先してください。"
            )
        elif error_budget_consumed > 50:
            recommendations.append(
                f"📊 エラーバジェットの{error_budget_consumed:.1f}%を消費。慎重なリリース管理が推奨されます。"
            )

        # インシデントパターンベースの推奨
        if incidents:
            frequent_services = {}
            for incident in incidents:
                service = incident['service']
                frequent_services[service] = frequent_services.get(service, 0) + 1

            most_frequent = max(frequent_services.items(), key=lambda x: x[1])
            if most_frequent[1] >= 3:
                recommendations.append(
                    f"🔄 {most_frequent[0]}で{most_frequent[1]}回のインシデント発生。根本原因分析が必要です。"
                )

        # パフォーマンス最適化推奨
        latency_metrics = [m for m in sli_metrics if "latency" in m.name and m.compliance_percentage < 95]
        if latency_metrics:
            recommendations.append(
                "⚡ レイテンシ最適化の機会: キャッシュ戦略見直し、データベース最適化、CDN活用を検討してください。"
            )

        # 予防的推奨
        if not recommendations:
            recommendations.append("✅ 全てのSLOが達成されています。継続的な監視とさらなる最適化の機会を探索してください。")

        return recommendations


class SLAReportGenerator:
    """SLAレポート生成器"""

    def __init__(self, prometheus_url: str, config_path: str):
        self.prometheus_url = prometheus_url
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)

        # Jinja2テンプレート設定
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        # SLO設定読み込み
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.slo_config = yaml.safe_load(f)

    async def generate_report(self, report_type: str = "daily", service: str = "all") -> SLAReport:
        """SLAレポート生成"""
        end_time = datetime.now()

        if report_type == "daily":
            start_time = end_time - timedelta(days=1)
        elif report_type == "weekly":
            start_time = end_time - timedelta(weeks=1)
        elif report_type == "monthly":
            start_time = end_time - timedelta(days=30)
        elif report_type == "quarterly":
            start_time = end_time - timedelta(days=90)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")

        async with PrometheusClient(self.prometheus_url) as prometheus:
            calculator = SLACalculator(prometheus, self.slo_config)
            incident_detector = IncidentDetector(prometheus)
            recommendation_engine = RecommendationEngine()

            # サービス一覧取得
            if service == "all":
                services = list(self.slo_config['services'].keys())
            else:
                services = [service]

            all_metrics = []
            total_error_budget = 0.0

            # 各サービスのSLI計算
            for svc in services:
                metrics = await calculator.calculate_sli_metrics(svc, start_time, end_time)
                all_metrics.extend(metrics)

                error_budget = await calculator.calculate_error_budget(svc, start_time, end_time)
                total_error_budget += error_budget

            # 平均エラーバジェット消費率
            avg_error_budget = total_error_budget / len(services) if services else 0.0

            # 全体コンプライアンス計算
            if all_metrics:
                compliance_values = [m.compliance_percentage for m in all_metrics]
                overall_compliance = np.mean(compliance_values)
            else:
                overall_compliance = 0.0

            # インシデント検出
            incidents = await incident_detector.detect_incidents(start_time, end_time)

            # 推奨事項生成
            recommendations = recommendation_engine.generate_recommendations(
                all_metrics, incidents, avg_error_budget
            )

            report = SLAReport(
                period_start=start_time,
                period_end=end_time,
                report_type=report_type,
                service_name=service,
                sli_metrics=all_metrics,
                overall_compliance=overall_compliance,
                error_budget_consumed=avg_error_budget,
                incidents=incidents,
                recommendations=recommendations,
                generated_at=datetime.now()
            )

            self.logger.info(f"Generated {report_type} SLA report for {service}")
            return report

    def export_to_html(self, report: SLAReport, output_path: Path) -> None:
        """HTMLレポート出力"""
        template = self.jinja_env.get_template('sla_report.html')

        html_content = template.render(
            report=report,
            format_datetime=lambda dt: dt.strftime('%Y-%m-%d %H:%M:%S'),
            format_duration=lambda minutes: f"{minutes//60}h {minutes%60}m" if minutes > 60 else f"{minutes}m"
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"HTML report exported to {output_path}")

    def export_to_json(self, report: SLAReport, output_path: Path) -> None:
        """JSONレポート出力"""
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        report_dict = asdict(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, default=serialize_datetime, indent=2, ensure_ascii=False)

        self.logger.info(f"JSON report exported to {output_path}")

    async def send_email_report(self, report: SLAReport, recipients: List[str], smtp_config: Dict) -> None:
        """メールレポート送信"""
        try:
            # HTMLレポート生成
            temp_html_path = Path("/tmp/sla_report.html")
            self.export_to_html(report, temp_html_path)

            # メール構成
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"SLA Report - {report.report_type.title()} ({report.period_start.strftime('%Y-%m-%d')})"

            # メール本文
            body = f"""
Day Trade ML System SLA Report

Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}
Overall Compliance: {report.overall_compliance:.2f}%
Error Budget Consumed: {report.error_budget_consumed:.1f}%

Key Metrics:
{chr(10).join([f"• {m.name}: {m.value:.2f} {m.unit} (target: {m.target}) - {m.status.upper()}" for m in report.sli_metrics[:5]])}

Recommendations:
{chr(10).join([f"• {rec}" for rec in report.recommendations[:3]])}

Full report attached.
            """

            msg.attach(MIMEText(body, 'plain'))

            # HTMLファイル添付
            with open(temp_html_path, 'rb') as f:
                attachment = MIMEApplication(f.read(), _subtype='html')
                attachment.add_header('Content-Disposition', 'attachment', filename='sla_report.html')
                msg.attach(attachment)

            # SMTP送信
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])

                server.send_message(msg)

            self.logger.info(f"Email report sent to {recipients}")

            # 一時ファイル削除
            temp_html_path.unlink()

        except Exception as e:
            self.logger.error(f"Failed to send email report: {e}")


class ScheduledReporter:
    """スケジュール化レポーター"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)

        self.report_generator = SLAReportGenerator(
            self.config['prometheus_url'],
            self.config['slo_config_path']
        )

    def _load_config(self, config_path: str) -> Dict:
        """設定読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    async def run_daily_reports(self):
        """日次レポート実行"""
        self.logger.info("Running daily SLA reports...")

        for service in self.config['services']:
            try:
                report = await self.report_generator.generate_report("daily", service)

                # ファイル出力
                output_dir = Path(self.config['output_directory']) / "daily"
                output_dir.mkdir(parents=True, exist_ok=True)

                date_str = report.period_start.strftime('%Y%m%d')
                html_path = output_dir / f"sla_report_{service}_{date_str}.html"
                json_path = output_dir / f"sla_report_{service}_{date_str}.json"

                self.report_generator.export_to_html(report, html_path)
                self.report_generator.export_to_json(report, json_path)

                # 重要度に応じてメール送信
                if report.overall_compliance < 95 or report.error_budget_consumed > 50:
                    await self.report_generator.send_email_report(
                        report,
                        self.config['alert_recipients'],
                        self.config['smtp']
                    )

            except Exception as e:
                self.logger.error(f"Failed to generate daily report for {service}: {e}")

    async def run_weekly_reports(self):
        """週次レポート実行"""
        self.logger.info("Running weekly SLA reports...")

        report = await self.report_generator.generate_report("weekly", "all")

        # ファイル出力
        output_dir = Path(self.config['output_directory']) / "weekly"
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = report.period_start.strftime('%Y%W')
        html_path = output_dir / f"sla_report_weekly_{date_str}.html"
        json_path = output_dir / f"sla_report_weekly_{date_str}.json"

        self.report_generator.export_to_html(report, html_path)
        self.report_generator.export_to_json(report, json_path)

        # 週次レポートは常にメール送信
        await self.report_generator.send_email_report(
            report,
            self.config['management_recipients'],
            self.config['smtp']
        )


async def main():
    """メイン実行関数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 設定ファイルパス
    config_path = "/app/config/sla_reporter_config.yaml"

    # レポーター初期化
    reporter = ScheduledReporter(config_path)

    # 日次レポート実行
    await reporter.run_daily_reports()

    # 週次レポート実行（月曜日のみ）
    if datetime.now().weekday() == 0:  # Monday
        await reporter.run_weekly_reports()


if __name__ == "__main__":
    asyncio.run(main())