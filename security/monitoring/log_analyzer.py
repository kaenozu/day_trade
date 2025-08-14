#!/usr/bin/env python3
"""
Advanced Log Analyzer - Day Trade ML System
高度なログ分析システム
"""

import re
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Pattern
from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LogPattern:
    """ログパターン定義"""
    name: str
    pattern: Pattern[str]
    severity: str
    description: str
    auto_response: bool = False


@dataclass
class LogAnomaly:
    """ログ異常検知結果"""
    timestamp: datetime
    source_file: str
    line_number: int
    content: str
    anomaly_type: str
    confidence_score: float
    details: Dict[str, any]


class LogPatternMatcher:
    """ログパターンマッチング"""

    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> List[LogPattern]:
        """セキュリティパターン初期化"""
        return [
            # 認証関連
            LogPattern(
                name="failed_login",
                pattern=re.compile(r"Failed login.*?(\d+\.\d+\.\d+\.\d+)", re.IGNORECASE),
                severity="medium",
                description="ログイン失敗"
            ),
            LogPattern(
                name="brute_force",
                pattern=re.compile(r"(\d+\.\d+\.\d+\.\d+).*failed.*login.*attempts.*(\d+)", re.IGNORECASE),
                severity="high",
                description="ブルートフォース攻撃",
                auto_response=True
            ),

            # Web攻撃
            LogPattern(
                name="sql_injection",
                pattern=re.compile(r"(union|select|insert|delete|drop|update).*?(from|into|table)", re.IGNORECASE),
                severity="critical",
                description="SQLインジェクション試行",
                auto_response=True
            ),
            LogPattern(
                name="xss_attempt",
                pattern=re.compile(r"<script|javascript:|onload=|onerror=", re.IGNORECASE),
                severity="high",
                description="XSS攻撃試行",
                auto_response=True
            ),
            LogPattern(
                name="directory_traversal",
                pattern=re.compile(r"\.\./|\.\.\\", re.IGNORECASE),
                severity="high",
                description="ディレクトリトラバーサル",
                auto_response=True
            ),

            # システム攻撃
            LogPattern(
                name="command_injection",
                pattern=re.compile(r"(\||;|&|\$\(|\`|>|<).*?(cat|ls|ps|netstat|whoami|id)", re.IGNORECASE),
                severity="critical",
                description="コマンドインジェクション",
                auto_response=True
            ),
            LogPattern(
                name="port_scan",
                pattern=re.compile(r"(\d+\.\d+\.\d+\.\d+).*?port.*?scan", re.IGNORECASE),
                severity="medium",
                description="ポートスキャン検知"
            ),

            # データ漏洩
            LogPattern(
                name="data_exfiltration",
                pattern=re.compile(r"(password|credit|card|ssn|api_key|token).*?(leak|exposed|dump)", re.IGNORECASE),
                severity="critical",
                description="データ漏洩の可能性",
                auto_response=True
            ),

            # 異常トラフィック
            LogPattern(
                name="high_frequency_requests",
                pattern=re.compile(r"(\d+\.\d+\.\d+\.\d+).*?(\d{3})\s+(\d+)", re.IGNORECASE),
                severity="medium",
                description="高頻度リクエスト"
            ),
            LogPattern(
                name="suspicious_user_agent",
                pattern=re.compile(r'User-Agent.*?(bot|crawler|scanner|exploit|hack)', re.IGNORECASE),
                severity="medium",
                description="疑わしいユーザーエージェント"
            ),

            # システムエラー
            LogPattern(
                name="system_error",
                pattern=re.compile(r"(error|exception|crash|segfault|core dump)", re.IGNORECASE),
                severity="medium",
                description="システムエラー"
            ),
            LogPattern(
                name="privilege_escalation",
                pattern=re.compile(r"(sudo|su|privilege|escalation|root)", re.IGNORECASE),
                severity="high",
                description="権限昇格の試行"
            )
        ]

    def match_patterns(self, log_line: str) -> List[Tuple[LogPattern, re.Match]]:
        """ログ行のパターンマッチング"""
        matches = []
        for pattern in self.patterns:
            match = pattern.pattern.search(log_line)
            if match:
                matches.append((pattern, match))
        return matches


class AnomalyDetector:
    """機械学習ベースの異常検知"""

    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def extract_features(self, log_entries: List[Dict]) -> pd.DataFrame:
        """ログエントリから特徴量抽出"""
        features = []

        for entry in log_entries:
            feature_dict = {
                'hour': entry.get('timestamp', datetime.now()).hour,
                'day_of_week': entry.get('timestamp', datetime.now()).weekday(),
                'response_code': self._extract_response_code(entry.get('message', '')),
                'request_size': len(entry.get('message', '')),
                'ip_frequency': self._get_ip_frequency(entry.get('source_ip', '')),
                'endpoint_frequency': self._get_endpoint_frequency(entry.get('endpoint', '')),
                'user_agent_length': len(entry.get('user_agent', '')),
                'response_time': entry.get('response_time', 0),
                'is_weekend': entry.get('timestamp', datetime.now()).weekday() >= 5,
                'error_keywords': self._count_error_keywords(entry.get('message', ''))
            }
            features.append(feature_dict)

        df = pd.DataFrame(features)
        self.feature_names = df.columns.tolist()
        return df

    def _extract_response_code(self, message: str) -> int:
        """レスポンスコード抽出"""
        match = re.search(r'\b([1-5]\d{2})\b', message)
        return int(match.group(1)) if match else 200

    def _get_ip_frequency(self, ip: str) -> int:
        """IP頻度取得（実装簡略化）"""
        # 実際の実装では、過去のログからIP頻度を計算
        return hash(ip) % 100

    def _get_endpoint_frequency(self, endpoint: str) -> int:
        """エンドポイント頻度取得（実装簡略化）"""
        # 実際の実装では、過去のログからエンドポイント頻度を計算
        return hash(endpoint) % 50

    def _count_error_keywords(self, message: str) -> int:
        """エラーキーワード数"""
        error_keywords = ['error', 'exception', 'fail', 'crash', 'timeout', 'unauthorized']
        return sum(1 for keyword in error_keywords if keyword.lower() in message.lower())

    def train(self, normal_log_entries: List[Dict]) -> None:
        """正常ログでの学習"""
        if len(normal_log_entries) < 10:
            logger.warning("Insufficient training data for anomaly detection")
            return

        features_df = self.extract_features(normal_log_entries)
        features_scaled = self.scaler.fit_transform(features_df)

        self.model.fit(features_scaled)
        self.is_trained = True

        logger.info(f"Anomaly detector trained with {len(normal_log_entries)} samples")

    def detect_anomalies(self, log_entries: List[Dict]) -> List[LogAnomaly]:
        """異常検知実行"""
        if not self.is_trained:
            logger.warning("Anomaly detector not trained yet")
            return []

        if not log_entries:
            return []

        features_df = self.extract_features(log_entries)
        features_scaled = self.scaler.transform(features_df)

        anomaly_scores = self.model.decision_function(features_scaled)
        predictions = self.model.predict(features_scaled)

        anomalies = []
        for i, (entry, score, prediction) in enumerate(zip(log_entries, anomaly_scores, predictions)):
            if prediction == -1:  # 異常として分類
                anomaly = LogAnomaly(
                    timestamp=entry.get('timestamp', datetime.now()),
                    source_file=entry.get('source_file', ''),
                    line_number=entry.get('line_number', 0),
                    content=entry.get('message', ''),
                    anomaly_type='statistical_anomaly',
                    confidence_score=abs(score),
                    details={
                        'features': dict(zip(self.feature_names, features_df.iloc[i])),
                        'anomaly_score': score
                    }
                )
                anomalies.append(anomaly)

        return anomalies


class LogAnalyzer:
    """統合ログ分析システム"""

    def __init__(self, log_directories: List[str]):
        self.log_directories = [Path(d) for d in log_directories]
        self.pattern_matcher = LogPatternMatcher()
        self.anomaly_detector = AnomalyDetector()
        self.analyzed_files = set()

        # ログ監視状態
        self.monitoring = False
        self.analysis_interval = 60  # 60秒間隔

    async def start_monitoring(self) -> None:
        """ログ監視開始"""
        self.monitoring = True
        logger.info("Log monitoring started")

        # 初期学習用の正常ログを収集
        await self._train_anomaly_detector()

        # 継続監視ループ
        while self.monitoring:
            try:
                await self._analyze_recent_logs()
                await asyncio.sleep(self.analysis_interval)
            except Exception as e:
                logger.error(f"Log monitoring error: {e}")
                await asyncio.sleep(self.analysis_interval * 2)

    def stop_monitoring(self) -> None:
        """ログ監視停止"""
        self.monitoring = False
        logger.info("Log monitoring stopped")

    async def _train_anomaly_detector(self) -> None:
        """異常検知モデルの学習"""
        logger.info("Training anomaly detection model...")

        # 過去24時間の正常ログを収集
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)

        normal_logs = []
        for log_dir in self.log_directories:
            logs = await self._parse_logs_in_timerange(log_dir, start_time, end_time)
            # パターンマッチで異常を除外
            filtered_logs = [log for log in logs if not self._has_security_patterns(log['message'])]
            normal_logs.extend(filtered_logs)

        if normal_logs:
            self.anomaly_detector.train(normal_logs)
        else:
            logger.warning("No training data found for anomaly detection")

    async def _analyze_recent_logs(self) -> None:
        """最近のログ分析"""
        logger.debug("Analyzing recent logs...")

        # 最後の分析以降の新しいログを取得
        cutoff_time = datetime.now() - timedelta(seconds=self.analysis_interval * 2)

        all_recent_logs = []
        for log_dir in self.log_directories:
            recent_logs = await self._parse_logs_in_timerange(log_dir, cutoff_time, datetime.now())
            all_recent_logs.extend(recent_logs)

        if not all_recent_logs:
            return

        # パターンマッチング分析
        pattern_alerts = await self._pattern_analysis(all_recent_logs)

        # 異常検知分析
        anomaly_alerts = self.anomaly_detector.detect_anomalies(all_recent_logs)

        # アラート処理
        await self._process_alerts(pattern_alerts + anomaly_alerts)

    async def _parse_logs_in_timerange(
        self,
        log_dir: Path,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """指定時間範囲のログ解析"""
        logs = []

        if not log_dir.exists():
            return logs

        # ログファイル検索
        log_files = list(log_dir.glob('*.log')) + list(log_dir.glob('*.log.*'))

        for log_file in log_files:
            try:
                file_logs = await self._parse_log_file(log_file, start_time, end_time)
                logs.extend(file_logs)
            except Exception as e:
                logger.error(f"Failed to parse {log_file}: {e}")

        return logs

    async def _parse_log_file(
        self,
        log_file: Path,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        """ログファイル解析"""
        logs = []
        line_number = 0

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue

                    # タイムスタンプ抽出
                    timestamp = self._extract_timestamp(line)
                    if timestamp and start_time <= timestamp <= end_time:
                        log_entry = {
                            'timestamp': timestamp,
                            'source_file': str(log_file),
                            'line_number': line_number,
                            'message': line,
                            'source_ip': self._extract_ip(line),
                            'endpoint': self._extract_endpoint(line),
                            'user_agent': self._extract_user_agent(line),
                            'response_time': self._extract_response_time(line)
                        }
                        logs.append(log_entry)

        except Exception as e:
            logger.error(f"Error parsing {log_file}: {e}")

        return logs

    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """タイムスタンプ抽出"""
        # 複数のタイムスタンプ形式に対応
        patterns = [
            r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})',
            r'(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})',
            r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    timestamp_str = match.group(1)
                    # 形式に応じて解析
                    if '-' in timestamp_str and ' ' in timestamp_str:
                        return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    elif '/' in timestamp_str and ':' in timestamp_str:
                        return datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S')
                except ValueError:
                    continue

        return None

    def _extract_ip(self, line: str) -> str:
        """IP アドレス抽出"""
        match = re.search(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', line)
        return match.group(1) if match else ''

    def _extract_endpoint(self, line: str) -> str:
        """エンドポイント抽出"""
        match = re.search(r'"[A-Z]+\s+([^\s"]+)', line)
        return match.group(1) if match else ''

    def _extract_user_agent(self, line: str) -> str:
        """ユーザーエージェント抽出"""
        match = re.search(r'User-Agent:\s*([^"]+)"', line, re.IGNORECASE)
        return match.group(1) if match else ''

    def _extract_response_time(self, line: str) -> float:
        """レスポンス時間抽出"""
        match = re.search(r'(\d+\.?\d*)\s*ms', line)
        return float(match.group(1)) if match else 0.0

    def _has_security_patterns(self, message: str) -> bool:
        """セキュリティパターンの有無確認"""
        matches = self.pattern_matcher.match_patterns(message)
        return len(matches) > 0

    async def _pattern_analysis(self, logs: List[Dict]) -> List[LogAnomaly]:
        """パターンマッチング分析"""
        alerts = []

        for log_entry in logs:
            matches = self.pattern_matcher.match_patterns(log_entry['message'])

            for pattern, match in matches:
                alert = LogAnomaly(
                    timestamp=log_entry['timestamp'],
                    source_file=log_entry['source_file'],
                    line_number=log_entry['line_number'],
                    content=log_entry['message'],
                    anomaly_type=f'pattern_match_{pattern.name}',
                    confidence_score=1.0,  # パターンマッチは確実
                    details={
                        'pattern_name': pattern.name,
                        'severity': pattern.severity,
                        'description': pattern.description,
                        'matched_groups': match.groups(),
                        'auto_response': pattern.auto_response
                    }
                )
                alerts.append(alert)

        return alerts

    async def _process_alerts(self, alerts: List[LogAnomaly]) -> None:
        """アラート処理"""
        if not alerts:
            return

        logger.info(f"Processing {len(alerts)} security alerts")

        for alert in alerts:
            # アラート記録
            logger.warning(f"SECURITY ALERT: {alert.anomaly_type} - {alert.content[:100]}...")

            # 自動対応が必要な場合
            if (alert.details.get('auto_response', False) and
                alert.details.get('severity') in ['critical', 'high']):
                await self._execute_auto_response(alert)

            # アラート保存
            await self._save_alert(alert)

    async def _execute_auto_response(self, alert: LogAnomaly) -> None:
        """自動対応実行"""
        logger.info(f"Executing auto-response for: {alert.anomaly_type}")

        # IP ブロック、セッション無効化等の自動対応
        # 実装は環境に応じて調整

        if 'brute_force' in alert.anomaly_type:
            # ブルートフォース攻撃の場合：IP ブロック
            source_ip = alert.details.get('source_ip')
            if source_ip:
                logger.info(f"Auto-blocking IP: {source_ip}")
                # await self._block_ip(source_ip)

        elif 'sql_injection' in alert.anomaly_type:
            # SQLインジェクションの場合：WAF ルール更新
            logger.info("Updating WAF rules for SQL injection")
            # await self._update_waf_rules(alert.content)

    async def _save_alert(self, alert: LogAnomaly) -> None:
        """アラート保存"""
        try:
            # JSON形式でアラートを保存
            alert_data = {
                'timestamp': alert.timestamp.isoformat(),
                'source_file': alert.source_file,
                'line_number': alert.line_number,
                'content': alert.content,
                'anomaly_type': alert.anomaly_type,
                'confidence_score': alert.confidence_score,
                'details': alert.details
            }

            # 保存先ディレクトリ
            alerts_dir = Path('/var/log/security_alerts')
            alerts_dir.mkdir(exist_ok=True)

            # 日付別ファイル
            alert_file = alerts_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"

            with open(alert_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert_data) + '\n')

        except Exception as e:
            logger.error(f"Failed to save alert: {e}")


# 使用例
async def main():
    """ログ分析システムのメイン関数"""

    # 監視対象ログディレクトリ
    log_directories = [
        '/var/log/nginx',
        '/var/log/apache2',
        '/var/log/day_trade',
        '/var/log/auth.log'
    ]

    # ログ分析システム初期化
    analyzer = LogAnalyzer(log_directories)

    try:
        # 監視開始
        await analyzer.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Shutting down log analyzer...")
        analyzer.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())