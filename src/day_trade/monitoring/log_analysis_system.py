#!/usr/bin/env python3
"""
統合ログ分析システム
Phase G: 本番運用最適化フェーズ

ログ収集・分析・パターン検出・異常検知システム
"""

import hashlib
import json
import re
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class LogLevel(Enum):
    """ログレベル"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AnomalyType(Enum):
    """異常タイプ"""

    ERROR_SPIKE = "error_spike"
    UNUSUAL_PATTERN = "unusual_pattern"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_THREAT = "security_threat"
    SYSTEM_FAILURE = "system_failure"


@dataclass
class LogEntry:
    """ログエントリ"""

    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    metadata: Dict[str, Any]
    raw_line: str
    line_number: int
    file_path: str


@dataclass
class LogPattern:
    """ログパターン"""

    pattern_id: str
    regex: str
    description: str
    severity: LogLevel
    count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    examples: List[str] = None


@dataclass
class Anomaly:
    """異常検知結果"""

    anomaly_id: str
    type: AnomalyType
    severity: LogLevel
    description: str
    timestamp: datetime
    affected_components: List[str]
    metrics: Dict[str, Any]
    evidence: List[str]


class LogParser:
    """ログパーサー"""

    def __init__(self):
        self.patterns = {
            # 一般的なログフォーマット
            "standard": re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]\d{3})"
                r"\s*-\s*(?P<component>\S+)\s*-\s*(?P<level>\w+)\s*-\s*"
                r"(?P<message>.*)"
            ),
            # Python logging フォーマット
            "python": re.compile(
                r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]\d{3})"
                r"\s+(?P<level>\w+)\s+(?P<component>\S+)\s+(?P<message>.*)"
            ),
            # JSON形式ログ
            "json": re.compile(r"^{.*}$"),
            # エラーパターン
            "error_traceback": re.compile(r"Traceback \(most recent call last\):"),
            "exception": re.compile(
                r"(?P<exception>\w*Error|Exception):\s*(?P<message>.*)"
            ),
        }

    def parse_log_line(
        self, line: str, line_number: int, file_path: str
    ) -> Optional[LogEntry]:
        """ログ行パース"""
        line = line.strip()
        if not line:
            return None

        # JSON形式チェック
        if self.patterns["json"].match(line):
            try:
                data = json.loads(line)
                return LogEntry(
                    timestamp=datetime.fromisoformat(data.get("timestamp", "")),
                    level=LogLevel(data.get("level", "INFO")),
                    component=data.get("component", "unknown"),
                    message=data.get("message", ""),
                    metadata=data,
                    raw_line=line,
                    line_number=line_number,
                    file_path=file_path,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # 標準フォーマットパース
        for pattern_name, pattern in [
            ("standard", self.patterns["standard"]),
            ("python", self.patterns["python"]),
        ]:
            match = pattern.match(line)
            if match:
                try:
                    timestamp_str = match.group("timestamp")
                    # タイムスタンプ形式の正規化
                    timestamp_str = timestamp_str.replace(",", ".")
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

                    level_str = match.group("level").upper()
                    level = (
                        LogLevel(level_str)
                        if level_str in [l.value for l in LogLevel]
                        else LogLevel.INFO
                    )

                    return LogEntry(
                        timestamp=timestamp,
                        level=level,
                        component=match.group("component"),
                        message=match.group("message"),
                        metadata={"parser": pattern_name},
                        raw_line=line,
                        line_number=line_number,
                        file_path=file_path,
                    )
                except (ValueError, KeyError):
                    continue

        # パース失敗時のフォールバック
        return LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            component="unknown",
            message=line,
            metadata={"parser": "fallback"},
            raw_line=line,
            line_number=line_number,
            file_path=file_path,
        )


class PatternDetector:
    """パターン検出器"""

    def __init__(self):
        self.known_patterns: Dict[str, LogPattern] = {}
        self.pattern_cache: Dict[str, str] = {}  # メッセージハッシュ -> パターンID
        self._initialize_default_patterns()

    def _initialize_default_patterns(self):
        """デフォルトパターン初期化"""
        default_patterns = [
            LogPattern(
                pattern_id="error_connection",
                regex=r".*connection.*(?:failed|error|timeout|refused).*",
                description="接続エラー",
                severity=LogLevel.ERROR,
            ),
            LogPattern(
                pattern_id="error_authentication",
                regex=r".*(?:auth|login|credential).*(?:failed|error|invalid).*",
                description="認証エラー",
                severity=LogLevel.ERROR,
            ),
            LogPattern(
                pattern_id="error_permission",
                regex=r".*(?:permission|access).*denied.*",
                description="権限エラー",
                severity=LogLevel.ERROR,
            ),
            LogPattern(
                pattern_id="error_memory",
                regex=r".*(?:memory|ram|heap).*(?:error|full|limit|exceeded).*",
                description="メモリエラー",
                severity=LogLevel.CRITICAL,
            ),
            LogPattern(
                pattern_id="error_disk",
                regex=r".*(?:disk|storage|space).*(?:full|error|failed).*",
                description="ディスクエラー",
                severity=LogLevel.CRITICAL,
            ),
            LogPattern(
                pattern_id="warning_performance",
                regex=r".*(?:slow|timeout|delay|latency).*",
                description="パフォーマンス警告",
                severity=LogLevel.WARNING,
            ),
            LogPattern(
                pattern_id="info_startup",
                regex=r".*(?:start|init|boot|launch).*(?:complete|success|ready).*",
                description="システム起動",
                severity=LogLevel.INFO,
            ),
            LogPattern(
                pattern_id="security_attempt",
                regex=r".*(?:suspicious|attack|intrusion|breach|hack).*",
                description="セキュリティ脅威",
                severity=LogLevel.CRITICAL,
            ),
        ]

        for pattern in default_patterns:
            pattern.examples = []
            self.known_patterns[pattern.pattern_id] = pattern

    def detect_patterns(self, log_entries: List[LogEntry]) -> List[LogPattern]:
        """パターン検出"""
        detected_patterns = []

        for entry in log_entries:
            message_lower = entry.message.lower()
            message_hash = hashlib.md5(message_lower.encode()).hexdigest()

            # キャッシュチェック
            if message_hash in self.pattern_cache:
                pattern_id = self.pattern_cache[message_hash]
                if pattern_id in self.known_patterns:
                    pattern = self.known_patterns[pattern_id]
                    pattern.count += 1
                    pattern.last_seen = entry.timestamp
                    continue

            # パターンマッチング
            matched = False
            for pattern in self.known_patterns.values():
                if re.search(pattern.regex, message_lower, re.IGNORECASE):
                    pattern.count += 1
                    if pattern.first_seen is None:
                        pattern.first_seen = entry.timestamp
                    pattern.last_seen = entry.timestamp

                    if len(pattern.examples) < 5:
                        pattern.examples.append(entry.message[:200])

                    self.pattern_cache[message_hash] = pattern.pattern_id
                    matched = True
                    break

            # 新パターン検出（頻度ベース）
            if not matched:
                self._check_new_pattern(entry, message_hash)

        return [p for p in self.known_patterns.values() if p.count > 0]

    def _check_new_pattern(self, entry: LogEntry, message_hash: str):
        """新パターンチェック"""
        # 簡易的な新パターン検出
        # 実際の実装では機械学習ベースのクラスタリングを使用
        words = entry.message.lower().split()
        if len(words) >= 3:
            # キーワードベースのパターン生成
            keywords = [w for w in words if len(w) > 3][:3]
            if keywords:
                pattern_id = (
                    f"auto_{hashlib.md5('_'.join(keywords).encode()).hexdigest()[:8]}"
                )

                if pattern_id not in self.known_patterns:
                    self.known_patterns[pattern_id] = LogPattern(
                        pattern_id=pattern_id,
                        regex=".*" + ".*".join(re.escape(k) for k in keywords) + ".*",
                        description=f"Auto-detected: {' '.join(keywords)}",
                        severity=entry.level,
                        count=1,
                        first_seen=entry.timestamp,
                        last_seen=entry.timestamp,
                        examples=[entry.message[:200]],
                    )

                    self.pattern_cache[message_hash] = pattern_id


class AnomalyDetector:
    """異常検知器"""

    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # 標準偏差の倍数

    def detect_anomalies(
        self, log_entries: List[LogEntry], patterns: List[LogPattern]
    ) -> List[Anomaly]:
        """異常検知"""
        anomalies = []

        # エラー急増検知
        error_spike = self._detect_error_spike(log_entries)
        if error_spike:
            anomalies.append(error_spike)

        # 異常パターン検知
        unusual_patterns = self._detect_unusual_patterns(patterns)
        anomalies.extend(unusual_patterns)

        # パフォーマンス劣化検知
        perf_degradation = self._detect_performance_degradation(log_entries)
        if perf_degradation:
            anomalies.append(perf_degradation)

        # セキュリティ脅威検知
        security_threats = self._detect_security_threats(log_entries, patterns)
        anomalies.extend(security_threats)

        return anomalies

    def _detect_error_spike(self, log_entries: List[LogEntry]) -> Optional[Anomaly]:
        """エラー急増検知"""
        # 時間窓ごとのエラー数をカウント
        window_minutes = 5
        error_counts = defaultdict(int)

        for entry in log_entries:
            if entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                time_bucket = entry.timestamp.replace(
                    minute=entry.timestamp.minute // window_minutes * window_minutes,
                    second=0,
                    microsecond=0,
                )
                error_counts[time_bucket] += 1

        if len(error_counts) < 2:
            return None

        counts = list(error_counts.values())
        avg_count = sum(counts) / len(counts)

        # 最新の窓でエラー数が平均の3倍以上
        latest_count = counts[-1]
        if latest_count > avg_count * 3 and latest_count > 10:
            return Anomaly(
                anomaly_id=f"error_spike_{int(time.time())}",
                type=AnomalyType.ERROR_SPIKE,
                severity=LogLevel.CRITICAL,
                description=f"エラー急増検知: {latest_count}件/5分 (平均: {avg_count:.1f}件)",
                timestamp=datetime.now(),
                affected_components=self._get_affected_components(
                    log_entries, LogLevel.ERROR
                ),
                metrics={"error_count": latest_count, "avg_count": avg_count},
                evidence=[
                    f"エラー数: {latest_count}",
                    f"平均エラー数: {avg_count:.1f}",
                ],
            )

        return None

    def _detect_unusual_patterns(self, patterns: List[LogPattern]) -> List[Anomaly]:
        """異常パターン検知"""
        anomalies = []

        for pattern in patterns:
            if pattern.count > 100:  # 頻出パターン
                if pattern.severity in [LogLevel.ERROR, LogLevel.CRITICAL]:
                    anomalies.append(
                        Anomaly(
                            anomaly_id=f"pattern_{pattern.pattern_id}_{int(time.time())}",
                            type=AnomalyType.UNUSUAL_PATTERN,
                            severity=pattern.severity,
                            description=f"異常パターン頻発: {pattern.description} ({pattern.count}回)",
                            timestamp=datetime.now(),
                            affected_components=[],
                            metrics={"pattern_count": pattern.count},
                            evidence=[
                                f"パターン: {pattern.description}",
                                f"発生回数: {pattern.count}",
                            ],
                        )
                    )

        return anomalies

    def _detect_performance_degradation(
        self, log_entries: List[LogEntry]
    ) -> Optional[Anomaly]:
        """パフォーマンス劣化検知"""
        perf_keywords = ["slow", "timeout", "delay", "latency", "performance"]
        perf_count = 0

        for entry in log_entries:
            if any(keyword in entry.message.lower() for keyword in perf_keywords):
                perf_count += 1

        # 総ログエントリの10%以上がパフォーマンス関連
        if len(log_entries) > 50 and perf_count / len(log_entries) > 0.1:
            return Anomaly(
                anomaly_id=f"perf_degradation_{int(time.time())}",
                type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=LogLevel.WARNING,
                description=f"パフォーマンス劣化検知: {perf_count}件のパフォーマンス関連ログ",
                timestamp=datetime.now(),
                affected_components=self._get_affected_components(log_entries),
                metrics={"perf_log_count": perf_count, "total_logs": len(log_entries)},
                evidence=[
                    f"パフォーマンス関連ログ: {perf_count}件",
                    f"全ログ: {len(log_entries)}件",
                ],
            )

        return None

    def _detect_security_threats(
        self, log_entries: List[LogEntry], patterns: List[LogPattern]
    ) -> List[Anomaly]:
        """セキュリティ脅威検知"""
        anomalies = []

        security_indicators = [
            "attack",
            "intrusion",
            "breach",
            "unauthorized",
            "suspicious",
            "malware",
            "virus",
            "hack",
            "exploit",
            "injection",
        ]

        security_logs = []
        for entry in log_entries:
            if any(
                indicator in entry.message.lower() for indicator in security_indicators
            ):
                security_logs.append(entry)

        if len(security_logs) > 5:  # セキュリティ関連ログが5件以上
            anomalies.append(
                Anomaly(
                    anomaly_id=f"security_threat_{int(time.time())}",
                    type=AnomalyType.SECURITY_THREAT,
                    severity=LogLevel.CRITICAL,
                    description=f"セキュリティ脅威検知: {len(security_logs)}件の疑わしい活動",
                    timestamp=datetime.now(),
                    affected_components=self._get_affected_components(security_logs),
                    metrics={"security_log_count": len(security_logs)},
                    evidence=[log.message[:100] for log in security_logs[:3]],
                )
            )

        return anomalies

    def _get_affected_components(
        self, log_entries: List[LogEntry], level_filter: LogLevel = None
    ) -> List[str]:
        """影響を受けるコンポーネント取得"""
        components = set()
        for entry in log_entries:
            if level_filter is None or entry.level == level_filter:
                components.add(entry.component)
        return list(components)[:10]  # 最大10個


class LogAnalysisSystem:
    """統合ログ分析システム"""

    def __init__(self, log_directories: List[str] = None):
        self.log_directories = log_directories or ["logs"]
        self.parser = LogParser()
        self.pattern_detector = PatternDetector()
        self.anomaly_detector = AnomalyDetector()

        self.analysis_history = deque(maxlen=100)
        self.processing_lock = threading.Lock()

        print("=" * 80)
        print("[LOG] 統合ログ分析システム")
        print("Phase G: 本番運用最適化フェーズ")
        print("=" * 80)

    def scan_log_files(self, max_files: int = 50) -> List[Path]:
        """ログファイルスキャン"""
        log_files = []

        for directory in self.log_directories:
            log_dir = Path(directory)
            if log_dir.exists():
                # .log, .txt ファイルを収集
                for pattern in ["*.log", "*.txt"]:
                    files = list(log_dir.rglob(pattern))
                    log_files.extend(files)

        # ファイルサイズでソート（大きいファイルを優先）
        log_files.sort(
            key=lambda f: f.stat().st_size if f.exists() else 0, reverse=True
        )

        return log_files[:max_files]

    def parse_log_file(self, file_path: Path, max_lines: int = 10000) -> List[LogEntry]:
        """ログファイル解析"""
        entries = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > max_lines:
                        break

                    entry = self.parser.parse_log_line(line, line_num, str(file_path))
                    if entry:
                        entries.append(entry)

        except Exception as e:
            print(f"ログファイル読み込みエラー {file_path}: {e}")

        return entries

    def analyze_logs(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """ログ分析実行"""
        with self.processing_lock:
            start_time = time.time()

            print(f"\n[ANALYZE] ログ分析開始 (時間窓: {time_window_hours}時間)")

            # ログファイルスキャン
            log_files = self.scan_log_files()
            print(f"[INFO] 発見ログファイル: {len(log_files)}個")

            if not log_files:
                return {"error": "ログファイルが見つかりません"}

            # ログエントリ収集
            all_entries = []
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

            for file_path in log_files:
                entries = self.parse_log_file(file_path)
                # 時間窓内のエントリのみ
                recent_entries = [e for e in entries if e.timestamp >= cutoff_time]
                all_entries.extend(recent_entries)

            print(f"[INFO] 解析ログエントリ: {len(all_entries)}個")

            if not all_entries:
                return {"error": "ログエントリが見つかりません"}

            # パターン検出
            patterns = self.pattern_detector.detect_patterns(all_entries)
            print(f"[INFO] 検出パターン: {len(patterns)}個")

            # 異常検知
            anomalies = self.anomaly_detector.detect_anomalies(all_entries, patterns)
            print(f"[INFO] 検出異常: {len(anomalies)}個")

            # 統計情報生成
            stats = self._generate_statistics(all_entries)

            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "time_window_hours": time_window_hours,
                "processed_files": len(log_files),
                "total_entries": len(all_entries),
                "statistics": stats,
                "patterns": [asdict(p) for p in patterns],
                "anomalies": [asdict(a) for a in anomalies],
                "execution_time": time.time() - start_time,
            }

            # 履歴保存
            self.analysis_history.append(analysis_result)

            print(
                f"[COMPLETE] ログ分析完了 ({analysis_result['execution_time']:.2f}秒)"
            )

            return analysis_result

    def _generate_statistics(self, log_entries: List[LogEntry]) -> Dict[str, Any]:
        """統計情報生成"""
        if not log_entries:
            return {}

        # レベル別統計
        level_counts = Counter(entry.level.value for entry in log_entries)

        # コンポーネント別統計
        component_counts = Counter(entry.component for entry in log_entries)

        # 時間別統計（1時間単位）
        hourly_counts = defaultdict(int)
        for entry in log_entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        # エラー率計算
        total_entries = len(log_entries)
        error_entries = sum(
            1 for e in log_entries if e.level in [LogLevel.ERROR, LogLevel.CRITICAL]
        )
        error_rate = (error_entries / total_entries) * 100 if total_entries > 0 else 0

        return {
            "total_entries": total_entries,
            "error_rate_percent": round(error_rate, 2),
            "level_distribution": dict(level_counts),
            "component_distribution": dict(component_counts.most_common(10)),
            "hourly_distribution": dict(hourly_counts),
            "time_range": {
                "start": min(entry.timestamp for entry in log_entries).isoformat(),
                "end": max(entry.timestamp for entry in log_entries).isoformat(),
            },
        }

    def get_analysis_report(self, format_type: str = "summary") -> str:
        """分析レポート取得"""
        if not self.analysis_history:
            return "分析履歴がありません。analyze_logs() を実行してください。"

        latest_analysis = self.analysis_history[-1]

        if format_type == "summary":
            return self._format_summary_report(latest_analysis)
        elif format_type == "detailed":
            return self._format_detailed_report(latest_analysis)
        else:
            return json.dumps(latest_analysis, indent=2, ensure_ascii=False)

    def _format_summary_report(self, analysis: Dict[str, Any]) -> str:
        """サマリーレポート形式"""
        stats = analysis.get("statistics", {})
        anomalies = analysis.get("anomalies", [])
        patterns = analysis.get("patterns", [])

        report = f"""
=== ログ分析サマリーレポート ===
分析時刻: {analysis['timestamp']}
時間窓: {analysis['time_window_hours']}時間
処理ファイル数: {analysis['processed_files']}個
総ログエントリ数: {analysis['total_entries']}個

=== 統計情報 ===
エラー率: {stats.get('error_rate_percent', 0)}%
レベル分布: {stats.get('level_distribution', {})}

=== 検出結果 ===
パターン数: {len(patterns)}個
異常検知数: {len(anomalies)}個

=== 異常一覧 ==="""

        for anomaly in anomalies:
            report += f"""
- {anomaly['type']}: {anomaly['description']} [{anomaly['severity']}]"""

        return report

    def _format_detailed_report(self, analysis: Dict[str, Any]) -> str:
        """詳細レポート形式"""
        summary = self._format_summary_report(analysis)

        patterns = analysis.get("patterns", [])
        anomalies = analysis.get("anomalies", [])

        detailed = summary + "\n\n=== パターン詳細 ==="

        for pattern in patterns[:10]:  # 上位10パターン
            detailed += f"""
パターンID: {pattern['pattern_id']}
説明: {pattern['description']}
重要度: {pattern['severity']}
発生回数: {pattern['count']}
初回検出: {pattern.get('first_seen', 'N/A')}
最終検出: {pattern.get('last_seen', 'N/A')}
"""

        detailed += "\n=== 異常詳細 ==="
        for anomaly in anomalies:
            detailed += f"""
異常ID: {anomaly['anomaly_id']}
タイプ: {anomaly['type']}
説明: {anomaly['description']}
重要度: {anomaly['severity']}
影響コンポーネント: {', '.join(anomaly.get('affected_components', []))}
証拠: {'; '.join(anomaly.get('evidence', []))}
"""

        return detailed

    def save_analysis_report(self, filename: str = None) -> str:
        """分析レポート保存"""
        if not self.analysis_history:
            return "保存する分析結果がありません"

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log_analysis_report_{timestamp}.json"

        latest_analysis = self.analysis_history[-1]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(latest_analysis, f, indent=2, ensure_ascii=False)

        return f"分析レポート保存完了: {filename}"


def main():
    """メイン実行"""
    log_analyzer = LogAnalysisSystem()

    try:
        # ログ分析実行
        result = log_analyzer.analyze_logs(time_window_hours=24)

        if "error" in result:
            print(f"[ERROR] {result['error']}")
            return

        # サマリーレポート表示
        print("\n" + "=" * 80)
        print(log_analyzer.get_analysis_report("summary"))
        print("=" * 80)

        # レポート保存
        saved_file = log_analyzer.save_analysis_report()
        print(f"\n[REPORT] {saved_file}")

    except KeyboardInterrupt:
        print("\n[STOP] ログ分析中断")
    except Exception as e:
        print(f"\n[ERROR] ログ分析エラー: {e}")


if __name__ == "__main__":
    main()
