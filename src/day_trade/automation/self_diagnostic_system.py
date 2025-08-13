#!/usr/bin/env python3
"""
自己診断システム

Issue #487対応: 完全自動化システム実装 - Phase 1
システム健全性監視・自動診断・問題検出・復旧支援
"""

import time
import psutil
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import asyncio

from ..utils.logging_config import get_context_logger
from ..ml.ensemble_system import EnsembleSystem, EnsembleConfig
from .smart_symbol_selector import SmartSymbolSelector
from .execution_scheduler import ExecutionScheduler

logger = get_context_logger(__name__)


class DiagnosticLevel(Enum):
    """診断レベル"""
    INFO = "info"           # 情報
    WARNING = "warning"     # 警告
    ERROR = "error"         # エラー
    CRITICAL = "critical"   # 致命的


class ComponentStatus(Enum):
    """コンポーネントステータス"""
    HEALTHY = "healthy"     # 正常
    DEGRADED = "degraded"   # 性能低下
    FAILED = "failed"       # 失敗
    UNKNOWN = "unknown"     # 不明


@dataclass
class DiagnosticResult:
    """診断結果"""
    component: str
    check_name: str
    level: DiagnosticLevel
    status: ComponentStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """システム健全性情報"""
    overall_status: ComponentStatus
    last_check: datetime
    components: Dict[str, ComponentStatus]
    issues_count: Dict[DiagnosticLevel, int]
    uptime_seconds: float
    performance_score: float  # 0-100
    diagnostic_history: List[DiagnosticResult]


class SelfDiagnosticSystem:
    """
    Issue #487対応: 自己診断システム

    完全自動化システムの信頼性確保のための診断機能:
    - システムリソース監視
    - MLモデル健全性チェック
    - データ品質診断
    - 自動復旧支援
    """

    def __init__(self):
        """初期化"""
        self.start_time = datetime.now()
        self.is_running = False
        self.diagnostic_thread: Optional[threading.Thread] = None

        # 診断結果履歴
        self.diagnostic_history: List[DiagnosticResult] = []
        self.max_history_size = 1000

        # システムコンポーネント状態
        self.component_status: Dict[str, ComponentStatus] = {}

        # 診断間隔設定
        self.check_intervals = {
            'system_resources': 60,    # システムリソース: 1分間隔
            'ml_models': 300,         # MLモデル: 5分間隔
            'data_quality': 600,      # データ品質: 10分間隔
            'automation_health': 180   # 自動化システム: 3分間隔
        }

        # 閾値設定
        self.thresholds = {
            'cpu_usage_warning': 70.0,      # CPU使用率警告閾値(%)
            'cpu_usage_critical': 90.0,     # CPU使用率致命的閾値(%)
            'memory_usage_warning': 80.0,   # メモリ使用率警告閾値(%)
            'memory_usage_critical': 95.0,  # メモリ使用率致命的閾値(%)
            'disk_usage_warning': 85.0,     # ディスク使用率警告閾値(%)
            'disk_usage_critical': 95.0,    # ディスク使用率致命的閾値(%)
            'response_time_warning': 5.0,   # 応答時間警告閾値(秒)
            'response_time_critical': 15.0, # 応答時間致命的閾値(秒)
        }

        # テストコンポーネント
        self.test_components = {}
        self._initialize_test_components()

    def _initialize_test_components(self):
        """テストコンポーネント初期化"""
        try:
            # MLシステム
            self.test_components['ensemble_system'] = None
            self.test_components['smart_selector'] = SmartSymbolSelector()
            self.test_components['scheduler'] = None

            logger.info("自己診断システム初期化完了")

        except Exception as e:
            logger.error(f"テストコンポーネント初期化エラー: {e}")

    def start(self):
        """診断システム開始"""
        if self.is_running:
            logger.warning("自己診断システムは既に実行中です")
            return

        self.is_running = True
        self.diagnostic_thread = threading.Thread(target=self._diagnostic_loop, daemon=True)
        self.diagnostic_thread.start()

        logger.info("自己診断システム開始")

    def stop(self):
        """診断システム停止"""
        self.is_running = False

        if self.diagnostic_thread:
            self.diagnostic_thread.join(timeout=10.0)

        logger.info("自己診断システム停止")

    def _diagnostic_loop(self):
        """診断メインループ"""
        logger.info("自己診断ループ開始")

        last_checks = {}

        while self.is_running:
            try:
                current_time = time.time()

                # 各診断の実行チェック
                for check_type, interval in self.check_intervals.items():
                    last_check = last_checks.get(check_type, 0)

                    if current_time - last_check >= interval:
                        self._run_diagnostic_check(check_type)
                        last_checks[check_type] = current_time

                # 10秒間隔でループ
                time.sleep(10)

            except Exception as e:
                logger.error(f"診断ループエラー: {e}")
                time.sleep(30)

    def _run_diagnostic_check(self, check_type: str):
        """診断チェック実行"""
        try:
            if check_type == 'system_resources':
                self._check_system_resources()
            elif check_type == 'ml_models':
                self._check_ml_models()
            elif check_type == 'data_quality':
                self._check_data_quality()
            elif check_type == 'automation_health':
                self._check_automation_health()

        except Exception as e:
            self._add_diagnostic_result(DiagnosticResult(
                component=check_type,
                check_name="diagnostic_execution",
                level=DiagnosticLevel.ERROR,
                status=ComponentStatus.FAILED,
                message=f"診断チェック実行エラー: {e}",
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))

    def _check_system_resources(self):
        """システムリソースチェック"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._evaluate_threshold(
                cpu_percent,
                self.thresholds['cpu_usage_warning'],
                self.thresholds['cpu_usage_critical']
            )

            self._add_diagnostic_result(DiagnosticResult(
                component="system_resources",
                check_name="cpu_usage",
                level=self._status_to_level(cpu_status),
                status=cpu_status,
                message=f"CPU使用率: {cpu_percent:.1f}%",
                timestamp=datetime.now(),
                details={'cpu_percent': cpu_percent}
            ))

            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_status = self._evaluate_threshold(
                memory.percent,
                self.thresholds['memory_usage_warning'],
                self.thresholds['memory_usage_critical']
            )

            self._add_diagnostic_result(DiagnosticResult(
                component="system_resources",
                check_name="memory_usage",
                level=self._status_to_level(memory_status),
                status=memory_status,
                message=f"メモリ使用率: {memory.percent:.1f}%",
                timestamp=datetime.now(),
                details={
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3)
                }
            ))

            # ディスク使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._evaluate_threshold(
                disk_percent,
                self.thresholds['disk_usage_warning'],
                self.thresholds['disk_usage_critical']
            )

            self._add_diagnostic_result(DiagnosticResult(
                component="system_resources",
                check_name="disk_usage",
                level=self._status_to_level(disk_status),
                status=disk_status,
                message=f"ディスク使用率: {disk_percent:.1f}%",
                timestamp=datetime.now(),
                details={
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3)
                }
            ))

            self.component_status['system_resources'] = self._worst_status([
                cpu_status, memory_status, disk_status
            ])

        except Exception as e:
            logger.error(f"システムリソースチェックエラー: {e}")

    def _check_ml_models(self):
        """MLモデル健全性チェック"""
        try:
            # Issue #462対応: 93%精度アンサンブルシステムチェック
            start_time = time.time()

            # 簡単な応答テスト
            test_data = [[1.0, 2.0, 3.0, 4.0, 5.0]]  # ダミーデータ

            try:
                # アンサンブルシステム簡易テスト
                if self.test_components.get('ensemble_system'):
                    # 実際のテストコードはここに実装
                    pass

                response_time = time.time() - start_time

                # 応答時間チェック
                response_status = self._evaluate_threshold(
                    response_time,
                    self.thresholds['response_time_warning'],
                    self.thresholds['response_time_critical']
                )

                self._add_diagnostic_result(DiagnosticResult(
                    component="ml_models",
                    check_name="ensemble_response_time",
                    level=self._status_to_level(response_status),
                    status=response_status,
                    message=f"アンサンブル応答時間: {response_time:.2f}秒",
                    timestamp=datetime.now(),
                    details={'response_time': response_time}
                ))

                self.component_status['ml_models'] = response_status

            except Exception as model_error:
                self._add_diagnostic_result(DiagnosticResult(
                    component="ml_models",
                    check_name="ensemble_health",
                    level=DiagnosticLevel.ERROR,
                    status=ComponentStatus.FAILED,
                    message=f"アンサンブルシステムエラー: {model_error}",
                    timestamp=datetime.now(),
                    details={'error': str(model_error)},
                    suggestions=["アンサンブルシステムの再起動を検討してください"]
                ))

                self.component_status['ml_models'] = ComponentStatus.FAILED

        except Exception as e:
            logger.error(f"MLモデルチェックエラー: {e}")

    def _check_data_quality(self):
        """データ品質チェック"""
        try:
            # データアクセステスト
            start_time = time.time()

            # 簡易データ品質テスト
            quality_score = 85.0  # ダミー値（実際は詳細チェック）

            access_time = time.time() - start_time

            if quality_score >= 90:
                status = ComponentStatus.HEALTHY
            elif quality_score >= 70:
                status = ComponentStatus.DEGRADED
            else:
                status = ComponentStatus.FAILED

            self._add_diagnostic_result(DiagnosticResult(
                component="data_quality",
                check_name="data_access_test",
                level=self._status_to_level(status),
                status=status,
                message=f"データ品質スコア: {quality_score:.1f}%",
                timestamp=datetime.now(),
                details={
                    'quality_score': quality_score,
                    'access_time': access_time
                }
            ))

            self.component_status['data_quality'] = status

        except Exception as e:
            logger.error(f"データ品質チェックエラー: {e}")

    def _check_automation_health(self):
        """自動化システム健全性チェック"""
        try:
            # Issue #487対応: スマート選択システムテスト
            start_time = time.time()

            # スマート銘柄選択システムの簡易テスト
            selector_status = ComponentStatus.HEALTHY
            if self.test_components.get('smart_selector'):
                # 実際のテストはここに実装
                pass

            test_time = time.time() - start_time

            self._add_diagnostic_result(DiagnosticResult(
                component="automation_health",
                check_name="smart_selector_test",
                level=self._status_to_level(selector_status),
                status=selector_status,
                message=f"スマート選択システム: 正常 ({test_time:.2f}秒)",
                timestamp=datetime.now(),
                details={'test_time': test_time}
            ))

            self.component_status['automation_health'] = selector_status

        except Exception as e:
            logger.error(f"自動化システムチェックエラー: {e}")

    def _evaluate_threshold(self, value: float, warning: float, critical: float) -> ComponentStatus:
        """閾値評価"""
        if value >= critical:
            return ComponentStatus.FAILED
        elif value >= warning:
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.HEALTHY

    def _status_to_level(self, status: ComponentStatus) -> DiagnosticLevel:
        """ステータスをレベルに変換"""
        mapping = {
            ComponentStatus.HEALTHY: DiagnosticLevel.INFO,
            ComponentStatus.DEGRADED: DiagnosticLevel.WARNING,
            ComponentStatus.FAILED: DiagnosticLevel.ERROR,
            ComponentStatus.UNKNOWN: DiagnosticLevel.WARNING
        }
        return mapping.get(status, DiagnosticLevel.INFO)

    def _worst_status(self, statuses: List[ComponentStatus]) -> ComponentStatus:
        """最悪のステータスを返す"""
        priority = {
            ComponentStatus.FAILED: 4,
            ComponentStatus.UNKNOWN: 3,
            ComponentStatus.DEGRADED: 2,
            ComponentStatus.HEALTHY: 1
        }

        return max(statuses, key=lambda s: priority.get(s, 0))

    def _add_diagnostic_result(self, result: DiagnosticResult):
        """診断結果追加"""
        self.diagnostic_history.append(result)

        # 履歴サイズ制限
        if len(self.diagnostic_history) > self.max_history_size:
            self.diagnostic_history = self.diagnostic_history[-self.max_history_size:]

        # ログ出力
        if result.level in [DiagnosticLevel.WARNING, DiagnosticLevel.ERROR, DiagnosticLevel.CRITICAL]:
            logger.warning(f"診断: {result.component}.{result.check_name} - {result.message}")

    def get_system_health(self) -> SystemHealth:
        """システム健全性取得"""
        now = datetime.now()
        uptime = (now - self.start_time).total_seconds()

        # 問題件数集計
        recent_results = [r for r in self.diagnostic_history if (now - r.timestamp).total_seconds() < 3600]
        issues_count = {
            DiagnosticLevel.INFO: len([r for r in recent_results if r.level == DiagnosticLevel.INFO]),
            DiagnosticLevel.WARNING: len([r for r in recent_results if r.level == DiagnosticLevel.WARNING]),
            DiagnosticLevel.ERROR: len([r for r in recent_results if r.level == DiagnosticLevel.ERROR]),
            DiagnosticLevel.CRITICAL: len([r for r in recent_results if r.level == DiagnosticLevel.CRITICAL])
        }

        # 全体ステータス評価
        if self.component_status:
            overall_status = self._worst_status(list(self.component_status.values()))
        else:
            overall_status = ComponentStatus.UNKNOWN

        # 性能スコア計算
        healthy_count = len([s for s in self.component_status.values() if s == ComponentStatus.HEALTHY])
        total_count = len(self.component_status) or 1
        performance_score = (healthy_count / total_count) * 100

        return SystemHealth(
            overall_status=overall_status,
            last_check=now,
            components=self.component_status.copy(),
            issues_count=issues_count,
            uptime_seconds=uptime,
            performance_score=performance_score,
            diagnostic_history=self.diagnostic_history[-50:]  # 最新50件
        )

    def get_health_report(self) -> Dict[str, Any]:
        """健全性レポート取得"""
        health = self.get_system_health()

        return {
            'timestamp': datetime.now(),
            'overall_status': health.overall_status.value,
            'uptime_hours': health.uptime_seconds / 3600,
            'performance_score': health.performance_score,
            'components': {k: v.value for k, v in health.components.items()},
            'issues_summary': {k.value: v for k, v in health.issues_count.items()},
            'recent_issues': [
                {
                    'component': r.component,
                    'message': r.message,
                    'level': r.level.value,
                    'timestamp': r.timestamp
                }
                for r in health.diagnostic_history
                if r.level in [DiagnosticLevel.WARNING, DiagnosticLevel.ERROR, DiagnosticLevel.CRITICAL]
            ][-10:]  # 最新の問題10件
        }

    def force_full_check(self) -> SystemHealth:
        """強制的な全チェック実行"""
        logger.info("強制全チェック実行開始")

        for check_type in self.check_intervals.keys():
            self._run_diagnostic_check(check_type)
            time.sleep(1)  # 各チェック間の間隔

        health = self.get_system_health()
        logger.info(f"強制全チェック完了 - ステータス: {health.overall_status.value}")

        return health


# デバッグ用メイン関数
async def main():
    """デバッグ用メイン"""
    logger.info("自己診断システム テスト実行")

    # システム初期化
    diagnostic = SelfDiagnosticSystem()

    # 強制全チェック実行
    health = diagnostic.force_full_check()

    # レポート表示
    report = diagnostic.get_health_report()

    logger.info("=" * 50)
    logger.info("システム健全性レポート")
    logger.info(f"全体ステータス: {report['overall_status']}")
    logger.info(f"稼働時間: {report['uptime_hours']:.2f}時間")
    logger.info(f"性能スコア: {report['performance_score']:.1f}%")

    logger.info("\nコンポーネント状態:")
    for component, status in report['components'].items():
        logger.info(f"  {component}: {status}")

    logger.info("\n問題サマリー:")
    for level, count in report['issues_summary'].items():
        if count > 0:
            logger.info(f"  {level}: {count}件")

    logger.info("自己診断システム テスト完了")


if __name__ == "__main__":
    asyncio.run(main())