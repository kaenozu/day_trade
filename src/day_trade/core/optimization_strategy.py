"""
最適化戦略システム（Strategy Pattern実装）

重複コードを統合し、設定ベースで最適化レベルを選択する統一アーキテクチャ
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Type, List

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class OptimizationLevel(Enum):
    """最適化レベル列挙"""

    STANDARD = "standard"  # 標準実装（互換性重視）
    OPTIMIZED = "optimized"  # 最適化実装（性能重視）
    ADAPTIVE = "adaptive"  # 適応的（状況に応じて自動選択）
    DEBUG = "debug"  # デバッグ用（詳細ログ付き）
    GPU_ACCELERATED = "gpu_accelerated"  # GPU並列処理（Phase F）


@dataclass
class AdaptiveLevelConfig:
    """適応レベル選択設定 - Issue #638対応"""

    # リソース閾値（パーセント）
    high_load_cpu_threshold: float = 80.0
    high_load_memory_threshold: float = 80.0
    medium_load_cpu_threshold: float = 60.0
    medium_load_memory_threshold: float = 60.0

    # GPU利用可能性チェック
    enable_gpu_detection: bool = True
    gpu_memory_threshold: float = 70.0

    # デバッグモード検出
    enable_debug_mode_detection: bool = True
    debug_mode_env_vars: List[str] = field(default_factory=lambda: ["DEBUG", "DAYTRADE_DEBUG", "CI_DEBUG"])

    # レベル選択優先順位（負荷レベル別）
    high_load_preference: List[OptimizationLevel] = field(default_factory=lambda: [
        OptimizationLevel.STANDARD,
        OptimizationLevel.DEBUG,
    ])

    medium_load_preference: List[OptimizationLevel] = field(default_factory=lambda: [
        OptimizationLevel.OPTIMIZED,
        OptimizationLevel.STANDARD,
    ])

    low_load_preference: List[OptimizationLevel] = field(default_factory=lambda: [
        OptimizationLevel.GPU_ACCELERATED,
        OptimizationLevel.OPTIMIZED,
        OptimizationLevel.STANDARD,
    ])

    # フォールバック戦略
    fallback_level: OptimizationLevel = OptimizationLevel.STANDARD

    # パフォーマンス設定
    cpu_measurement_interval: float = 0.5
    enable_detailed_logging: bool = True


@dataclass
class OptimizationConfig:
    """最適化設定クラス"""

    level: OptimizationLevel = OptimizationLevel.STANDARD
    auto_fallback: bool = True  # 失敗時の自動フォールバック
    performance_monitoring: bool = True  # パフォーマンス監視
    cache_enabled: bool = True  # キャッシュ有効化
    parallel_processing: bool = False  # 並列処理
    batch_size: int = 100  # バッチサイズ
    timeout_seconds: int = 30  # タイムアウト（秒）
    memory_limit_mb: int = 512  # メモリ制限（MB）
    ci_test_mode: bool = False  # CI テストモード（軽量化）
    adaptive_config: Optional[AdaptiveLevelConfig] = None  # 適応レベル選択設定 - Issue #638対応

    @classmethod
    def from_env(cls) -> "OptimizationConfig":
        """環境変数から設定を読み込み"""
        level_str = os.getenv("DAYTRADE_OPTIMIZATION_LEVEL", "standard").lower()
        try:
            level = OptimizationLevel(level_str)
        except ValueError:
            logger.warning(f"無効な最適化レベル: {level_str}, 標準レベルを使用")
            level = OptimizationLevel.STANDARD

        return cls(
            level=level,
            auto_fallback=os.getenv("DAYTRADE_AUTO_FALLBACK", "true").lower() == "true",
            performance_monitoring=os.getenv("DAYTRADE_PERF_MONITORING", "true").lower()
            == "true",
            cache_enabled=os.getenv("DAYTRADE_CACHE_ENABLED", "true").lower() == "true",
            parallel_processing=os.getenv("DAYTRADE_PARALLEL", "false").lower()
            == "true",
            batch_size=int(os.getenv("DAYTRADE_BATCH_SIZE", "100")),
            timeout_seconds=int(os.getenv("DAYTRADE_TIMEOUT", "30")),
            memory_limit_mb=int(os.getenv("DAYTRADE_MEMORY_LIMIT", "512")),
            ci_test_mode=os.getenv("CI", "false").lower() == "true",  # CI環境自動検出
        )

    @classmethod
    def from_file(cls, config_path: str) -> "OptimizationConfig":
        """設定ファイルから読み込み"""
        try:
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)

            # OptimizationLevelの変換
            level_str = data.get("level", "standard").lower()
            level = OptimizationLevel(level_str)

            return cls(
                level=level,
                auto_fallback=data.get("auto_fallback", True),
                performance_monitoring=data.get("performance_monitoring", True),
                cache_enabled=data.get("cache_enabled", True),
                parallel_processing=data.get("parallel_processing", False),
                batch_size=data.get("batch_size", 100),
                timeout_seconds=data.get("timeout_seconds", 30),
                memory_limit_mb=data.get("memory_limit_mb", 512),
            )

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"設定ファイル読み込み失敗: {e}, デフォルト設定を使用")
            return cls()


class OptimizationStrategy(ABC):
    """最適化戦略の基底クラス"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.performance_metrics = {
            "execution_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_time": 0.0,
            "average_time": 0.0,
        }

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """戦略の実行（サブクラスで実装）"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """戦略名の取得"""
        pass

    def record_execution(self, execution_time: float, success: bool) -> None:
        """実行記録"""
        self.performance_metrics["execution_count"] += 1
        self.performance_metrics["total_time"] += execution_time

        if success:
            self.performance_metrics["success_count"] += 1
        else:
            self.performance_metrics["error_count"] += 1

        # 平均時間の更新
        self.performance_metrics["average_time"] = (
            self.performance_metrics["total_time"]
            / self.performance_metrics["execution_count"]
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標の取得"""
        metrics = self.performance_metrics.copy()
        if metrics["execution_count"] > 0:
            metrics["success_rate"] = (
                metrics["success_count"] / metrics["execution_count"]
            )
            metrics["error_rate"] = metrics["error_count"] / metrics["execution_count"]
        return metrics

    def reset_metrics(self) -> None:
        """指標のリセット"""
        self.performance_metrics = {
            "execution_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_time": 0.0,
            "average_time": 0.0,
        }


class OptimizationStrategyFactory:
    """最適化戦略ファクトリー"""

    _strategies: Dict[str, Dict[OptimizationLevel, Type[OptimizationStrategy]]] = {}
    _config: Optional[OptimizationConfig] = None

    @classmethod
    def register_strategy(
        cls,
        component_name: str,
        level: OptimizationLevel,
        strategy_class: Type[OptimizationStrategy],
    ) -> None:
        """戦略の登録"""
        if component_name not in cls._strategies:
            cls._strategies[component_name] = {}
        cls._strategies[component_name][level] = strategy_class
        logger.info(f"戦略登録: {component_name} - {level.value}")

    @classmethod
    def get_strategy(
        cls, component_name: str, config: Optional[OptimizationConfig] = None
    ) -> OptimizationStrategy:
        """戦略の取得"""
        if config is None:
            if cls._config is None:
                cls._config = OptimizationConfig.from_env()
            config = cls._config

        # コンポーネント用の戦略マップを取得
        component_strategies = cls._strategies.get(component_name, {})

        if not component_strategies:
            raise ValueError(f"未登録のコンポーネント: {component_name}")

        # 設定レベルに対応する戦略を取得
        target_level = config.level

        # 適応的モードの場合、実行時条件に基づいて選択
        if target_level == OptimizationLevel.ADAPTIVE:
            target_level = cls._select_adaptive_level(component_name, config)

        # 対象レベルの戦略を取得
        strategy_class = component_strategies.get(target_level)

        # フォールバック処理
        if strategy_class is None and config.auto_fallback:
            # 利用可能な戦略から選択（優先順位: OPTIMIZED -> STANDARD -> その他）
            fallback_order = [OptimizationLevel.OPTIMIZED, OptimizationLevel.STANDARD]
            for fallback_level in fallback_order:
                if fallback_level in component_strategies:
                    strategy_class = component_strategies[fallback_level]
                    logger.warning(
                        f"戦略フォールバック: {component_name} "
                        f"{target_level.value} -> {fallback_level.value}"
                    )
                    break

            # フォールバックでも見つからない場合、利用可能な最初の戦略を使用
            if strategy_class is None and component_strategies:
                strategy_class = list(component_strategies.values())[0]
                logger.warning(f"最終フォールバック戦略使用: {component_name}")

        if strategy_class is None:
            raise ValueError(
                f"利用可能な戦略なし: {component_name} - {target_level.value}"
            )

        return strategy_class(config)

    @classmethod
    def _select_adaptive_level(
        cls, component_name: str, config: OptimizationConfig
    ) -> OptimizationLevel:
        """適応的レベル選択 - Issue #638対応: 改善とパラメータ化"""
        # 適応設定の取得
        adaptive_config = config.adaptive_config or AdaptiveLevelConfig()

        try:
            # 利用可能な戦略レベルを取得
            available_levels = cls._get_available_levels(component_name)

            # システムリソース状況の取得
            resource_info = cls._gather_system_resources(adaptive_config)

            # 特殊条件チェック（デバッグモード、GPU利用可能性等）
            special_conditions = cls._check_special_conditions(adaptive_config, available_levels)

            # 負荷レベルの決定
            load_level = cls._determine_load_level(resource_info, adaptive_config)

            # 最適なレベルの選択
            selected_level = cls._select_optimal_level(
                load_level, special_conditions, available_levels, adaptive_config
            )

            # 詳細ログ出力
            if adaptive_config.enable_detailed_logging:
                cls._log_adaptive_selection_details(
                    component_name, selected_level, load_level,
                    resource_info, special_conditions, available_levels
                )

            return selected_level

        except Exception as e:
            logger.error(f"適応的レベル選択エラー ({component_name}): {e}")
            return adaptive_config.fallback_level

    @classmethod
    def _get_available_levels(cls, component_name: str) -> List[OptimizationLevel]:
        """コンポーネントで利用可能な戦略レベルを取得"""
        component_strategies = cls._strategies.get(component_name, {})
        return list(component_strategies.keys())

    @classmethod
    def _gather_system_resources(cls, adaptive_config: AdaptiveLevelConfig) -> Dict[str, float]:
        """システムリソース情報を収集"""
        import psutil

        resource_info = {}

        try:
            # CPU使用率
            resource_info['cpu_percent'] = psutil.cpu_percent(
                interval=adaptive_config.cpu_measurement_interval
            )

            # メモリ使用率
            memory = psutil.virtual_memory()
            resource_info['memory_percent'] = memory.percent
            resource_info['memory_available_gb'] = memory.available / (1024**3)

            # GPU使用率（利用可能な場合）
            if adaptive_config.enable_gpu_detection:
                resource_info['gpu_available'] = cls._check_gpu_availability()
                if resource_info['gpu_available']:
                    resource_info['gpu_memory_percent'] = cls._get_gpu_memory_usage()

            # システム負荷
            resource_info['load_average'] = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0

        except Exception as e:
            logger.warning(f"システムリソース情報収集エラー: {e}")
            # デフォルト値を設定
            resource_info.update({
                'cpu_percent': 50.0,
                'memory_percent': 50.0,
                'gpu_available': False,
                'load_average': 1.0
            })

        return resource_info

    @classmethod
    def _check_special_conditions(
        cls, adaptive_config: AdaptiveLevelConfig, available_levels: List[OptimizationLevel]
    ) -> Dict[str, bool]:
        """特殊条件をチェック（デバッグモード、CI環境等）"""
        conditions = {}

        # デバッグモード検出
        if adaptive_config.enable_debug_mode_detection:
            conditions['debug_mode'] = any(
                os.getenv(var, '').lower() in ['true', '1', 'yes', 'on']
                for var in adaptive_config.debug_mode_env_vars
            )
        else:
            conditions['debug_mode'] = False

        # CI環境検出
        conditions['ci_environment'] = os.getenv('CI', '').lower() in ['true', '1', 'yes']

        # 利用可能なレベル別条件
        conditions['gpu_level_available'] = OptimizationLevel.GPU_ACCELERATED in available_levels
        conditions['debug_level_available'] = OptimizationLevel.DEBUG in available_levels
        conditions['optimized_level_available'] = OptimizationLevel.OPTIMIZED in available_levels

        return conditions

    @classmethod
    def _determine_load_level(
        cls, resource_info: Dict[str, float], adaptive_config: AdaptiveLevelConfig
    ) -> str:
        """システム負荷レベルを決定"""
        cpu_percent = resource_info.get('cpu_percent', 50.0)
        memory_percent = resource_info.get('memory_percent', 50.0)

        # 高負荷判定
        if (cpu_percent > adaptive_config.high_load_cpu_threshold or
            memory_percent > adaptive_config.high_load_memory_threshold):
            return 'high'

        # 中負荷判定
        elif (cpu_percent > adaptive_config.medium_load_cpu_threshold or
              memory_percent > adaptive_config.medium_load_memory_threshold):
            return 'medium'

        # 低負荷
        else:
            return 'low'

    @classmethod
    def _select_optimal_level(
        cls,
        load_level: str,
        special_conditions: Dict[str, bool],
        available_levels: List[OptimizationLevel],
        adaptive_config: AdaptiveLevelConfig
    ) -> OptimizationLevel:
        """最適なレベルを選択"""

        # 特殊条件の優先処理
        if special_conditions.get('debug_mode') and special_conditions.get('debug_level_available'):
            return OptimizationLevel.DEBUG

        if special_conditions.get('ci_environment'):
            # CI環境では軽量で安定した実装を優先
            for level in [OptimizationLevel.STANDARD, OptimizationLevel.DEBUG]:
                if level in available_levels:
                    return level

        # 負荷レベル別の選択
        if load_level == 'high':
            preferences = adaptive_config.high_load_preference
        elif load_level == 'medium':
            preferences = adaptive_config.medium_load_preference
        else:  # low
            preferences = adaptive_config.low_load_preference

        # 利用可能なレベルから優先順位に基づいて選択
        for preferred_level in preferences:
            if preferred_level in available_levels:
                return preferred_level

        # フォールバック
        return adaptive_config.fallback_level

    @classmethod
    def _check_gpu_availability(cls) -> bool:
        """GPU利用可能性をチェック"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    @classmethod
    def _get_gpu_memory_usage(cls) -> float:
        """GPU メモリ使用率を取得"""
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    used, total = map(int, lines[0].split(', '))
                    return (used / total) * 100.0
        except Exception:
            pass
        return 0.0

    @classmethod
    def _log_adaptive_selection_details(
        cls,
        component_name: str,
        selected_level: OptimizationLevel,
        load_level: str,
        resource_info: Dict[str, float],
        special_conditions: Dict[str, bool],
        available_levels: List[OptimizationLevel]
    ) -> None:
        """適応選択の詳細ログを出力"""
        logger.info(
            f"適応レベル選択 ({component_name}): {selected_level.value} "
            f"[負荷: {load_level}, CPU: {resource_info.get('cpu_percent', 0):.1f}%, "
            f"MEM: {resource_info.get('memory_percent', 0):.1f}%]"
        )

        if special_conditions.get('debug_mode'):
            logger.debug(f"デバッグモード検出: {component_name}")

        if special_conditions.get('ci_environment'):
            logger.debug(f"CI環境検出: {component_name}")

        if resource_info.get('gpu_available'):
            logger.debug(
                f"GPU利用可能 ({component_name}): "
                f"VRAM使用率 {resource_info.get('gpu_memory_percent', 0):.1f}%"
            )

    @classmethod
    def set_global_config(cls, config: OptimizationConfig) -> None:
        """グローバル設定の設定"""
        cls._config = config
        logger.info(f"グローバル最適化設定更新: {config.level.value}")

    @classmethod
    def get_registered_components(cls) -> Dict[str, list]:
        """登録済みコンポーネント一覧の取得"""
        result = {}
        for component, strategies in cls._strategies.items():
            result[component] = [level.value for level in strategies.keys()]
        return result

    @classmethod
    def create_config_template(cls, output_path: str) -> None:
        """設定テンプレートファイルの作成"""
        template = {
            "level": "standard",
            "auto_fallback": True,
            "performance_monitoring": True,
            "cache_enabled": True,
            "parallel_processing": False,
            "batch_size": 100,
            "timeout_seconds": 30,
            "memory_limit_mb": 512,
            "_comments": {
                "level": "standard, optimized, adaptive, debug",
                "auto_fallback": "失敗時に他の実装に自動的にフォールバック",
                "performance_monitoring": "パフォーマンス監視の有効化",
                "cache_enabled": "キャッシュ機能の有効化",
                "parallel_processing": "並列処理の有効化",
                "batch_size": "バッチ処理のサイズ",
                "timeout_seconds": "処理タイムアウト時間",
                "memory_limit_mb": "メモリ使用量制限",
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        logger.info(f"設定テンプレート作成: {output_path}")


# デコレータ関数
def optimization_strategy(component_name: str, level: OptimizationLevel):
    """最適化戦略デコレータ"""

    def decorator(cls: Type[OptimizationStrategy]):
        OptimizationStrategyFactory.register_strategy(component_name, level, cls)
        return cls

    return decorator


def get_optimized_implementation(
    component_name: str, config: Optional[OptimizationConfig] = None
):
    """最適化実装取得ヘルパー"""
    return OptimizationStrategyFactory.get_strategy(component_name, config)
