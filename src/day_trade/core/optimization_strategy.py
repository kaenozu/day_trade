"""
最適化戦略システム（Strategy Pattern実装）

重複コードを統合し、設定ベースで最適化レベルを選択する統一アーキテクチャ
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Type

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

    @classmethod
    def from_env(cls) -> "OptimizationConfig":
        """環境変数から設定を読み込み - Issue #642対応"""
        level_str = os.getenv("DAYTRADE_OPTIMIZATION_LEVEL", "standard").lower()
        try:
            level = OptimizationLevel(level_str)
        except ValueError:
            logger.warning(f"無効な最適化レベル: {level_str}, 標準レベルを使用")
            level = OptimizationLevel.STANDARD

        return cls(
            level=level,
            auto_fallback=cls._parse_env_bool("DAYTRADE_AUTO_FALLBACK", True),
            performance_monitoring=cls._parse_env_bool("DAYTRADE_PERF_MONITORING", True),
            cache_enabled=cls._parse_env_bool("DAYTRADE_CACHE_ENABLED", True),
            parallel_processing=cls._parse_env_bool("DAYTRADE_PARALLEL", False),
            batch_size=cls._parse_env_int("DAYTRADE_BATCH_SIZE", 100),
            timeout_seconds=cls._parse_env_int("DAYTRADE_TIMEOUT", 30),
            memory_limit_mb=cls._parse_env_int("DAYTRADE_MEMORY_LIMIT", 512),
            ci_test_mode=cls._parse_env_bool("CI", False),  # CI環境自動検出
        )

    @staticmethod
    def _parse_env_bool(env_var: str, default: bool) -> bool:
        """
        環境変数から堅牢なブール値パース - Issue #642対応

        Args:
            env_var: 環境変数名
            default: デフォルト値

        Returns:
            bool: パース結果

        Note:
            以下の値をTrueとして扱う（大文字小文字無視）:
            - "true", "yes", "1", "on", "enable", "enabled"
            以下の値をFalseとして扱う:
            - "false", "no", "0", "off", "disable", "disabled"
        """
        env_value = os.getenv(env_var)

        if env_value is None:
            return default

        # 文字列を正規化
        normalized = env_value.strip().lower()

        # 空文字列の場合はデフォルト値
        if not normalized:
            logger.debug(f"環境変数 {env_var} が空文字列のため、デフォルト値 {default} を使用")
            return default

        # Trueとして扱う値
        true_values = {"true", "yes", "1", "on", "enable", "enabled", "y", "t"}

        # Falseとして扱う値
        false_values = {"false", "no", "0", "off", "disable", "disabled", "n", "f"}

        if normalized in true_values:
            logger.debug(f"環境変数 {env_var}='{env_value}' -> True")
            return True
        elif normalized in false_values:
            logger.debug(f"環境変数 {env_var}='{env_value}' -> False")
            return False
        else:
            # 不明な値の場合は警告してデフォルト値を使用
            logger.warning(
                f"環境変数 {env_var} の値 '{env_value}' を解釈できません。"
                f"デフォルト値 {default} を使用します。"
                f"有効な値: {', '.join(true_values | false_values)}"
            )
            return default

    @staticmethod
    def _parse_env_int(env_var: str, default: int) -> int:
        """
        環境変数から堅牢な整数パース - Issue #642対応

        Args:
            env_var: 環境変数名
            default: デフォルト値

        Returns:
            int: パース結果
        """
        env_value = os.getenv(env_var)

        if env_value is None:
            return default

        try:
            # 空白削除後に整数変換
            normalized = env_value.strip()
            if not normalized:
                logger.debug(f"環境変数 {env_var} が空文字列のため、デフォルト値 {default} を使用")
                return default

            result = int(normalized)
            logger.debug(f"環境変数 {env_var}='{env_value}' -> {result}")
            return result

        except ValueError as e:
            logger.warning(
                f"環境変数 {env_var} の値 '{env_value}' を整数として解釈できません。"
                f"デフォルト値 {default} を使用します。エラー: {e}"
            )
            return default

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
        """実行記録（堅牢化）- Issue #636対応"""
        # 入力値の検証
        if not self._validate_execution_time(execution_time):
            logger.warning(f"無効な実行時間: {execution_time}, 記録をスキップ")
            return

        # 基本メトリクスの更新
        self.performance_metrics["execution_count"] += 1
        self.performance_metrics["total_time"] += execution_time

        # 成功/失敗カウントの更新
        if success:
            self.performance_metrics["success_count"] += 1
        else:
            self.performance_metrics["error_count"] += 1

        # 平均時間の安全な更新
        self._update_average_time_safely(execution_time)

    def _validate_execution_time(self, execution_time: float) -> bool:
        """実行時間の妥当性検証

        Args:
            execution_time: 実行時間（秒）

        Returns:
            bool: 妥当な場合True
        """
        try:
            # 基本的な数値検証
            if not isinstance(execution_time, (int, float)):
                return False

            # NaN, Inf の検証
            if not self._is_finite_positive(execution_time):
                return False

            # 合理的な範囲内か検証（0以上、1時間未満）
            if execution_time < 0 or execution_time > 3600:
                return False

            return True

        except Exception as e:
            logger.error(f"実行時間検証エラー: {e}")
            return False

    def _is_finite_positive(self, value: float) -> bool:
        """有限で非負の数値かチェック

        Args:
            value: チェック対象の値

        Returns:
            bool: 有限で非負の場合True
        """
        import math
        return math.isfinite(value) and value >= 0

    def _update_average_time_safely(self, execution_time: float) -> None:
        """平均時間の安全な更新（ZeroDivisionError対策）

        Args:
            execution_time: 新しい実行時間
        """
        try:
            execution_count = self.performance_metrics["execution_count"]

            # ゼロ除算の防止
            if execution_count <= 0:
                logger.warning("実行回数が0以下です。平均時間を0に設定")
                self.performance_metrics["average_time"] = 0.0
                return

            # 指数移動平均の使用（外れ値に対してより堅牢）
            if execution_count == 1:
                # 初回実行時は単純に実行時間を設定
                self.performance_metrics["average_time"] = execution_time
            else:
                # 指数移動平均を使用（α=0.1、最近の値により重みを置く）
                alpha = self._calculate_smoothing_factor(execution_count)
                current_avg = self.performance_metrics["average_time"]
                self.performance_metrics["average_time"] = (
                    alpha * execution_time + (1 - alpha) * current_avg
                )

        except Exception as e:
            logger.error(f"平均時間更新エラー: {e}")
            # フォールバック: 単純平均を使用
            self._fallback_average_calculation()

    def _calculate_smoothing_factor(self, execution_count: int) -> float:
        """適応的な平滑化係数の計算

        Args:
            execution_count: 実行回数

        Returns:
            float: 平滑化係数（0.05-0.3の範囲）
        """
        # 実行回数が少ない場合は新しい値により重みを置く
        if execution_count <= 5:
            return 0.3  # 新しい値に30%の重み
        elif execution_count <= 20:
            return 0.15  # 新しい値に15%の重み
        else:
            return 0.05  # 新しい値に5%の重み（安定した平均）

    def _fallback_average_calculation(self) -> None:
        """フォールバック用の単純平均計算"""
        try:
            execution_count = self.performance_metrics["execution_count"]
            total_time = self.performance_metrics["total_time"]

            if execution_count > 0:
                self.performance_metrics["average_time"] = total_time / execution_count
            else:
                self.performance_metrics["average_time"] = 0.0

        except Exception as e:
            logger.error(f"フォールバック平均計算エラー: {e}")
            self.performance_metrics["average_time"] = 0.0

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
        """戦略の登録 - Issue #639対応: 型検証強化"""
        # 型検証の実行
        cls._validate_strategy_class(strategy_class, component_name, level)

        if component_name not in cls._strategies:
            cls._strategies[component_name] = {}
        cls._strategies[component_name][level] = strategy_class
        logger.info(f"戦略登録: {component_name} - {level.value}")

    @classmethod
    def _validate_strategy_class(
        cls,
        strategy_class: Type[OptimizationStrategy],
        component_name: str,
        level: OptimizationLevel,
    ) -> None:
        """
        戦略クラスの型検証 - Issue #639対応

        Args:
            strategy_class: 検証対象の戦略クラス
            component_name: コンポーネント名（エラーメッセージ用）
            level: 最適化レベル（エラーメッセージ用）

        Raises:
            TypeError: 型検証に失敗した場合
            ValueError: 抽象メソッドが未実装の場合
        """
        import inspect

        # 1. 基本的な型検証
        if not inspect.isclass(strategy_class):
            raise TypeError(
                f"戦略登録エラー: {component_name}[{level.value}] - "
                f"strategy_classはクラスである必要があります。受信: {type(strategy_class)}"
            )

        # 2. OptimizationStrategyのサブクラス検証
        if not issubclass(strategy_class, OptimizationStrategy):
            raise TypeError(
                f"戦略登録エラー: {component_name}[{level.value}] - "
                f"strategy_classはOptimizationStrategyのサブクラスである必要があります。"
                f"受信: {strategy_class.__name__}"
            )

        # 3. 抽象クラスでないことを検証
        if inspect.isabstract(strategy_class):
            # 未実装の抽象メソッドを特定
            abstract_methods = getattr(strategy_class, '__abstractmethods__', set())
            raise ValueError(
                f"戦略登録エラー: {component_name}[{level.value}] - "
                f"strategy_class '{strategy_class.__name__}' は抽象クラスです。"
                f"未実装の抽象メソッド: {', '.join(abstract_methods)}"
            )

        # 4. 必須メソッドの実装検証
        required_methods = ['execute', 'get_strategy_name']
        missing_methods = []

        for method_name in required_methods:
            if not hasattr(strategy_class, method_name):
                missing_methods.append(method_name)
            else:
                method = getattr(strategy_class, method_name)
                if not callable(method):
                    missing_methods.append(f"{method_name} (not callable)")

        if missing_methods:
            raise ValueError(
                f"戦略登録エラー: {component_name}[{level.value}] - "
                f"strategy_class '{strategy_class.__name__}' に必須メソッドが不足: "
                f"{', '.join(missing_methods)}"
            )

        # 5. コンストラクタ検証（ConfigパラメータをOptimizationConfigとして受け取ることを確認）
        try:
            constructor_sig = inspect.signature(strategy_class.__init__)
            params = list(constructor_sig.parameters.values())[1:]  # selfを除外

            if len(params) < 1:
                raise ValueError(
                    f"戦略登録エラー: {component_name}[{level.value}] - "
                    f"strategy_class '{strategy_class.__name__}' のコンストラクタは"
                    f"configパラメータを受け取る必要があります"
                )

        except Exception as e:
            logger.warning(
                f"コンストラクタ検証をスキップ: {component_name}[{level.value}] - {e}"
            )

        # 6. インスタンス化テスト（軽量）
        try:
            # テスト用の軽量config
            test_config = OptimizationConfig(level=OptimizationLevel.STANDARD)

            # インスタンス化テスト
            test_instance = strategy_class(test_config)

            # 基本メソッドの呼び出しテスト
            strategy_name = test_instance.get_strategy_name()
            if not isinstance(strategy_name, str) or not strategy_name.strip():
                raise ValueError(
                    f"get_strategy_name()は空でない文字列を返す必要があります。"
                    f"受信: {repr(strategy_name)}"
                )

        except Exception as e:
            raise ValueError(
                f"戦略登録エラー: {component_name}[{level.value}] - "
                f"strategy_class '{strategy_class.__name__}' のインスタンス化テストに失敗: {e}"
            )

        logger.debug(f"戦略クラス検証成功: {component_name}[{level.value}] - {strategy_class.__name__}")

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

        # フォールバック処理 - Issue #640対応
        if strategy_class is None and config.auto_fallback:
            strategy_class, fallback_level = cls._find_fallback_strategy(
                component_strategies, target_level, component_name
            )

        if strategy_class is None:
            raise ValueError(
                f"利用可能な戦略なし: {component_name} - {target_level.value}"
            )

        return strategy_class(config)

    @classmethod
    def _find_fallback_strategy(
        cls,
        component_strategies: Dict[OptimizationLevel, Type[OptimizationStrategy]],
        target_level: OptimizationLevel,
        component_name: str,
    ) -> tuple:
        """
        フォールバック戦略の検索 - Issue #640対応

        Args:
            component_strategies: 利用可能な戦略マップ
            target_level: 目標最適化レベル
            component_name: コンポーネント名

        Returns:
            (戦略クラス, 使用されたレベル) のタプル
            見つからない場合は (None, None)
        """
        # 明確で予測可能なフォールバック順序を定義
        fallback_hierarchy = [
            OptimizationLevel.OPTIMIZED,  # 最優先: パフォーマンス重視
            OptimizationLevel.STANDARD,   # 次善: 安定性重視
            OptimizationLevel.DEBUG,      # 開発・デバッグ用
            OptimizationLevel.ADAPTIVE,   # 適応的（通常は最初で解決）
            OptimizationLevel.GPU_ACCELERATED,  # 特殊用途
        ]

        # 目標レベルを階層から除外（既に試行済みのため）
        available_fallbacks = [level for level in fallback_hierarchy if level != target_level]

        # 階層順にフォールバック戦略を検索
        for fallback_level in available_fallbacks:
            if fallback_level in component_strategies:
                strategy_class = component_strategies[fallback_level]
                logger.warning(
                    f"戦略フォールバック: {component_name} "
                    f"{target_level.value} -> {fallback_level.value} "
                    f"(優先順位に基づく選択)"
                )
                return strategy_class, fallback_level

        # 階層に存在しない戦略レベルがある場合の処理
        # (将来的な拡張や非標準レベル対応)
        remaining_strategies = [
            (level, strategy) for level, strategy in component_strategies.items()
            if level not in fallback_hierarchy and level != target_level
        ]

        if remaining_strategies:
            # レベル名でソートして予測可能な選択を実現
            remaining_strategies.sort(key=lambda x: x[0].value)
            fallback_level, strategy_class = remaining_strategies[0]
            logger.warning(
                f"非標準戦略フォールバック: {component_name} "
                f"{target_level.value} -> {fallback_level.value} "
                f"(レベル名ソート順による選択)"
            )
            return strategy_class, fallback_level

        # フォールバック戦略が見つからない
        logger.error(f"フォールバック戦略が見つかりません: {component_name}")
        return None, None

    @classmethod
    def _select_adaptive_level(
        cls, component_name: str, config: OptimizationConfig
    ) -> OptimizationLevel:
        """適応的レベル選択"""
        # システムリソース状況を考慮した適応的選択
        try:
            # psutilが利用可能な場合のみシステム監視を実行
            memory_percent, cpu_percent = cls._get_system_metrics()

            if memory_percent is not None and cpu_percent is not None:
                # システムメトリクスが取得できた場合の適応的選択
                return cls._select_level_by_metrics(memory_percent, cpu_percent)
            else:
                # システムメトリクスが取得できない場合のフォールバック選択
                return cls._select_level_fallback(config)

        except Exception as e:
            logger.error(f"適応的レベル選択エラー: {e}, フォールバック選択使用")
            return cls._select_level_fallback(config)

    @classmethod
    def _get_system_metrics(cls) -> tuple[Optional[float], Optional[float]]:
        """システムメトリクスを安全に取得（psutilオプショナル対応）

        Returns:
            tuple: (memory_percent, cpu_percent) または (None, None)
        """
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)  # interval短縮でレスポンス向上
            return memory_percent, cpu_percent
        except ImportError:
            logger.warning("psutilが利用できません。システム監視なしでフォールバック選択を使用")
            return None, None
        except Exception as e:
            logger.warning(f"システムメトリクス取得エラー: {e}")
            return None, None

    @classmethod
    def _select_level_by_metrics(cls, memory_percent: float, cpu_percent: float) -> OptimizationLevel:
        """システムメトリクスに基づく適応的レベル選択

        Args:
            memory_percent: メモリ使用率（%）
            cpu_percent: CPU使用率（%）

        Returns:
            OptimizationLevel: 選択された最適化レベル
        """
        # 高負荷時は標準実装を選択（安定性重視）
        if memory_percent > 80 or cpu_percent > 80:
            logger.info(
                f"高負荷検出、標準実装選択: CPU={cpu_percent:.1f}%, MEM={memory_percent:.1f}%"
            )
            return OptimizationLevel.STANDARD

        # 中負荷時は最適化実装を選択（バランス重視）
        elif memory_percent > 60 or cpu_percent > 60:
            logger.info(
                f"中負荷検出、最適化実装選択: CPU={cpu_percent:.1f}%, MEM={memory_percent:.1f}%"
            )
            return OptimizationLevel.OPTIMIZED

        # 低負荷時は最適化実装を選択（パフォーマンス重視）
        else:
            logger.info(
                f"低負荷検出、最適化実装選択: CPU={cpu_percent:.1f}%, MEM={memory_percent:.1f}%"
            )
            return OptimizationLevel.OPTIMIZED

    @classmethod
    def _select_level_fallback(cls, config: OptimizationConfig) -> OptimizationLevel:
        """システム監視が利用できない場合のフォールバック選択

        Args:
            config: 最適化設定

        Returns:
            OptimizationLevel: フォールバック最適化レベル
        """
        # CI環境や軽量モードでは標準実装を選択
        if config.ci_test_mode:
            logger.info("CI環境検出、標準実装選択")
            return OptimizationLevel.STANDARD

        # メモリ制限が厳しい場合は標準実装を選択
        if config.memory_limit_mb < 256:
            logger.info(f"メモリ制限検出({config.memory_limit_mb}MB)、標準実装選択")
            return OptimizationLevel.STANDARD

        # その他の場合は最適化実装を選択（デフォルト）
        logger.info("システム監視なし、最適化実装選択（デフォルト）")
        return OptimizationLevel.OPTIMIZED

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
