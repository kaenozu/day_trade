"""
最適化戦略システム（Strategy Pattern実装）

重複コードを統合し、設定ベースで最適化レベルを選択する統一アーキテクチャ
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, Type
import os
import json
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class OptimizationLevel(Enum):
    """最適化レベル列挙"""
    STANDARD = "standard"        # 標準実装（互換性重視）
    OPTIMIZED = "optimized"      # 最適化実装（性能重視）
    ADAPTIVE = "adaptive"        # 適応的（状況に応じて自動選択）
    DEBUG = "debug"              # デバッグ用（詳細ログ付き）
    GPU_ACCELERATED = "gpu_accelerated"  # GPU並列処理（Phase F）


@dataclass
class OptimizationConfig:
    """最適化設定クラス"""
    level: OptimizationLevel = OptimizationLevel.STANDARD
    auto_fallback: bool = True          # 失敗時の自動フォールバック
    performance_monitoring: bool = True  # パフォーマンス監視
    cache_enabled: bool = True          # キャッシュ有効化
    parallel_processing: bool = False   # 並列処理
    batch_size: int = 100              # バッチサイズ
    timeout_seconds: int = 30          # タイムアウト（秒）
    memory_limit_mb: int = 512         # メモリ制限（MB）
    
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
            performance_monitoring=os.getenv("DAYTRADE_PERF_MONITORING", "true").lower() == "true",
            cache_enabled=os.getenv("DAYTRADE_CACHE_ENABLED", "true").lower() == "true",
            parallel_processing=os.getenv("DAYTRADE_PARALLEL", "false").lower() == "true",
            batch_size=int(os.getenv("DAYTRADE_BATCH_SIZE", "100")),
            timeout_seconds=int(os.getenv("DAYTRADE_TIMEOUT", "30")),
            memory_limit_mb=int(os.getenv("DAYTRADE_MEMORY_LIMIT", "512")),
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "OptimizationConfig":
        """設定ファイルから読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # OptimizationLevelの変換
            level_str = data.get('level', 'standard').lower()
            level = OptimizationLevel(level_str)
            
            return cls(
                level=level,
                auto_fallback=data.get('auto_fallback', True),
                performance_monitoring=data.get('performance_monitoring', True),
                cache_enabled=data.get('cache_enabled', True),
                parallel_processing=data.get('parallel_processing', False),
                batch_size=data.get('batch_size', 100),
                timeout_seconds=data.get('timeout_seconds', 30),
                memory_limit_mb=data.get('memory_limit_mb', 512),
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
            self.performance_metrics["total_time"] / 
            self.performance_metrics["execution_count"]
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標の取得"""
        metrics = self.performance_metrics.copy()
        if metrics["execution_count"] > 0:
            metrics["success_rate"] = metrics["success_count"] / metrics["execution_count"]
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
        self, 
        component_name: str, 
        level: OptimizationLevel, 
        strategy_class: Type[OptimizationStrategy]
    ) -> None:
        """戦略の登録"""
        if component_name not in self._strategies:
            self._strategies[component_name] = {}
        self._strategies[component_name][level] = strategy_class
        logger.info(f"戦略登録: {component_name} - {level.value}")
    
    @classmethod
    def get_strategy(
        self, 
        component_name: str, 
        config: Optional[OptimizationConfig] = None
    ) -> OptimizationStrategy:
        """戦略の取得"""
        if config is None:
            if self._config is None:
                self._config = OptimizationConfig.from_env()
            config = self._config
        
        # コンポーネント用の戦略マップを取得
        component_strategies = self._strategies.get(component_name, {})
        
        if not component_strategies:
            raise ValueError(f"未登録のコンポーネント: {component_name}")
        
        # 設定レベルに対応する戦略を取得
        target_level = config.level
        
        # 適応的モードの場合、実行時条件に基づいて選択
        if target_level == OptimizationLevel.ADAPTIVE:
            target_level = self._select_adaptive_level(component_name, config)
        
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
            raise ValueError(f"利用可能な戦略なし: {component_name} - {target_level.value}")
        
        return strategy_class(config)
    
    @classmethod
    def _select_adaptive_level(
        self, 
        component_name: str, 
        config: OptimizationConfig
    ) -> OptimizationLevel:
        """適応的レベル選択"""
        # システムリソース状況を考慮した適応的選択
        import psutil
        
        try:
            # メモリ使用率チェック
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 高負荷時は標準実装を選択
            if memory_percent > 80 or cpu_percent > 80:
                logger.info(f"高負荷検出、標準実装選択: CPU={cpu_percent}%, MEM={memory_percent}%")
                return OptimizationLevel.STANDARD
            
            # 中負荷時は最適化実装を選択
            elif memory_percent > 60 or cpu_percent > 60:
                logger.info(f"中負荷検出、最適化実装選択: CPU={cpu_percent}%, MEM={memory_percent}%")
                return OptimizationLevel.OPTIMIZED
            
            # 低負荷時は最適化実装を選択（デフォルト）
            else:
                logger.info(f"低負荷検出、最適化実装選択: CPU={cpu_percent}%, MEM={memory_percent}%")
                return OptimizationLevel.OPTIMIZED
                
        except Exception as e:
            logger.error(f"適応的レベル選択エラー: {e}, 標準レベル使用")
            return OptimizationLevel.STANDARD
    
    @classmethod
    def set_global_config(self, config: OptimizationConfig) -> None:
        """グローバル設定の設定"""
        self._config = config
        logger.info(f"グローバル最適化設定更新: {config.level.value}")
    
    @classmethod
    def get_registered_components(self) -> Dict[str, list]:
        """登録済みコンポーネント一覧の取得"""
        result = {}
        for component, strategies in self._strategies.items():
            result[component] = [level.value for level in strategies.keys()]
        return result
    
    @classmethod
    def create_config_template(self, output_path: str) -> None:
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
                "memory_limit_mb": "メモリ使用量制限"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"設定テンプレート作成: {output_path}")


# デコレータ関数
def optimization_strategy(component_name: str, level: OptimizationLevel):
    """最適化戦略デコレータ"""
    def decorator(cls: Type[OptimizationStrategy]):
        OptimizationStrategyFactory.register_strategy(component_name, level, cls)
        return cls
    return decorator


def get_optimized_implementation(component_name: str, config: Optional[OptimizationConfig] = None):
    """最適化実装取得ヘルパー"""
    return OptimizationStrategyFactory.get_strategy(component_name, config)