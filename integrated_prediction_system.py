#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Prediction System - ML統合予測システム

複数のMLシステム間での予測統合とルーティング
Issue #855実装：責務明確化と設定外部化による柔軟な統合システム
"""

import asyncio
import sqlite3
import yaml
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

# 共通ユーティリティ
from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix

# Windows環境対応
apply_windows_encoding_fix()

# ロギング設定
logger = logging.getLogger(__name__)


class SystemType(Enum):
    """利用可能なMLシステムタイプ"""
    ORIGINAL = "original_system"
    ADVANCED = "advanced_system"
    SIMPLE = "simple_prediction"


class PredictionStrategy(Enum):
    """予測戦略"""
    SINGLE_BEST = "single_best"
    ENSEMBLE = "ensemble"
    FALLBACK_CHAIN = "fallback_chain"


@dataclass
class SystemMetrics:
    """システム性能メトリクス"""
    accuracy: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    prediction_count: int = 0


@dataclass
class PredictionResult:
    """予測結果"""
    prediction: Any
    system_used: str
    confidence: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """システム設定"""
    name: str
    description: str
    module: str
    class_name: str
    model_types: List[str]
    prediction_methods: Dict[str, str]
    expected_accuracy: float
    typical_response_time: float


class ModelCache:
    """モデルキャッシュ管理"""

    def __init__(self, max_size: int = 10, ttl_minutes: int = 60):
        self.max_size = max_size
        self.ttl_minutes = ttl_minutes
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Any]:
        """キャッシュからモデルを取得"""
        if key not in self.cache:
            return None

        # TTLチェック
        if self._is_expired(key):
            self.remove(key)
            return None

        # アクセス時刻を更新
        self.access_times[key] = datetime.now()
        return self.cache[key].get('model')

    def put(self, key: str, model: Any, metadata: Dict[str, Any] = None):
        """モデルをキャッシュに格納"""
        # キャッシュサイズ制限
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        self.cache[key] = {
            'model': model,
            'metadata': metadata or {},
            'created_at': datetime.now()
        }
        self.access_times[key] = datetime.now()

    def remove(self, key: str):
        """キャッシュからモデルを削除"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)

    def clear(self):
        """キャッシュをクリア"""
        self.cache.clear()
        self.access_times.clear()

    def _is_expired(self, key: str) -> bool:
        """TTL期限切れチェック"""
        if key not in self.cache:
            return True

        created_at = self.cache[key]['created_at']
        return datetime.now() - created_at > timedelta(minutes=self.ttl_minutes)

    def _evict_oldest(self):
        """最も古いエントリを削除"""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.keys(),
                        key=lambda k: self.access_times[k])
        self.remove(oldest_key)


class IntegratedPredictionSystem:
    """
    ML統合予測システム

    複数のMLシステム間での予測統合とルーティングを管理
    外部設定による柔軟なシステム選択とフォールバック機能
    """

    def __init__(self,
                 config_path: Optional[Path] = None,
                 db_path: Optional[Path] = None):

        self.config_path = config_path or Path("config/model_integration.yaml")
        self.db_path = db_path or Path("ml_models_data/integration_system.db")

        # 設定とメトリクス
        self.config = {}
        self.system_configs: Dict[str, SystemConfig] = {}
        self.system_metrics: Dict[str, SystemMetrics] = {}

        # システムインスタンス（遅延初期化）
        self.system_instances: Dict[str, Any] = {}

        # キャッシュシステム
        self.model_cache: Optional[ModelCache] = None

        # 初期化
        self._load_configuration()
        self._init_database()
        self._init_cache()

        logger.info("Integrated Prediction System initialized")

    def _load_configuration(self):
        """外部設定ファイルからの設定読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

            # システム設定の解析
            self._parse_system_configs()

            # ログレベル設定
            log_level = self.config.get('logging', {}).get('level', 'INFO')
            logger.setLevel(getattr(logging, log_level))

        except Exception as e:
            logger.error(f"設定読み込みに失敗しました: {e}")
            self._load_default_config()

    def _create_default_config(self):
        """デフォルト設定ファイルの作成"""
        default_config = {
            'integration': {
                'default_system': 'original_system',
                'fallback_enabled': True
            },
            'systems': {
                'original_system': {
                    'name': 'ML Prediction Models',
                    'module': 'ml_prediction_models',
                    'class': 'MLPredictionModels'
                }
            },
            'cache': {
                'enabled': True,
                'max_models_in_memory': 10
            }
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"デフォルト設定ファイルを作成しました: {self.config_path}")
            self.config = default_config
        except Exception as e:
            logger.error(f"デフォルト設定ファイルの作成に失敗しました: {e}")

    def _parse_system_configs(self):
        """システム設定の解析"""
        systems_config = self.config.get('systems', {})

        for system_id, system_data in systems_config.items():
            try:
                config = SystemConfig(
                    name=system_data.get('name', system_id),
                    description=system_data.get('description', ''),
                    module=system_data.get('module', ''),
                    class_name=system_data.get('class', ''),
                    model_types=system_data.get('model_types', []),
                    prediction_methods=system_data.get('prediction_methods', {}),
                    expected_accuracy=system_data.get('expected_accuracy', 80.0),
                    typical_response_time=system_data.get('typical_response_time', 1.0)
                )
                self.system_configs[system_id] = config
                self.system_metrics[system_id] = SystemMetrics()

            except Exception as e:
                logger.error(f"システム設定の解析に失敗しました {system_id}: {e}")

    def _load_default_config(self):
        """デフォルト設定の読み込み"""
        self.config = {
            'integration': {'default_system': 'original_system'},
            'cache': {'enabled': True}
        }
        logger.info("デフォルト設定を適用しました")

    def _init_database(self):
        """データベースの初期化"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # システムメトリクステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        system_id TEXT NOT NULL,
                        accuracy REAL,
                        response_time REAL,
                        error_rate REAL,
                        prediction_count INTEGER,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # 予測履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prediction_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        system_used TEXT NOT NULL,
                        response_time REAL,
                        confidence REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # システム選択履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_selection_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        selected_system TEXT NOT NULL,
                        fallback_used BOOLEAN DEFAULT FALSE,
                        selection_reason TEXT,
                        timestamp TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                logger.info("データベーステーブルを初期化しました")

        except Exception as e:
            logger.error(f"データベース初期化に失敗しました: {e}")
            raise

    def _init_cache(self):
        """キャッシュシステムの初期化"""
        cache_config = self.config.get('cache', {})

        if cache_config.get('enabled', True):
            max_size = cache_config.get('max_models_in_memory', 10)
            ttl_minutes = cache_config.get('cache_ttl_minutes', 60)
            self.model_cache = ModelCache(max_size, ttl_minutes)
            logger.info(f"モデルキャッシュを初期化しました: max_size={max_size}, ttl={ttl_minutes}分")

    async def _get_system_instance(self, system_id: str) -> Optional[Any]:
        """システムインスタンスの取得（遅延初期化）"""
        if system_id in self.system_instances:
            return self.system_instances[system_id]

        if system_id not in self.system_configs:
            logger.error(f"未知のシステムID: {system_id}")
            return None

        config = self.system_configs[system_id]

        try:
            # 動的インポート
            module = __import__(config.module, fromlist=[config.class_name])
            system_class = getattr(module, config.class_name)

            # インスタンス作成
            instance = system_class()
            self.system_instances[system_id] = instance

            logger.info(f"システムインスタンスを作成しました: {system_id}")
            return instance

        except Exception as e:
            logger.error(f"システムインスタンスの作成に失敗しました {system_id}: {e}")
            return None

    def _select_optimal_system(self, symbol: str) -> str:
        """最適なシステムの選択"""
        # 銘柄別設定チェック
        symbol_prefs = self.config.get('symbol_preferences', {})
        if symbol in symbol_prefs:
            preferred = symbol_prefs[symbol].get('preferred_system')
            if preferred and preferred in self.system_configs:
                logger.debug(f"銘柄別設定によりシステム選択: {symbol} -> {preferred}")
                return preferred

        # 性能ベース選択（将来実装）
        if self.config.get('integration', {}).get('performance_based_selection', False):
            # TODO: 性能メトリクスに基づく最適システム選択
            pass

        # デフォルトシステム
        default_system = self.config.get('integration', {}).get('default_system', 'original_system')
        logger.debug(f"デフォルトシステムを選択: {symbol} -> {default_system}")
        return default_system

    async def predict(self, symbol: str, features: Optional[Any] = None,
                     strategy_name: str = "single_best_confidence",
                     use_ensemble: bool = False) -> PredictionResult:
        """
        統合予測の実行

        Args:
            symbol: 銘柄コード
            features: 特徴量データ（システムによって不要な場合あり）
            strategy_name: 予測戦略名
            use_ensemble: アンサンブル予測の使用

        Returns:
            PredictionResult: 予測結果
        """
        start_time = datetime.now()

        try:
            if use_ensemble:
                # アンサンブル予測
                return await self._execute_ensemble_prediction(symbol, features, strategy_name)
            else:
                # 単一システム予測
                return await self._execute_single_prediction(symbol, features, strategy_name)

        except Exception as e:
            error_msg = f"統合予測エラー: {e}"
            logger.error(error_msg)

            return PredictionResult(
                prediction=None,
                system_used="error",
                confidence=0.0,
                response_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': error_msg}
            )

    async def _execute_single_prediction(self, symbol: str, features: Any, strategy_name: str) -> PredictionResult:
        """単一システム予測の実行"""
        start_time = datetime.now()

        # 最適システムの選択
        selected_system = self._select_optimal_system(symbol)

        # システム選択履歴の記録
        await self._record_system_selection(symbol, selected_system, False, "optimal_selection")

        # 予測実行
        result = await self._execute_prediction(selected_system, symbol, features)

        if result.prediction is not None:
            # 成功時の処理
            response_time = (datetime.now() - start_time).total_seconds()
            await self._record_prediction_success(symbol, selected_system, response_time, result.confidence)
            await self._update_system_metrics(selected_system, response_time, True)

            logger.info(f"単一システム予測成功: {symbol} using {selected_system}")
            return result

        # フォールバック処理
        if self.config.get('fallback', {}).get('enabled', True):
            logger.warning(f"メインシステム失敗、フォールバックを試行: {symbol}")
            return await self._execute_fallback(symbol, features, selected_system)

        # フォールバック無効時は失敗結果を返す
        logger.error(f"予測失敗、フォールバックも無効: {symbol}")
        return PredictionResult(
            prediction=None,
            system_used=selected_system,
            confidence=0.0,
            response_time=(datetime.now() - start_time).total_seconds(),
            metadata={'error': 'Prediction failed and fallback disabled'}
        )

    async def _execute_ensemble_prediction(self, symbol: str, features: Any, strategy_name: str) -> PredictionResult:
        """アンサンブル予測の実行"""
        start_time = datetime.now()

        # 利用可能なシステムで並行予測
        prediction_tasks = []
        available_systems = list(self.system_configs.keys())

        for system_id in available_systems:
            task = self._execute_prediction(system_id, symbol, features)
            prediction_tasks.append((system_id, task))

        # 並行実行
        candidates = []
        for system_id, task in prediction_tasks:
            try:
                result = await task
                if result.prediction is not None:
                    # PredictionCandidateに変換
                    from prediction_strategies import create_prediction_candidate
                    candidate = create_prediction_candidate(
                        result.prediction, system_id, result.confidence, result.response_time
                    )
                    metrics = self.system_metrics.get(system_id, SystemMetrics())
                    candidate.accuracy_history = metrics.accuracy
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"システム {system_id} での予測に失敗: {e}")

        if not candidates:
            logger.error(f"アンサンブル予測: 全システムで予測に失敗 {symbol}")
            return PredictionResult(
                prediction=None,
                system_used="ensemble",
                confidence=0.0,
                response_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': 'All systems failed'}
            )

        # 予測戦略の適用
        try:
            from prediction_strategies import strategy_manager
            selected_candidate = await strategy_manager.execute_strategy(strategy_name, candidates)

            response_time = (datetime.now() - start_time).total_seconds()

            # 結果の記録
            await self._record_prediction_success(
                symbol, selected_candidate.system_id, response_time,
                selected_candidate.confidence
            )
            await self._update_system_metrics(
                selected_candidate.system_id, response_time, True
            )

            logger.info(f"アンサンブル予測成功: {symbol} using {strategy_name} -> {selected_candidate.system_id}")

            return PredictionResult(
                prediction=selected_candidate.prediction,
                system_used=f"ensemble_{selected_candidate.system_id}",
                confidence=selected_candidate.confidence,
                response_time=response_time,
                metadata={
                    'strategy': strategy_name,
                    'candidates_count': len(candidates),
                    'selected_system': selected_candidate.system_id
                }
            )

        except Exception as e:
            logger.error(f"予測戦略の実行に失敗: {e}")
            # フォールバック: 最高信頼度の候補を選択
            best_candidate = max(candidates, key=lambda c: c.confidence)

            return PredictionResult(
                prediction=best_candidate.prediction,
                system_used=f"ensemble_{best_candidate.system_id}",
                confidence=best_candidate.confidence,
                response_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': f'Strategy failed, used best candidate: {best_candidate.system_id}'}
            )

    async def _execute_prediction(self, system_id: str, symbol: str, features: Any) -> PredictionResult:
        """指定システムでの予測実行"""
        start_time = datetime.now()

        try:
            system_instance = await self._get_system_instance(system_id)
            if not system_instance:
                return PredictionResult(prediction=None, system_used=system_id)

            config = self.system_configs[system_id]

            # システムに応じた予測メソッド呼び出し
            if hasattr(system_instance, 'predict'):
                if features is not None:
                    prediction = await system_instance.predict(symbol, features)
                else:
                    prediction = await system_instance.predict(symbol)
            elif hasattr(system_instance, 'predict_with_advanced_models'):
                prediction = await system_instance.predict_with_advanced_models(symbol)
            else:
                logger.error(f"システム {system_id} に適切な予測メソッドが見つかりません")
                return PredictionResult(prediction=None, system_used=system_id)

            response_time = (datetime.now() - start_time).total_seconds()

            # 信頼度の算出（予測結果に含まれていない場合）
            confidence = self._calculate_confidence(prediction, system_id)

            return PredictionResult(
                prediction=prediction,
                system_used=system_id,
                confidence=confidence,
                response_time=response_time,
                metadata={'method': 'direct_prediction'}
            )

        except Exception as e:
            logger.error(f"システム {system_id} での予測実行エラー: {e}")
            return PredictionResult(
                prediction=None,
                system_used=system_id,
                response_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )

    async def _execute_fallback(self, symbol: str, features: Any, failed_system: str) -> PredictionResult:
        """フォールバック予測の実行"""
        fallback_order = self.config.get('fallback', {}).get('fallback_order', ['original_system'])

        for fallback_system in fallback_order:
            if fallback_system == failed_system:
                continue  # 失敗したシステムはスキップ

            if fallback_system not in self.system_configs:
                continue  # 利用できないシステムはスキップ

            logger.info(f"フォールバックシステムを試行: {fallback_system}")

            # システム選択履歴の記録
            await self._record_system_selection(symbol, fallback_system, True, f"fallback_from_{failed_system}")

            result = await self._execute_prediction(fallback_system, symbol, features)

            if result.prediction is not None:
                logger.info(f"フォールバック成功: {symbol} using {fallback_system}")
                result.metadata['fallback'] = True
                result.metadata['original_system'] = failed_system
                return result

            # リトライ設定があれば適用
            max_retries = self.config.get('fallback', {}).get('max_retries', 0)
            if max_retries > 0:
                await asyncio.sleep(self.config.get('fallback', {}).get('retry_delay_seconds', 1))

        # すべてのフォールバックが失敗
        logger.error(f"すべてのフォールバックが失敗: {symbol}")
        return PredictionResult(
            prediction=None,
            system_used=failed_system,
            confidence=0.0,
            metadata={'error': 'All fallback systems failed'}
        )

    def _calculate_confidence(self, prediction: Any, system_id: str) -> float:
        """予測信頼度の算出"""
        # 予測結果が辞書形式で信頼度を含む場合
        if isinstance(prediction, dict) and 'confidence' in prediction:
            return float(prediction['confidence'])

        # システム別のデフォルト信頼度
        system_config = self.system_configs.get(system_id)
        if system_config:
            expected_accuracy = system_config.expected_accuracy
            return expected_accuracy / 100.0

        # デフォルト信頼度
        return 0.7

    async def _record_system_selection(self, symbol: str, system_id: str,
                                     is_fallback: bool, reason: str):
        """システム選択履歴の記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_selection_history
                    (symbol, selected_system, fallback_used, selection_reason, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (symbol, system_id, is_fallback, reason, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"システム選択履歴の記録に失敗: {e}")

    async def _record_prediction_success(self, symbol: str, system_id: str,
                                       response_time: float, confidence: float):
        """予測成功履歴の記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO prediction_history
                    (symbol, system_used, response_time, confidence, success, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (symbol, system_id, response_time, confidence, True, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"予測履歴の記録に失敗: {e}")

    async def _update_system_metrics(self, system_id: str, response_time: float, success: bool):
        """システムメトリクスの更新"""
        if system_id not in self.system_metrics:
            return

        metrics = self.system_metrics[system_id]
        metrics.prediction_count += 1
        metrics.last_updated = datetime.now()

        # レスポンス時間の移動平均
        if metrics.prediction_count == 1:
            metrics.response_time = response_time
        else:
            # 指数移動平均（α=0.1）
            metrics.response_time = 0.9 * metrics.response_time + 0.1 * response_time

        # エラー率の更新
        if not success:
            if metrics.prediction_count == 1:
                metrics.error_rate = 1.0
            else:
                metrics.error_rate = 0.9 * metrics.error_rate + 0.1
        else:
            metrics.error_rate = 0.9 * metrics.error_rate

        # データベースに記録
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics
                    (system_id, accuracy, response_time, error_rate, prediction_count, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (system_id, metrics.accuracy, metrics.response_time,
                     metrics.error_rate, metrics.prediction_count, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"システムメトリクス記録に失敗: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態の取得"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'config_path': str(self.config_path),
            'systems': {},
            'cache_status': {},
            'integration_settings': self.config.get('integration', {})
        }

        # 各システムの状態
        for system_id, config in self.system_configs.items():
            metrics = self.system_metrics.get(system_id, SystemMetrics())
            status['systems'][system_id] = {
                'name': config.name,
                'description': config.description,
                'available': system_id in self.system_instances,
                'metrics': {
                    'accuracy': metrics.accuracy,
                    'response_time': metrics.response_time,
                    'error_rate': metrics.error_rate,
                    'prediction_count': metrics.prediction_count,
                    'last_updated': metrics.last_updated.isoformat() if metrics.last_updated else None
                }
            }

        # キャッシュ状態
        if self.model_cache:
            status['cache_status'] = {
                'enabled': True,
                'cached_models': len(self.model_cache.cache),
                'max_size': self.model_cache.max_size,
                'ttl_minutes': self.model_cache.ttl_minutes
            }
        else:
            status['cache_status'] = {'enabled': False}

        return status

    def reload_configuration(self):
        """設定の再読み込み"""
        logger.info("設定ファイルを再読み込みします")
        self._load_configuration()
        # キャッシュもクリアして再初期化
        if self.model_cache:
            self.model_cache.clear()
        self._init_cache()


# ユーティリティ関数
def create_integrated_prediction_system(config_path: Optional[str] = None) -> IntegratedPredictionSystem:
    """IntegratedPredictionSystemインスタンスの作成"""
    path = Path(config_path) if config_path else None
    return IntegratedPredictionSystem(config_path=path)


if __name__ == "__main__":
    # 基本的な動作確認
    async def main():
        logger.info("Integrated Prediction System テスト開始")

        system = create_integrated_prediction_system()

        # システム状態確認
        status = system.get_system_status()
        logger.info(f"システム状態: {len(status['systems'])}システム利用可能")

        # テスト予測（ダミーデータ）
        test_symbol = "7203"
        try:
            result = await system.predict(test_symbol)
            logger.info(f"予測テスト結果: {test_symbol} -> {result.system_used}")
        except Exception as e:
            logger.error(f"予測テストエラー: {e}")

    # テスト実行
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
