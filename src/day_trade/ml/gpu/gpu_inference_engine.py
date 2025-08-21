"""
GPU 加速推論エンジン (メイン統合)

gpu_accelerated_inference.py からのリファクタリング抽出
全GPU推論機能を統合したメインエンジンクラス
"""

import time
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .gpu_config import GPUBackend, GPUInferenceConfig, GPUInferenceResult
from .gpu_device_manager import GPUDeviceManager, GPUMonitoringData
from .gpu_inference_session import GPUInferenceSession
from .gpu_stream_batch import GPUStreamManager, GPUBatchProcessor

# PyNVML (NVIDIA監視ライブラリ)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("PyNVML not available - GPU monitoring limited", stacklevel=2)

# UnifiedCacheManager
try:
    from day_trade.core.cache_manager import UnifiedCacheManager
except ImportError:
    UnifiedCacheManager = None
    warnings.warn("UnifiedCacheManager not available - caching disabled", stacklevel=2)

# MicrosecondTimer導入
try:
    from day_trade.utils.timer import MicrosecondTimer
except ImportError:
    # フォールバック実装
    class MicrosecondTimer:
        @staticmethod
        def now_ns():
            return time.time_ns()

        @staticmethod
        def elapsed_us(start_ns):
            return (time.time_ns() - start_ns) // 1000

# ロギング設定
import logging
logger = logging.getLogger(__name__)


class GPUAcceleratedInferenceEngine:
    """GPU加速推論エンジン（メイン）"""

    def __init__(self, config: GPUInferenceConfig = None):
        self.config = config or GPUInferenceConfig()

        # コンポーネント初期化
        self.device_manager = GPUDeviceManager()
        self.stream_manager = GPUStreamManager(self.config)
        self.batch_processor = GPUBatchProcessor(self.config)

        # セッション管理
        self.sessions: Dict[str, GPUInferenceSession] = {}
        self.session_device_mapping: Dict[str, int] = {}

        # GPU監視機能の初期化
        self.monitoring_enabled = self.config.enable_realtime_monitoring
        self.monitoring_data_history: Dict[int, List[GPUMonitoringData]] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_stop_event = threading.Event()

        # 監視データのバッファサイズ（最新N件を保持）
        self.monitoring_history_size = 100

        # キャッシュシステム統合
        try:
            if UnifiedCacheManager:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=100, l2_memory_mb=200, l3_disk_mb=1000
                )
            else:
                self.cache_manager = None
        except Exception as e:
            logger.warning(f"キャッシュマネージャー初期化失敗: {e}")
            self.cache_manager = None

        # エンジン統計
        self.engine_stats = {
            "models_loaded": 0,
            "total_gpu_inferences": 0,
            "total_gpu_time_us": 0,
            "avg_gpu_time_us": 0.0,
            "total_gpu_memory_mb": 0.0,
            "peak_gpu_utilization": 0.0,
        }

        logger.info(
            f"GPU加速推論エンジン初期化完了: {len(self.device_manager.available_devices)} GPU"
        )

    async def load_model(
        self, model_path: str, model_name: str, device_id: Optional[int] = None
    ) -> bool:
        """モデル読み込み"""
        try:
            # デバイス選択
            if device_id is None:
                optimal_device = self.device_manager.get_optimal_device()
                device_id = optimal_device["id"]

            # セッション作成
            session = GPUInferenceSession(
                model_path, self.config, device_id, model_name
            )

            if session.is_initialized:
                self.sessions[model_name] = session
                self.session_device_mapping[model_name] = device_id
                self.engine_stats["models_loaded"] += 1

                logger.info(
                    f"GPU モデル読み込み完了: {model_name} (デバイス {device_id})"
                )
                return True
            else:
                logger.error(f"GPU モデル読み込み失敗: {model_name}")
                return False

        except Exception as e:
            logger.error(f"GPU モデル読み込みエラー: {model_name}, {e}")
            return False

    async def predict_gpu(
        self,
        model_name: str,
        input_data: np.ndarray,
        use_cache: bool = True,
        stream_id: Optional[int] = None,
    ) -> GPUInferenceResult:
        """GPU推論実行"""
        start_time = MicrosecondTimer.now_ns()

        try:
            # キャッシュチェック
            cache_key = None
            if use_cache and self.cache_manager:
                cache_key = f"gpu_{model_name}_{hash(input_data.tobytes())}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    result = GPUInferenceResult(**cached_result)
                    result.cache_hit = True
                    result.execution_time_us = MicrosecondTimer.elapsed_us(start_time)
                    return result

            # モデル取得
            if model_name not in self.sessions:
                raise ValueError(f"GPU モデルが読み込まれていません: {model_name}")

            session = self.sessions[model_name]
            device_id = self.session_device_mapping[model_name]

            # 動的バッチング
            if self.config.dynamic_batching:
                result = await self.batch_processor.add_inference_request(
                    device_id, input_data, lambda data: session.predict_gpu(data)
                )
            else:
                result = await session.predict_gpu(input_data)

            # キャッシュ保存
            if use_cache and self.cache_manager and cache_key:
                self.cache_manager.put(
                    cache_key,
                    result.to_dict(),
                    priority=8.0,  # GPU結果は高優先度
                )

            # エンジン統計更新
            self.engine_stats["total_gpu_inferences"] += 1
            self.engine_stats["total_gpu_time_us"] += result.execution_time_us
            self.engine_stats["avg_gpu_time_us"] = (
                self.engine_stats["total_gpu_time_us"]
                / self.engine_stats["total_gpu_inferences"]
            )
            self.engine_stats["total_gpu_memory_mb"] += result.gpu_memory_used_mb
            self.engine_stats["peak_gpu_utilization"] = max(
                self.engine_stats["peak_gpu_utilization"],
                result.gpu_utilization_percent,
            )

            return result

        except Exception as e:
            execution_time = MicrosecondTimer.elapsed_us(start_time)
            logger.error(f"GPU推論実行エラー: {model_name}, {e}")

            # エラー時フォールバック結果
            return GPUInferenceResult(
                predictions=np.zeros((input_data.shape[0], 1)),
                execution_time_us=execution_time,
                batch_size=input_data.shape[0],
                device_id=-1,
                backend_used=GPUBackend.CPU_FALLBACK,
                model_name=model_name,
                input_shape=input_data.shape,
            )

    async def predict_multi_gpu(
        self, requests: List[Tuple[str, np.ndarray]]
    ) -> List[GPUInferenceResult]:
        """複数GPU並列推論"""
        if len(requests) <= 1:
            if requests:
                model_name, input_data = requests[0]
                return [await self.predict_gpu(model_name, input_data)]
            else:
                return []

        # デバイス別にリクエスト分散
        device_groups = {}
        for i, (model_name, input_data) in enumerate(requests):
            device_id = self.session_device_mapping.get(model_name, 0)
            if device_id not in device_groups:
                device_groups[device_id] = []
            device_groups[device_id].append((i, model_name, input_data))

        # 並列実行
        import asyncio
        tasks = []
        for device_id, group_requests in device_groups.items():
            for idx, model_name, input_data in group_requests:
                task = asyncio.create_task(self.predict_gpu(model_name, input_data))
                tasks.append((idx, task))

        # 結果収集
        results = [None] * len(requests)
        for idx, task in tasks:
            try:
                results[idx] = await task
            except Exception as e:
                logger.error(f"マルチGPU推論エラー (インデックス {idx}): {e}")
                # フォールバック結果
                model_name, input_data = requests[idx]
                results[idx] = GPUInferenceResult(
                    predictions=np.zeros((input_data.shape[0], 1)),
                    execution_time_us=0,
                    batch_size=input_data.shape[0],
                    device_id=-1,
                    backend_used=GPUBackend.CPU_FALLBACK,
                    model_name=model_name,
                    input_shape=input_data.shape,
                )

        return results

    async def benchmark_gpu_performance(
        self, model_name: str, test_data: np.ndarray, iterations: int = 100
    ) -> Dict[str, Any]:
        """GPU推論ベンチマーク"""
        logger.info(f"GPU推論ベンチマーク開始: {model_name}, {iterations}回")

        # ウォームアップ
        for _ in range(5):
            await self.predict_gpu(model_name, test_data, use_cache=False)

        # ベンチマーク実行
        times = []
        gpu_memory_usage = []
        gpu_utilizations = []

        for i in range(iterations):
            result = await self.predict_gpu(model_name, test_data, use_cache=False)

            times.append(result.execution_time_us)
            gpu_memory_usage.append(result.gpu_memory_used_mb)
            gpu_utilizations.append(result.gpu_utilization_percent)

        # 統計計算
        times_array = np.array(times)
        memory_array = np.array(gpu_memory_usage)
        utilization_array = np.array(gpu_utilizations)

        benchmark_results = {
            "model_name": model_name,
            "device_id": self.session_device_mapping.get(model_name, -1),
            "backend": self.config.backend.value,
            "iterations": iterations,
            "test_data_shape": test_data.shape,
            # 実行時間統計
            "avg_time_us": np.mean(times_array),
            "min_time_us": np.min(times_array),
            "max_time_us": np.max(times_array),
            "std_time_us": np.std(times_array),
            "median_time_us": np.median(times_array),
            "p95_time_us": np.percentile(times_array, 95),
            "p99_time_us": np.percentile(times_array, 99),
            # スループット
            "throughput_inferences_per_sec": 1_000_000 / np.mean(times_array),
            "throughput_samples_per_sec": (1_000_000 / np.mean(times_array))
            * test_data.shape[0],
            # GPU使用統計
            "avg_gpu_memory_mb": np.mean(memory_array),
            "peak_gpu_memory_mb": np.max(memory_array),
            "avg_gpu_utilization": np.mean(utilization_array),
            "peak_gpu_utilization": np.max(utilization_array),
        }

        logger.info(
            f"GPU ベンチマーク完了 - 平均: {benchmark_results['avg_time_us']:.1f}μs, "
            f"スループット: {benchmark_results['throughput_inferences_per_sec']:.0f}/秒, "
            f"GPU使用率: {benchmark_results['avg_gpu_utilization']:.1f}%"
        )

        return benchmark_results

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """総合統計取得"""
        stats = self.engine_stats.copy()

        # デバイス情報
        stats["devices"] = self.device_manager.get_device_summary()

        # セッション統計
        session_stats = {}
        for name, session in self.sessions.items():
            session_stats[name] = session.get_session_stats()
        stats["sessions"] = session_stats

        # ストリーム統計
        stats["stream_stats"] = self.stream_manager.get_stream_stats()

        # バッチ処理統計
        stats["batch_stats"] = self.batch_processor.get_batch_stats()

        # キャッシュ統計
        if self.cache_manager:
            cache_stats = self.cache_manager.get_comprehensive_stats()
            stats["cache_stats"] = cache_stats

        # 設定情報
        stats["config"] = self.config.to_dict()

        return stats

    # GPU監視機能メソッド

    def start_realtime_monitoring(self):
        """リアルタイムGPU監視開始"""
        if not self.monitoring_enabled:
            logger.info("GPU監視が無効化されています")
            return

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("GPU監視は既に実行中です")
            return

        logger.info("リアルタイムGPU監視を開始")
        self.monitoring_stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="GPU-Monitor",
            daemon=True
        )
        self.monitoring_thread.start()

    def stop_realtime_monitoring(self):
        """リアルタイムGPU監視停止"""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            return

        logger.info("リアルタイムGPU監視を停止")
        self.monitoring_stop_event.set()
        self.monitoring_thread.join(timeout=5.0)

        if self.monitoring_thread.is_alive():
            logger.warning("GPU監視スレッド停止タイムアウト")

    def _monitoring_worker(self):
        """GPU監視ワーカースレッド"""
        logger.debug("GPU監視ワーカースレッド開始")

        # 各デバイスの監視データ履歴初期化
        for device_id in self.config.device_ids:
            self.monitoring_data_history[device_id] = []

        interval_seconds = self.config.monitoring_interval_ms / 1000.0

        while not self.monitoring_stop_event.wait(interval_seconds):
            try:
                # 各GPUデバイスの監視データ収集
                for device_id in self.config.device_ids:
                    # セッションからGPU監視データ取得
                    monitoring_data = self._collect_device_monitoring_data(device_id)

                    if monitoring_data:
                        # 履歴に追加
                        self.monitoring_data_history[device_id].append(monitoring_data)

                        # 履歴サイズ制限
                        if len(self.monitoring_data_history[device_id]) > self.monitoring_history_size:
                            self.monitoring_data_history[device_id].pop(0)

                        # 健全性チェック
                        health_status = self._check_device_health(device_id, monitoring_data)

                        # 警告・アラートのログ出力
                        self._handle_monitoring_alerts(device_id, health_status)

            except Exception as e:
                logger.error(f"GPU監視ワーカーエラー: {e}")

        logger.debug("GPU監視ワーカースレッド終了")

    def _collect_device_monitoring_data(self, device_id: int) -> Optional[GPUMonitoringData]:
        """指定デバイスの監視データ収集"""
        try:
            # デバイス用のセッションを探す
            session = None
            for model_name, sess in self.sessions.items():
                if self.session_device_mapping.get(model_name) == device_id:
                    session = sess
                    break

            if session:
                return session.get_comprehensive_gpu_monitoring()
            else:
                # セッションがない場合はデバイスマネージャーから基本的な監視データ
                return self.device_manager.collect_monitoring_data(device_id)

        except Exception as e:
            logger.debug(f"デバイス{device_id}の監視データ収集エラー: {e}")
            return None

    def _check_device_health(self, device_id: int, monitoring_data: GPUMonitoringData) -> Dict[str, Any]:
        """デバイス健全性チェック"""
        health_status = {
            "device_id": device_id,
            "is_healthy": monitoring_data.is_healthy,
            "is_overloaded": monitoring_data.is_overloaded,
            "warnings": [],
            "critical_alerts": []
        }

        # 警告レベルチェック
        if monitoring_data.gpu_utilization_percent > self.config.gpu_utilization_threshold:
            health_status["warnings"].append(
                f"GPU使用率が閾値を超過: {monitoring_data.gpu_utilization_percent:.1f}% > {self.config.gpu_utilization_threshold}%"
            )

        if monitoring_data.memory_utilization_percent > self.config.gpu_memory_threshold:
            health_status["warnings"].append(
                f"GPUメモリ使用率が閾値を超過: {monitoring_data.memory_utilization_percent:.1f}% > {self.config.gpu_memory_threshold}%"
            )

        if monitoring_data.temperature_celsius > self.config.temperature_threshold:
            health_status["warnings"].append(
                f"GPU温度が閾値を超過: {monitoring_data.temperature_celsius:.1f}°C > {self.config.temperature_threshold}°C"
            )

        if monitoring_data.power_consumption_watts > self.config.power_threshold:
            health_status["warnings"].append(
                f"GPU電力消費が閾値を超過: {monitoring_data.power_consumption_watts:.1f}W > {self.config.power_threshold}W"
            )

        # クリティカルアラート
        if monitoring_data.gpu_utilization_percent > 98.0:
            health_status["critical_alerts"].append("GPU使用率が限界レベル (>98%)")

        if monitoring_data.memory_utilization_percent > 98.0:
            health_status["critical_alerts"].append("GPUメモリ使用率が限界レベル (>98%)")

        if monitoring_data.temperature_celsius > 90.0:
            health_status["critical_alerts"].append("GPU温度が危険レベル (>90°C)")

        if monitoring_data.has_errors:
            health_status["critical_alerts"].append(f"GPU監視エラー: {monitoring_data.error_message}")

        return health_status

    def _handle_monitoring_alerts(self, device_id: int, health_status: Dict[str, Any]):
        """監視アラートの処理"""
        # 警告ログ出力
        for warning in health_status["warnings"]:
            logger.warning(f"GPU {device_id} 警告: {warning}")

        # クリティカルアラートログ出力
        for alert in health_status["critical_alerts"]:
            logger.error(f"GPU {device_id} クリティカル: {alert}")

    def get_monitoring_data(self, device_id: Optional[int] = None) -> Dict[int, List[GPUMonitoringData]]:
        """監視データ履歴取得"""
        if device_id is not None:
            return {device_id: self.monitoring_data_history.get(device_id, [])}
        return self.monitoring_data_history.copy()

    def get_latest_monitoring_snapshot(self) -> Dict[str, Any]:
        """最新の監視データスナップショット取得"""
        snapshot = {
            "timestamp": time.time(),
            "devices": {},
            "summary": {
                "total_devices": len(self.config.device_ids),
                "healthy_devices": 0,
                "overloaded_devices": 0,
                "devices_with_errors": 0,
                "avg_gpu_utilization": 0.0,
                "avg_memory_utilization": 0.0,
                "avg_temperature": 0.0,
                "total_power_consumption": 0.0
            }
        }

        gpu_utils = []
        memory_utils = []
        temperatures = []
        power_consumptions = []

        for device_id in self.config.device_ids:
            history = self.monitoring_data_history.get(device_id, [])
            if history:
                latest_data = history[-1]
                snapshot["devices"][device_id] = latest_data.to_dict()

                # サマリ統計用
                gpu_utils.append(latest_data.gpu_utilization_percent)
                memory_utils.append(latest_data.memory_utilization_percent)
                temperatures.append(latest_data.temperature_celsius)
                power_consumptions.append(latest_data.power_consumption_watts)

                if latest_data.is_healthy:
                    snapshot["summary"]["healthy_devices"] += 1
                if latest_data.is_overloaded:
                    snapshot["summary"]["overloaded_devices"] += 1
                if latest_data.has_errors:
                    snapshot["summary"]["devices_with_errors"] += 1

        # サマリ統計計算
        if gpu_utils:
            snapshot["summary"]["avg_gpu_utilization"] = np.mean(gpu_utils)
            snapshot["summary"]["avg_memory_utilization"] = np.mean(memory_utils)
            snapshot["summary"]["avg_temperature"] = np.mean(temperatures)
            snapshot["summary"]["total_power_consumption"] = np.sum(power_consumptions)

        return snapshot

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # GPU監視停止
            self.stop_realtime_monitoring()

            # ストリーム同期
            self.stream_manager.synchronize_all_streams()

            # バッチプロセッサーのクリーンアップ
            self.batch_processor.cleanup()

            # ストリームマネージャーのクリーンアップ
            self.stream_manager.cleanup()

            # セッション削除
            for session in self.sessions.values():
                session.cleanup()
            self.sessions.clear()

            # キャッシュクリア
            if self.cache_manager:
                self.cache_manager.clear_all()

            logger.info("GPU推論エンジン クリーンアップ完了")

        except Exception as e:
            logger.error(f"GPU推論エンジン クリーンアップエラー: {e}")

    @property
    def is_ready(self) -> bool:
        """エンジン準備完了状態"""
        return len(self.sessions) > 0

    @property
    def loaded_models(self) -> List[str]:
        """読み込み済みモデル一覧"""
        return list(self.sessions.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """モデル情報取得"""
        if model_name not in self.sessions:
            return None

        session = self.sessions[model_name]
        return {
            "model_name": model_name,
            "device_id": self.session_device_mapping.get(model_name, -1),
            "input_info": session.input_info,
            "session_stats": session.get_session_stats(),
            "tensorrt_enabled": session.use_tensorrt,
            "is_initialized": session.is_initialized
        }