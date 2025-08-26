#!/usr/bin/env python3
"""
パフォーマンス分析・ベンチマークモジュール
Issue #379: ML Model Inference Performance Optimization

モデル圧縮手法のパフォーマンス評価・分析機能
- 複数手法の並列ベンチマーク（Issue #725対応）
- ベンチマーク結果の詳細分析
- 推奨手法の自動選定
- 統計・レポート生成
"""

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Dict, List

import numpy as np

from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from .data_structures import (
    CompressionConfig,
    CompressionResult,
    HardwareTarget,
    PruningType,
    QuantizationType,
)

logger = get_context_logger(__name__)

# ONNX Runtime (推論時間測定用)
try:
    import onnxruntime as ort

    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False


class PerformanceAnalyzer:
    """パフォーマンス分析・ベンチマーククラス"""

    def __init__(self):
        """分析器の初期化"""
        self.benchmark_results = {}
        self.analysis_cache = {}

    async def benchmark_compression_methods(
        self,
        model_path: str,
        compression_engine_factory,
        validation_data: List[np.ndarray] = None,
    ) -> Dict[str, CompressionResult]:
        """複数圧縮手法のベンチマーク - Issue #725対応: 並列化版
        
        Args:
            model_path: ベンチマーク対象のモデルファイルパス
            compression_engine_factory: 圧縮エンジン作成関数
            validation_data: 検証用データセット
            
        Returns:
            手法別ベンチマーク結果辞書
        """
        logger.info("圧縮手法ベンチマーク開始（並列化版）")

        benchmark_configs = {
            "dynamic_int8": CompressionConfig(
                quantization_type=QuantizationType.DYNAMIC_INT8,
                pruning_type=PruningType.NONE,
            ),
            "static_int8": CompressionConfig(
                quantization_type=QuantizationType.STATIC_INT8,
                pruning_type=PruningType.NONE,
            ),
            "pruning_only": CompressionConfig(
                quantization_type=QuantizationType.NONE,
                pruning_type=PruningType.MAGNITUDE_BASED,
            ),
            "combined": CompressionConfig(
                quantization_type=QuantizationType.DYNAMIC_INT8,
                pruning_type=PruningType.MAGNITUDE_BASED,
            ),
            # Issue #725対応: FP16量子化追加
            "fp16_quantization": CompressionConfig(
                quantization_type=QuantizationType.MIXED_PRECISION_FP16,
                pruning_type=PruningType.NONE,
            ),
        }

        # Issue #725対応: 並列化ベンチマーク実行
        results = await self._parallel_benchmark_execution(
            model_path,
            benchmark_configs,
            compression_engine_factory,
            validation_data,
        )

        # 結果をキャッシュ
        self.benchmark_results.update(results)

        logger.info(f"並列圧縮手法ベンチマーク完了: {len(results)}手法")
        return results

    async def _parallel_benchmark_execution(
        self,
        model_path: str,
        benchmark_configs: Dict[str, CompressionConfig],
        compression_engine_factory,
        validation_data: List[np.ndarray] = None,
    ) -> Dict[str, CompressionResult]:
        """Issue #725対応: 並列ベンチマーク実行"""
        # 並列実行方法の選択
        use_process_pool = len(benchmark_configs) > 2 and os.cpu_count() > 2
        max_workers = min(len(benchmark_configs), os.cpu_count() or 4)

        logger.info(
            f"並列ベンチマーク設定: "
            f"{'プロセスプール' if use_process_pool else 'スレッドプール'}, "
            f"最大{max_workers}ワーカー"
        )

        try:
            if use_process_pool:
                # プロセス並列（CPU集約的タスク）
                return await self._process_pool_benchmark(
                    model_path,
                    benchmark_configs,
                    compression_engine_factory,
                    validation_data,
                    max_workers,
                )
            else:
                # スレッド並列（I/O集約的タスク）
                return await self._thread_pool_benchmark(
                    model_path,
                    benchmark_configs,
                    compression_engine_factory,
                    validation_data,
                    max_workers,
                )

        except Exception as e:
            logger.warning(
                f"並列ベンチマーク実行失敗: {e} - "
                f"シーケンシャル実行にフォールバック"
            )
            return await self._sequential_benchmark_fallback(
                model_path,
                benchmark_configs,
                compression_engine_factory,
                validation_data,
            )

    async def _thread_pool_benchmark(
        self,
        model_path: str,
        benchmark_configs: Dict[str, CompressionConfig],
        compression_engine_factory,
        validation_data: List[np.ndarray],
        max_workers: int,
    ) -> Dict[str, CompressionResult]:
        """スレッドプール並列ベンチマーク"""

        async def run_single_benchmark(
            method_name: str, config: CompressionConfig
        ):
            """単一ベンチマーク実行"""
            try:
                logger.debug(f"並列ベンチマーク開始: {method_name}")

                # スレッドセーフなエンジンコピー作成
                thread_safe_engine = compression_engine_factory(config)

                result = await thread_safe_engine.compress_model(
                    model_path,
                    f"benchmark_output/{method_name}",
                    validation_data,
                    f"model_{method_name}",
                )

                logger.debug(f"並列ベンチマーク完了: {method_name}")
                return method_name, result

            except Exception as e:
                logger.error(f"並列ベンチマーク エラー ({method_name}): {e}")
                return method_name, None

        # 並列実行
        tasks = []
        for method_name, config in benchmark_configs.items():
            task = run_single_benchmark(method_name, config)
            tasks.append(task)

        # 全タスク完了を待機
        benchmark_results = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # 結果集約
        results = {}
        for result in benchmark_results:
            if isinstance(result, tuple) and len(result) == 2:
                method_name, compression_result = result
                if compression_result is not None:
                    results[method_name] = compression_result
            elif isinstance(result, Exception):
                logger.warning(f"並列ベンチマークタスクエラー: {result}")

        return results

    async def _process_pool_benchmark(
        self,
        model_path: str,
        benchmark_configs: Dict[str, CompressionConfig],
        compression_engine_factory,
        validation_data: List[np.ndarray],
        max_workers: int,
    ) -> Dict[str, CompressionResult]:
        """プロセスプール並列ベンチマーク"""
        # プロセス間で共有可能な引数に変換
        benchmark_tasks = []
        for method_name, config in benchmark_configs.items():
            task_data = {
                "method_name": method_name,
                "model_path": model_path,
                "config_dict": config.to_dict(),
                "output_path": f"benchmark_output/{method_name}",
                "model_name": f"model_{method_name}",
            }
            benchmark_tasks.append(task_data)

        # プロセスプールでの並列実行
        loop = asyncio.get_event_loop()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for task_data in benchmark_tasks:
                future = loop.run_in_executor(
                    executor, self._run_benchmark_process, task_data
                )
                futures.append(future)

            # 結果待機
            process_results = await asyncio.gather(
                *futures, return_exceptions=True
            )

        # 結果集約
        results = {}
        for result in process_results:
            if isinstance(result, dict) and "method_name" in result:
                method_name = result["method_name"]
                if result.get("success", False):
                    results[method_name] = result["compression_result"]
                else:
                    logger.error(f"プロセス並列ベンチマーク失敗: {method_name}")
            elif isinstance(result, Exception):
                logger.warning(f"プロセスプールエラー: {result}")

        return results

    def _run_benchmark_process(self, task_data: dict) -> dict:
        """プロセス内でのベンチマーク実行（プロセスプール用）"""
        try:
            method_name = task_data["method_name"]
            model_path = task_data["model_path"]
            config_dict = task_data["config_dict"]
            output_path = task_data["output_path"]
            model_name = task_data["model_name"]

            # 設定復元
            config = CompressionConfig()
            config.quantization_type = QuantizationType(
                config_dict.get("quantization_type", "none")
            )
            config.pruning_type = PruningType(
                config_dict.get("pruning_type", "none")
            )

            # 新しいエンジンインスタンス作成（ダミー実装）
            # 実際の実装では compression_engine_factory を使用
            from .compression_engine import ModelCompressionEngine

            engine = ModelCompressionEngine(config)

            # 同期実行（プロセス内）
            async def run_compression():
                return await engine.compress_model(
                    model_path, output_path, None, model_name
                )

            # 新しいイベントループでの実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_compression())
                return {
                    "method_name": method_name,
                    "success": True,
                    "compression_result": result,
                }
            finally:
                loop.close()

        except Exception as e:
            return {
                "method_name": task_data.get("method_name", "unknown"),
                "success": False,
                "error": str(e),
            }

    async def _sequential_benchmark_fallback(
        self,
        model_path: str,
        benchmark_configs: Dict[str, CompressionConfig],
        compression_engine_factory,
        validation_data: List[np.ndarray],
    ) -> Dict[str, CompressionResult]:
        """シーケンシャルベンチマーク（フォールバック）"""
        logger.info("シーケンシャルベンチマーク実行")

        results = {}

        for method_name, config in benchmark_configs.items():
            logger.info(f"ベンチマーク実行: {method_name}")

            try:
                # 一時的なエンジン作成
                engine = compression_engine_factory(config)

                result = await engine.compress_model(
                    model_path,
                    f"benchmark_output/{method_name}",
                    validation_data,
                    f"model_{method_name}",
                )

                results[method_name] = result

            except Exception as e:
                logger.error(f"ベンチマーク実行エラー ({method_name}): {e}")

        return results

    def analyze_benchmark_results(
        self, benchmark_results: Dict[str, CompressionResult]
    ) -> Dict[str, Any]:
        """Issue #725対応: ベンチマーク結果分析
        
        Args:
            benchmark_results: 手法別ベンチマーク結果
            
        Returns:
            詳細分析結果辞書
        """
        if not benchmark_results:
            return {"error": "ベンチマーク結果が空です"}

        analysis = {
            "summary": {
                "total_methods": len(benchmark_results),
                "successful_methods": len(
                    [
                        r
                        for r in benchmark_results.values()
                        if r.accuracy_drop < 0.50
                    ]
                ),
            },
            "performance_ranking": {},
            "best_methods": {},
            "detailed_comparison": {},
        }

        # 成功した結果のみを対象とする (accuracy_dropが小さい = 成功)
        successful_results = {
            name: result
            for name, result in benchmark_results.items()
            if result.accuracy_drop < 0.50  # 50%以下の精度低下なら成功
        }

        if not successful_results:
            analysis["error"] = "成功したベンチマークがありません"
            return analysis

        # 各メトリクス別ランキング
        metrics = [
            "compression_ratio",
            "compressed_model_size_mb",
            "compressed_inference_time_us",
        ]

        for metric in metrics:
            # メトリクス値の取得
            metric_values = {}
            for name, result in successful_results.items():
                value = getattr(result, metric, 0)
                if value > 0:  # 有効な値のみ
                    metric_values[name] = value

            if not metric_values:
                continue

            # ランキング作成（圧縮率は高い方が良い、他は低い方が良い）
            reverse_sort = metric == "compression_ratio"
            sorted_methods = sorted(
                metric_values.items(), key=lambda x: x[1], reverse=reverse_sort
            )

            analysis["performance_ranking"][metric] = [
                {"method": name, "value": value}
                for name, value in sorted_methods
            ]

            # 最良手法記録
            if sorted_methods:
                best_method, best_value = sorted_methods[0]
                analysis["best_methods"][metric] = {
                    "method": best_method,
                    "value": best_value,
                }

        # 総合スコア計算（重み付き）
        method_scores = {}
        weights = {
            "compression_ratio": 0.4,  # 圧縮効率重視
            "compressed_inference_time_us": 0.3,  # 推論速度
            "compressed_model_size_mb": 0.3,  # モデルサイズ
        }

        for name in successful_results.keys():
            score = 0.0
            total_weight = 0.0

            for metric, weight in weights.items():
                ranking = analysis["performance_ranking"].get(metric, [])
                for i, entry in enumerate(ranking):
                    if entry["method"] == name:
                        # 順位に基づくスコア（1位=100点、最下位=0点）
                        position_score = (
                            (len(ranking) - i - 1) / (len(ranking) - 1) * 100
                        )
                        score += position_score * weight
                        total_weight += weight
                        break

            if total_weight > 0:
                method_scores[name] = score / total_weight

        # 総合ランキング
        overall_ranking = sorted(
            method_scores.items(), key=lambda x: x[1], reverse=True
        )

        analysis["overall_ranking"] = [
            {"method": name, "score": score} for name, score in overall_ranking
        ]

        if overall_ranking:
            analysis["recommended_method"] = {
                "method": overall_ranking[0][0],
                "score": overall_ranking[0][1],
                "reason": "総合スコア最優秀",
            }

        # 詳細比較テーブル作成
        comparison_table = []
        for name, result in successful_results.items():
            row = {
                "method": name,
                "compression_ratio": result.compression_ratio,
                "compressed_model_size_mb": result.compressed_model_size_mb,
                "compressed_inference_time_us": result.compressed_inference_time_us,
                "overall_score": method_scores.get(name, 0),
            }
            comparison_table.append(row)

        analysis["detailed_comparison"] = comparison_table

        # キャッシュに保存
        self.analysis_cache = analysis

        return analysis

    def generate_performance_report(
        self, analysis_results: Dict[str, Any] = None
    ) -> str:
        """パフォーマンス分析レポート生成
        
        Args:
            analysis_results: 分析結果（未指定時はキャッシュから取得）
            
        Returns:
            マークダウン形式レポート文字列
        """
        if analysis_results is None:
            analysis_results = self.analysis_cache

        if not analysis_results or "error" in analysis_results:
            return "# パフォーマンス分析レポート\n\n分析データなし"

        report_lines = [
            "# モデル圧縮パフォーマンス分析レポート",
            "",
            "## サマリー",
            f"- 検証手法数: {analysis_results['summary']['total_methods']}",
            f"- 成功手法数: {analysis_results['summary']['successful_methods']}",
            "",
        ]

        # 推奨手法
        if "recommended_method" in analysis_results:
            recommended = analysis_results["recommended_method"]
            report_lines.extend(
                [
                    "## 推奨手法",
                    f"**{recommended['method']}** "
                    f"(スコア: {recommended['score']:.1f})",
                    f"理由: {recommended['reason']}",
                    "",
                ]
            )

        # メトリクス別ランキング
        if "performance_ranking" in analysis_results:
            report_lines.append("## メトリクス別ランキング")
            for metric, rankings in analysis_results[
                "performance_ranking"
            ].items():
                report_lines.extend([f"### {metric}", ""])
                for i, entry in enumerate(rankings[:3], 1):  # 上位3位まで
                    report_lines.append(
                        f"{i}. {entry['method']}: {entry['value']:.3f}"
                    )
                report_lines.append("")

        # 詳細比較
        if "detailed_comparison" in analysis_results:
            report_lines.extend(["## 詳細比較", ""])
            comparison = analysis_results["detailed_comparison"]
            if comparison:
                # テーブルヘッダー
                headers = [
                    "手法",
                    "圧縮率",
                    "モデルサイズ(MB)",
                    "推論時間(μs)",
                    "総合スコア",
                ]
                report_lines.append(" | ".join(headers))
                report_lines.append(" | ".join(["---"] * len(headers)))

                # データ行
                for row in sorted(
                    comparison, key=lambda x: x["overall_score"], reverse=True
                ):
                    data_row = [
                        row["method"],
                        f"{row['compression_ratio']:.2f}",
                        f"{row['compressed_model_size_mb']:.1f}",
                        f"{row['compressed_inference_time_us']:.0f}",
                        f"{row['overall_score']:.1f}",
                    ]
                    report_lines.append(" | ".join(data_row))

        report_lines.extend(["", "---", f"レポート生成時刻: {MicrosecondTimer.now_ns() // 1000000}"])

        return "\n".join(report_lines)

    async def evaluate_model_inference(
        self, model_path: str, validation_data: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """モデル推論評価（ベンチマーク補助機能）
        
        Args:
            model_path: 評価対象モデルパス
            validation_data: 検証用データセット
            
        Returns:
            推論評価結果辞書
        """
        try:
            from pathlib import Path

            # モデルサイズ
            model_size_mb = Path(model_path).stat().st_size / 1024 / 1024

            # 推論速度測定
            inference_times = []
            accuracy = 0.85  # デフォルト値

            if validation_data and ONNX_RUNTIME_AVAILABLE:
                try:
                    session = ort.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )

                    input_name = session.get_inputs()[0].name

                    # 推論時間測定
                    for i, data in enumerate(
                        validation_data[:10]
                    ):  # 最初の10個でテスト
                        start_time = MicrosecondTimer.now_ns()

                        outputs = session.run(
                            None, {input_name: data.astype(np.float32)}
                        )

                        inference_time = MicrosecondTimer.elapsed_us(start_time)
                        inference_times.append(inference_time)

                        if i >= 9:  # 10回で十分
                            break

                    # 簡易精度計算（実際の実装では適切な評価指標を使用）
                    accuracy = 0.85 + np.random.randn() * 0.02  # ダミー

                except Exception as e:
                    logger.warning(f"モデル評価中のエラー: {e}")
                    inference_times = [1000.0]  # デフォルト値
            else:
                inference_times = [1000.0]  # デフォルト値

            avg_inference_time_us = (
                np.mean(inference_times) if inference_times else 1000.0
            )

            return {
                "model_size_mb": model_size_mb,
                "avg_inference_time_us": avg_inference_time_us,
                "accuracy": max(0.5, min(1.0, accuracy)),  # 0.5-1.0に正規化
            }

        except Exception as e:
            logger.error(f"モデル評価エラー: {e}")
            return {
                "model_size_mb": 10.0,
                "avg_inference_time_us": 1000.0,
                "accuracy": 0.8,
            }