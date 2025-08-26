"""
リソース管理・クリーンアップ
システムリソースの管理とクリーンアップ処理を担当
"""

import gc
from typing import Any, Dict

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ResourceManager:
    """リソース管理・クリーンアップクラス"""

    def __init__(self, core):
        """
        初期化

        Args:
            core: オーケストレーターコア
        """
        self.core = core

    def cleanup(self) -> Dict[str, Any]:
        """
        包括的リソースクリーンアップ

        Returns:
            クリーンアップサマリー
        """
        logger.info("Next-Gen AI Orchestrator リソースクリーンアップ開始")

        cleanup_summary = {
            "analysis_engines": 0,
            "batch_fetcher": False,
            "ml_engine": False,
            "parallel_manager": False,
            "performance_monitor": False,
            "stock_fetcher": False,
            "errors": []
        }

        try:
            # 分析エンジンクリーンアップ
            self._cleanup_analysis_engines(cleanup_summary)

            # MLエンジンクリーンアップ
            self._cleanup_ml_engine(cleanup_summary)

            # バッチフェッチャークリーンアップ
            self._cleanup_batch_fetcher(cleanup_summary)

            # 並列マネージャークリーンアップ
            self._cleanup_parallel_manager(cleanup_summary)

            # パフォーマンスモニタークリーンアップ
            self._cleanup_performance_monitor(cleanup_summary)

            # ストックフェッチャークリーンアップ
            self._cleanup_stock_fetcher(cleanup_summary)

            # 実行履歴クリア
            self._cleanup_execution_history()

            # ガベージコレクション実行
            self._force_garbage_collection()

            # クリーンアップサマリーログ
            self._log_cleanup_summary(cleanup_summary)

        except Exception as e:
            error_msg = f"クリーンアップ致命的エラー: {e}"
            logger.error(error_msg)
            cleanup_summary["errors"].append(error_msg)

        return cleanup_summary

    def _cleanup_analysis_engines(self, summary: Dict[str, Any]) -> None:
        """分析エンジンクリーンアップ"""
        try:
            if hasattr(self.core, 'analysis_engines') and self.core.analysis_engines:
                for symbol, engine in self.core.analysis_engines.items():
                    try:
                        # エンジン停止・クローズ処理
                        if hasattr(engine, "stop"):
                            engine.stop()
                        if hasattr(engine, "close"):
                            engine.close()
                        if hasattr(engine, "cleanup"):
                            engine.cleanup()
                        
                        summary["analysis_engines"] += 1
                        logger.debug(f"エンジン {symbol} クリーンアップ完了")
                        
                    except Exception as e:
                        error_msg = f"エンジン {symbol} クリーンアップエラー: {e}"
                        logger.warning(error_msg)
                        summary["errors"].append(error_msg)

                self.core.analysis_engines.clear()
                logger.info(f"分析エンジン {summary['analysis_engines']} 個をクリーンアップ")
                
        except Exception as e:
            error_msg = f"分析エンジンクリーンアップエラー: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)

    def _cleanup_ml_engine(self, summary: Dict[str, Any]) -> None:
        """MLエンジンクリーンアップ"""
        try:
            if hasattr(self.core, 'ml_engine') and self.core.ml_engine:
                # PyTorchモデルのクリーンアップ
                if hasattr(self.core.ml_engine, "model") and self.core.ml_engine.model:
                    # GPUからCPUに移動してメモリ解放
                    if hasattr(self.core.ml_engine.model, "cpu"):
                        self.core.ml_engine.model.cpu()
                    del self.core.ml_engine.model

                # キャッシュクリア
                if hasattr(self.core.ml_engine, "clear_cache"):
                    self.core.ml_engine.clear_cache()

                # その他のリソースクリーンアップ
                if hasattr(self.core.ml_engine, "close"):
                    self.core.ml_engine.close()
                if hasattr(self.core.ml_engine, "cleanup"):
                    self.core.ml_engine.cleanup()

                # パフォーマンス履歴クリア
                if hasattr(self.core.ml_engine, "performance_history"):
                    self.core.ml_engine.performance_history.clear()

                # 統計データクリア
                if hasattr(self.core.ml_engine, "model_metadata"):
                    self.core.ml_engine.model_metadata.clear()

                self.core.ml_engine = None
                summary["ml_engine"] = True
                logger.debug("MLエンジン クリーンアップ完了")
                
        except Exception as e:
            error_msg = f"MLエンジン クリーンアップエラー: {e}"
            logger.warning(error_msg)
            summary["errors"].append(error_msg)

    def _cleanup_batch_fetcher(self, summary: Dict[str, Any]) -> None:
        """バッチフェッチャークリーンアップ"""
        try:
            if hasattr(self.core, 'batch_fetcher') and self.core.batch_fetcher:
                # バッチフェッチャー停止
                if hasattr(self.core.batch_fetcher, "stop"):
                    self.core.batch_fetcher.stop()
                
                # コネクションプールクローズ
                if hasattr(self.core.batch_fetcher, "close_connections"):
                    self.core.batch_fetcher.close_connections()
                
                # 通常のクローズ処理
                if hasattr(self.core.batch_fetcher, "close"):
                    self.core.batch_fetcher.close()

                # キャッシュクリア
                if hasattr(self.core.batch_fetcher, "clear_cache"):
                    self.core.batch_fetcher.clear_cache()

                self.core.batch_fetcher = None
                summary["batch_fetcher"] = True
                logger.debug("バッチフェッチャー クリーンアップ完了")
                
        except Exception as e:
            error_msg = f"バッチフェッチャー クリーンアップエラー: {e}"
            logger.warning(error_msg)
            summary["errors"].append(error_msg)

    def _cleanup_parallel_manager(self, summary: Dict[str, Any]) -> None:
        """並列マネージャークリーンアップ"""
        try:
            if hasattr(self.core, 'parallel_manager') and self.core.parallel_manager:
                # 実行中タスクの停止
                if hasattr(self.core.parallel_manager, "cancel_all_tasks"):
                    self.core.parallel_manager.cancel_all_tasks()
                
                # エクゼキューターのシャットダウン
                if hasattr(self.core.parallel_manager, "shutdown"):
                    self.core.parallel_manager.shutdown(wait=True, timeout=30)

                # 統計データクリア
                if hasattr(self.core.parallel_manager, "clear_stats"):
                    self.core.parallel_manager.clear_stats()

                self.core.parallel_manager = None
                summary["parallel_manager"] = True
                logger.debug("並列マネージャー クリーンアップ完了")
                
        except Exception as e:
            error_msg = f"並列マネージャー クリーンアップエラー: {e}"
            logger.warning(error_msg)
            summary["errors"].append(error_msg)

    def _cleanup_performance_monitor(self, summary: Dict[str, Any]) -> None:
        """パフォーマンスモニタークリーンアップ"""
        try:
            if hasattr(self.core, 'performance_monitor') and self.core.performance_monitor:
                # モニタリング停止
                if hasattr(self.core.performance_monitor, "stop"):
                    self.core.performance_monitor.stop()
                
                # データクリア
                if hasattr(self.core.performance_monitor, "clear_data"):
                    self.core.performance_monitor.clear_data()
                
                # クローズ処理
                if hasattr(self.core.performance_monitor, "close"):
                    self.core.performance_monitor.close()

                self.core.performance_monitor = None
                summary["performance_monitor"] = True
                logger.debug("パフォーマンスモニター クリーンアップ完了")
                
        except Exception as e:
            error_msg = f"パフォーマンスモニター クリーンアップエラー: {e}"
            logger.warning(error_msg)
            summary["errors"].append(error_msg)

    def _cleanup_stock_fetcher(self, summary: Dict[str, Any]) -> None:
        """ストックフェッチャークリーンアップ"""
        try:
            if hasattr(self.core, 'stock_fetcher') and self.core.stock_fetcher:
                # セッションクローズ
                if hasattr(self.core.stock_fetcher, "close_session"):
                    self.core.stock_fetcher.close_session()
                
                # キャッシュクリア
                if hasattr(self.core.stock_fetcher, "clear_cache"):
                    self.core.stock_fetcher.clear_cache()
                
                # 通常のクローズ処理
                if hasattr(self.core.stock_fetcher, "close"):
                    self.core.stock_fetcher.close()

                self.core.stock_fetcher = None
                summary["stock_fetcher"] = True
                logger.debug("ストックフェッチャー クリーンアップ完了")
                
        except Exception as e:
            error_msg = f"ストックフェッチャー クリーンアップエラー: {e}"
            logger.warning(error_msg)
            summary["errors"].append(error_msg)

    def _cleanup_execution_history(self) -> None:
        """実行履歴クリア"""
        try:
            if hasattr(self.core, 'execution_history'):
                self.core.execution_history.clear()
                logger.debug("実行履歴クリア完了")
                
            if hasattr(self.core, 'performance_metrics'):
                self.core.performance_metrics.clear()
                logger.debug("パフォーマンスメトリクスクリア完了")
                
        except Exception as e:
            logger.warning(f"実行履歴クリアエラー: {e}")

    def _force_garbage_collection(self) -> None:
        """ガベージコレクション強制実行"""
        try:
            # メモリリークを防ぐため、複数回実行
            for i in range(3):
                collected = gc.collect()
                logger.debug(f"ガベージコレクション {i+1}/3: {collected} オブジェクト回収")
            
            # 循環参照の確認
            if gc.garbage:
                logger.warning(f"循環参照が検出されました: {len(gc.garbage)} オブジェクト")
                gc.garbage.clear()
                
        except Exception as e:
            logger.warning(f"ガベージコレクションエラー: {e}")

    def _log_cleanup_summary(self, summary: Dict[str, Any]) -> None:
        """クリーンアップサマリーログ出力"""
        if summary["errors"]:
            logger.warning(
                f"クリーンアップ完了（エラー{len(summary['errors'])}件あり）: "
                f"エンジン={summary['analysis_engines']}, "
                f"ML={summary['ml_engine']}, "
                f"バッチ={summary['batch_fetcher']}, "
                f"並列={summary['parallel_manager']}, "
                f"モニター={summary['performance_monitor']}, "
                f"フェッチャー={summary['stock_fetcher']}"
            )
            for error in summary["errors"][:5]:  # 最初の5件のエラーのみ表示
                logger.warning(f"  - {error}")
                
            if len(summary["errors"]) > 5:
                logger.warning(f"  - 他 {len(summary['errors']) - 5} 件のエラー")
        else:
            logger.info(
                f"Next-Gen AI Orchestrator クリーンアップ完了: "
                f"エンジン={summary['analysis_engines']}, "
                f"ML={summary['ml_engine']}, "
                f"バッチ={summary['batch_fetcher']}, "
                f"並列={summary['parallel_manager']}, "
                f"モニター={summary['performance_monitor']}, "
                f"フェッチャー={summary['stock_fetcher']}"
            )

    def get_resource_usage(self) -> Dict[str, Any]:
        """現在のリソース使用状況取得"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            return {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0,
                "threads_count": process.num_threads(),
                "gc_counts": gc.get_count(),
                "gc_stats": {
                    "collected": gc.get_stats() if hasattr(gc, 'get_stats') else [],
                    "garbage_count": len(gc.garbage),
                },
                "active_components": {
                    "analysis_engines": len(self.core.analysis_engines) if hasattr(self.core, 'analysis_engines') else 0,
                    "ml_engine": bool(self.core.ml_engine) if hasattr(self.core, 'ml_engine') else False,
                    "batch_fetcher": bool(self.core.batch_fetcher) if hasattr(self.core, 'batch_fetcher') else False,
                    "parallel_manager": bool(self.core.parallel_manager) if hasattr(self.core, 'parallel_manager') else False,
                },
            }
            
        except ImportError:
            # psutilが利用できない場合の基本情報
            return {
                "memory_usage_mb": "N/A (psutil not available)",
                "cpu_percent": "N/A",
                "gc_counts": gc.get_count(),
                "gc_garbage_count": len(gc.garbage),
                "active_components": {
                    "analysis_engines": len(self.core.analysis_engines) if hasattr(self.core, 'analysis_engines') else 0,
                    "ml_engine": bool(self.core.ml_engine) if hasattr(self.core, 'ml_engine') else False,
                    "batch_fetcher": bool(self.core.batch_fetcher) if hasattr(self.core, 'batch_fetcher') else False,
                    "parallel_manager": bool(self.core.parallel_manager) if hasattr(self.core, 'parallel_manager') else False,
                },
            }
        except Exception as e:
            return {"error": str(e)}

    def emergency_cleanup(self) -> Dict[str, Any]:
        """緊急時クリーンアップ"""
        logger.warning("緊急時クリーンアップ実行")
        
        emergency_summary = {
            "forced_cleanup": True,
            "components_terminated": 0,
            "errors": []
        }

        try:
            # 全コンポーネントを強制的にNoneに設定
            components = [
                'analysis_engines', 'ml_engine', 'batch_fetcher', 
                'parallel_manager', 'performance_monitor', 'stock_fetcher',
                'execution_history', 'performance_metrics'
            ]

            for component in components:
                if hasattr(self.core, component):
                    try:
                        setattr(self.core, component, None)
                        emergency_summary["components_terminated"] += 1
                    except Exception as e:
                        emergency_summary["errors"].append(f"{component}: {e}")

            # 強制ガベージコレクション
            for _ in range(5):
                gc.collect()

            logger.warning(f"緊急時クリーンアップ完了: {emergency_summary}")
            
        except Exception as e:
            emergency_summary["errors"].append(f"緊急時クリーンアップエラー: {e}")
            logger.error(f"緊急時クリーンアップ失敗: {e}")

        return emergency_summary