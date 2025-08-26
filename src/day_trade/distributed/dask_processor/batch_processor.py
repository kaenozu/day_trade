#!/usr/bin/env python3
"""
Dask Batch Processor
Issue #384: 並列処理のさらなる強化 - Batch Processing Module

Dask分散処理による市場データパイプライン処理とバッチ処理機能を提供する
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Dask関連インポート（オプショナル）
try:
    from dask import compute
    from dask.delayed import delayed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# プロジェクトモジュール
try:
    from ...analysis.technical_analysis import TechnicalAnalyzer
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class TechnicalAnalyzer:
        def analyze(self, data):
            return data


logger = get_context_logger(__name__)


class DaskBatchProcessor:
    """Dask特化バッチプロセッサー"""

    def __init__(self, dask_processor):
        """
        初期化

        Args:
            dask_processor: DaskDataProcessorインスタンス
        """
        self.dask_processor = dask_processor

    async def process_market_data_pipeline(
        self,
        symbols: List[str],
        pipeline_steps: List[str],
        start_date: datetime,
        end_date: datetime,
        output_format: str = "parquet",
    ) -> Dict[str, Any]:
        """
        市場データ処理パイプライン

        Args:
            symbols: 処理対象銘柄リスト
            pipeline_steps: 処理ステップリスト
            start_date: 開始日
            end_date: 終了日
            output_format: 出力形式（parquet, csv）

        Returns:
            パイプライン処理結果
        """
        logger.info(
            f"市場データパイプライン開始: {len(symbols)}銘柄, {len(pipeline_steps)}ステップ"
        )

        try:
            # ステップ1: 基本データ取得
            if "fetch_data" in pipeline_steps:
                raw_data = await self.dask_processor.process_multiple_symbols_parallel(
                    symbols, start_date, end_date, include_technical=False
                )
            else:
                raw_data = pd.DataFrame()

            # ステップ2: テクニカル分析
            if "technical_analysis" in pipeline_steps and not raw_data.empty:
                enhanced_data = await self._apply_technical_analysis_distributed(
                    raw_data
                )
            else:
                enhanced_data = raw_data

            # ステップ3: データクリーニング
            if "data_cleaning" in pipeline_steps and not enhanced_data.empty:
                cleaned_data = await self._apply_data_cleaning_distributed(
                    enhanced_data
                )
            else:
                cleaned_data = enhanced_data

            # ステップ4: 出力
            output_path = None
            if not cleaned_data.empty and output_format:
                output_path = await self._save_pipeline_output(
                    cleaned_data, output_format
                )

            results = {
                "pipeline_timestamp": datetime.now().isoformat(),
                "symbols_processed": len(symbols),
                "steps_executed": pipeline_steps,
                "records_processed": len(cleaned_data) if not cleaned_data.empty else 0,
                "output_path": output_path,
                "processing_successful": not cleaned_data.empty,
            }

            logger.info(f"パイプライン完了: {results['records_processed']}レコード処理")
            return results

        except Exception as e:
            logger.error(f"市場データパイプラインエラー: {e}")
            return {"error": str(e), "processing_successful": False}

    async def _apply_technical_analysis_distributed(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        分散テクニカル分析適用

        Args:
            data: 元データ

        Returns:
            テクニカル分析適用後のデータ
        """
        if not DASK_AVAILABLE:
            return self._apply_technical_analysis_sequential(data)

        try:

            @delayed
            def apply_technical_to_symbol(symbol_data):
                try:
                    analyzer = TechnicalAnalyzer()
                    return analyzer.analyze(symbol_data)
                except Exception as e:
                    logger.debug(f"テクニカル分析エラー: {e}")
                    return symbol_data

            # シンボル別に分散処理
            symbol_groups = data.groupby("symbol")
            analysis_tasks = [
                apply_technical_to_symbol(group.reset_index(drop=True))
                for symbol, group in symbol_groups
            ]

            analyzed_results = compute(*analysis_tasks)

            # 結果統合
            if analyzed_results:
                return pd.concat(
                    [r for r in analyzed_results if not r.empty], ignore_index=True
                )
            else:
                return data

        except Exception as e:
            logger.error(f"分散テクニカル分析エラー: {e}")
            return self._apply_technical_analysis_sequential(data)

    def _apply_technical_analysis_sequential(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        シーケンシャルテクニカル分析（フォールバック）

        Args:
            data: 元データ

        Returns:
            テクニカル分析適用後のデータ
        """
        try:
            analyzer = TechnicalAnalyzer()
            return analyzer.analyze(data)
        except Exception as e:
            logger.error(f"シーケンシャルテクニカル分析エラー: {e}")
            return data

    async def _apply_data_cleaning_distributed(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        分散データクリーニング

        Args:
            data: 元データ

        Returns:
            クリーニング後のデータ
        """
        if not DASK_AVAILABLE or data.empty:
            return self._apply_data_cleaning_sequential(data)

        try:

            @delayed
            def clean_data_chunk(chunk):
                try:
                    # 基本クリーニング
                    chunk = chunk.dropna(subset=["close", "volume"])
                    chunk = chunk[chunk["volume"] > 0]
                    chunk = chunk[chunk["close"] > 0]

                    # 異常値除去（簡易）
                    for col in ["close", "high", "low", "volume"]:
                        if col in chunk.columns:
                            q1 = chunk[col].quantile(0.01)
                            q99 = chunk[col].quantile(0.99)
                            chunk = chunk[(chunk[col] >= q1) & (chunk[col] <= q99)]

                    return chunk

                except Exception as e:
                    logger.debug(f"データクリーニングエラー: {e}")
                    return chunk

            # データを適当なチャンクに分割
            chunk_size = max(1000, len(data) // (self.dask_processor.n_workers * 4))
            data_chunks = [
                data.iloc[i : i + chunk_size] for i in range(0, len(data), chunk_size)
            ]

            # 分散クリーニング実行
            cleaning_tasks = [clean_data_chunk(chunk) for chunk in data_chunks]
            cleaned_chunks = compute(*cleaning_tasks)

            # 結果統合
            if cleaned_chunks:
                cleaned_data = pd.concat(
                    [c for c in cleaned_chunks if not c.empty], ignore_index=True
                )
                return cleaned_data
            else:
                return data

        except Exception as e:
            logger.error(f"分散データクリーニングエラー: {e}")
            return self._apply_data_cleaning_sequential(data)

    def _apply_data_cleaning_sequential(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        シーケンシャルデータクリーニング

        Args:
            data: 元データ

        Returns:
            クリーニング後のデータ
        """
        try:
            if data.empty:
                return data

            # 基本クリーニング
            cleaned = data.dropna(subset=["close"])
            cleaned = cleaned[cleaned["close"] > 0]

            if "volume" in cleaned.columns:
                cleaned = cleaned[cleaned["volume"] >= 0]

            return cleaned

        except Exception as e:
            logger.error(f"シーケンシャルデータクリーニングエラー: {e}")
            return data

    async def _save_pipeline_output(
        self, data: pd.DataFrame, output_format: str
    ) -> Optional[str]:
        """
        パイプライン出力保存

        Args:
            data: 保存対象データ
            output_format: 出力形式

        Returns:
            保存ファイルパス
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_format.lower() == "parquet":
                output_path = (
                    self.dask_processor.temp_dir
                    / f"pipeline_output_{timestamp}.parquet"
                )
                data.to_parquet(output_path, compression="snappy", index=False)
            elif output_format.lower() == "csv":
                output_path = (
                    self.dask_processor.temp_dir / f"pipeline_output_{timestamp}.csv"
                )
                data.to_csv(output_path, index=False)
            else:
                logger.warning(f"サポートされていない出力形式: {output_format}")
                return None

            logger.info(f"パイプライン出力保存: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"パイプライン出力保存エラー: {e}")
            return None

    async def process_custom_analysis_pipeline(
        self,
        symbols: List[str],
        analysis_functions: List[callable],
        start_date: datetime,
        end_date: datetime,
        parallel_execution: bool = True,
    ) -> Dict[str, Any]:
        """
        カスタム分析パイプライン処理

        Args:
            symbols: 処理対象銘柄リスト
            analysis_functions: 分析関数リスト
            start_date: 開始日
            end_date: 終了日
            parallel_execution: 並列実行フラグ

        Returns:
            カスタム分析結果
        """
        logger.info(
            f"カスタム分析パイプライン開始: {len(symbols)}銘柄, "
            f"{len(analysis_functions)}分析関数"
        )

        try:
            # 基本データ取得
            raw_data = await self.dask_processor.process_multiple_symbols_parallel(
                symbols, start_date, end_date, include_technical=False
            )

            if raw_data.empty:
                logger.warning("カスタム分析用データが不足しています")
                return {"processing_successful": False, "error": "No data available"}

            # カスタム分析実行
            if parallel_execution and DASK_AVAILABLE:
                analysis_results = await self._execute_custom_analysis_distributed(
                    raw_data, analysis_functions
                )
            else:
                analysis_results = await self._execute_custom_analysis_sequential(
                    raw_data, analysis_functions
                )

            results = {
                "pipeline_timestamp": datetime.now().isoformat(),
                "symbols_processed": len(symbols),
                "analysis_functions_executed": len(analysis_functions),
                "records_analyzed": len(raw_data),
                "analysis_results": analysis_results,
                "processing_successful": True,
            }

            logger.info(
                f"カスタム分析パイプライン完了: "
                f"{len(analysis_functions)}関数, {len(raw_data)}レコード"
            )
            return results

        except Exception as e:
            logger.error(f"カスタム分析パイプラインエラー: {e}")
            return {"error": str(e), "processing_successful": False}

    async def _execute_custom_analysis_distributed(
        self, data: pd.DataFrame, analysis_functions: List[callable]
    ) -> Dict[str, Any]:
        """
        分散カスタム分析実行

        Args:
            data: 分析対象データ
            analysis_functions: 分析関数リスト

        Returns:
            分析結果
        """
        try:

            @delayed
            def apply_analysis_function(func, data_subset):
                try:
                    return func.__name__, func(data_subset)
                except Exception as e:
                    logger.debug(f"カスタム分析関数エラー {func.__name__}: {e}")
                    return func.__name__, {"error": str(e)}

            # 各分析関数を並列実行
            analysis_tasks = [
                apply_analysis_function(func, data) for func in analysis_functions
            ]

            analysis_results = compute(*analysis_tasks)

            # 結果整理
            results = {}
            for func_name, result in analysis_results:
                results[func_name] = result

            return results

        except Exception as e:
            logger.error(f"分散カスタム分析実行エラー: {e}")
            return await self._execute_custom_analysis_sequential(
                data, analysis_functions
            )

    async def _execute_custom_analysis_sequential(
        self, data: pd.DataFrame, analysis_functions: List[callable]
    ) -> Dict[str, Any]:
        """
        シーケンシャルカスタム分析実行（フォールバック）

        Args:
            data: 分析対象データ
            analysis_functions: 分析関数リスト

        Returns:
            分析結果
        """
        results = {}

        for func in analysis_functions:
            try:
                results[func.__name__] = func(data)
            except Exception as e:
                logger.error(f"カスタム分析関数エラー {func.__name__}: {e}")
                results[func.__name__] = {"error": str(e)}

        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        パイプライン統計情報取得

        Returns:
            統計情報
        """
        try:
            base_stats = self.dask_processor.get_stats()
            
            pipeline_stats = {
                **base_stats,
                "batch_processor_status": "active",
                "supported_output_formats": ["parquet", "csv"],
                "available_pipeline_steps": [
                    "fetch_data",
                    "technical_analysis", 
                    "data_cleaning"
                ],
            }

            return pipeline_stats

        except Exception as e:
            logger.error(f"パイプライン統計取得エラー: {e}")
            return {"error": str(e)}

    def validate_pipeline_configuration(
        self, pipeline_steps: List[str]
    ) -> Dict[str, Any]:
        """
        パイプライン設定検証

        Args:
            pipeline_steps: 検証対象パイプラインステップ

        Returns:
            検証結果
        """
        valid_steps = ["fetch_data", "technical_analysis", "data_cleaning"]
        
        validation_results = {
            "is_valid": True,
            "invalid_steps": [],
            "missing_required_steps": [],
            "recommendations": [],
        }

        # 無効なステップチェック
        invalid_steps = [step for step in pipeline_steps if step not in valid_steps]
        if invalid_steps:
            validation_results["is_valid"] = False
            validation_results["invalid_steps"] = invalid_steps

        # 必須ステップチェック
        if "fetch_data" not in pipeline_steps:
            validation_results["missing_required_steps"].append("fetch_data")
            validation_results["recommendations"].append(
                "データ取得ステップ（fetch_data）を含めることを推奨します"
            )

        # 順序チェック
        if "technical_analysis" in pipeline_steps and "fetch_data" not in pipeline_steps:
            validation_results["recommendations"].append(
                "テクニカル分析にはデータ取得ステップが必要です"
            )

        if "data_cleaning" in pipeline_steps and len(pipeline_steps) == 1:
            validation_results["recommendations"].append(
                "データクリーニングのみの実行は推奨されません"
            )

        return validation_results