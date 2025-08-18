#!/usr/bin/env python3
"""
非同期対応アプリケーション拡張
Issue #918 項目8対応: 並行性・非同期処理の改善

メインアプリケーションの非同期機能拡張
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from .dependency_injection import get_container
from .async_services import (
    IAsyncExecutorService, IProgressMonitorService, ISchedulerService,
    AsyncTask, TaskPriority, WorkerType, run_parallel_analysis
)
from .application import StockAnalysisApplication
from ..utils.logging_config import get_context_logger


class AsyncStockAnalysisApplication(StockAnalysisApplication):
    """非同期対応株価分析アプリケーション"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 非同期サービス取得
        container = get_container()
        self.async_executor = container.resolve(IAsyncExecutorService)
        self.progress_monitor = container.resolve(IProgressMonitorService)
        self.scheduler = container.resolve(ISchedulerService)
        
        self.logger.info("AsyncStockAnalysisApplication initialized with async services")

    async def run_async_multi_analysis(self, symbols: List[str], max_concurrent: int = 10) -> Dict[str, Any]:
        """非同期マルチ銘柄分析実行"""
        self.logger.info(f"Starting async multi-analysis for {len(symbols)} symbols (max concurrent: {max_concurrent})")
        
        start_time = time.time()
        
        # 進捗監視開始
        task_id = f"async_multi_analysis_{int(time.time())}"
        self.progress_monitor.start_progress(task_id, len(symbols))
        
        # 並列分析実行
        results = await run_parallel_analysis(
            symbols, 
            self._analyze_symbol_async, 
            max_concurrent=max_concurrent
        )
        
        execution_time = time.time() - start_time
        
        # 結果統計
        successful_analyses = sum(1 for result in results.values() if not isinstance(result, Exception))
        failed_analyses = len(results) - successful_analyses
        
        summary = {
            'total_symbols': len(symbols),
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'execution_time': execution_time,
            'average_time_per_symbol': execution_time / len(symbols) if symbols else 0,
            'results': results,
            'progress_id': task_id
        }
        
        self.logger.info(
            f"Async multi-analysis completed: {successful_analyses}/{len(symbols)} successful, "
            f"execution time: {execution_time:.2f}s"
        )
        
        return summary

    async def _analyze_symbol_async(self, symbol: str) -> Dict[str, Any]:
        """非同期銘柄分析（最適化版）"""
        try:
            # データプロバイダーサービスを使用（非同期対応）
            loop = asyncio.get_event_loop()
            
            # データ取得を非同期で実行
            stock_data = await loop.run_in_executor(
                None, 
                self.data_provider_service.get_stock_data, 
                symbol, 
                "3mo"
            )
            
            if stock_data is None or stock_data.empty:
                return {
                    'symbol': symbol,
                    'status': 'no_data',
                    'recommendation': 'SKIP',
                    'confidence': 0.0,
                    'reason': 'データ取得失敗'
                }
            
            # テクニカル分析を非同期で実行
            analysis_result = await loop.run_in_executor(
                None,
                self._compute_technical_analysis,
                symbol,
                stock_data
            )
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Async analysis failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'error',
                'recommendation': 'HOLD',
                'confidence': 0.4,
                'error': str(e)
            }

    def _compute_technical_analysis(self, symbol: str, stock_data) -> Dict[str, Any]:
        """テクニカル分析計算（CPU集約処理）"""
        import pandas as pd
        import numpy as np
        
        # 最新価格
        current_price = stock_data['Close'].iloc[-1]
        
        # RSI計算
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # MACD計算  
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            return macd_line, macd_signal
        
        # 指標計算
        rsi = calculate_rsi(stock_data['Close'])
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        macd_line, macd_signal = calculate_macd(stock_data['Close'])
        current_macd = macd_line.iloc[-1] - macd_signal.iloc[-1] if not macd_line.empty else 0
        
        # 移動平均
        sma_config = self.analysis_config['technical_indicators'].get('sma', {'short_period': 20})
        sma_period = sma_config.get('short_period', 20)
        sma_data = stock_data['Close'].rolling(window=sma_period).mean()
        current_sma = sma_data.iloc[-1] if not sma_data.empty else current_price
        
        # 判定ロジック
        confidence = 0.5
        trend_score = 0.0
        
        rsi_config = self.analysis_config['technical_indicators'].get('rsi', {
            'oversold_threshold': 30, 'overbought_threshold': 70
        })
        
        # RSI判定
        if current_rsi < rsi_config.get('oversold_threshold', 30):
            trend_score += 0.4
            confidence += 0.2
        elif current_rsi > rsi_config.get('overbought_threshold', 70):
            trend_score -= 0.4
            confidence += 0.2
        
        # MACD判定
        if current_macd > 0:
            trend_score += 0.3
            confidence += 0.15
        else:
            trend_score -= 0.3
            confidence += 0.15
        
        # 移動平均判定
        if current_price > current_sma:
            trend_score += 0.2
            confidence += 0.1
        else:
            trend_score -= 0.2
            confidence += 0.1
        
        # 最終判定
        if confidence > 0.7 and trend_score > 0.4:
            recommendation = 'BUY'
        elif confidence > 0.6 and trend_score < -0.4:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        confidence = min(confidence, 0.95)
        reason = f'RSI:{current_rsi:.1f}, MACD:{current_macd:.3f}, SMA比:{(current_price/current_sma-1)*100:.1f}%'
        
        return {
            'symbol': symbol,
            'status': 'completed',
            'recommendation': recommendation,
            'confidence': confidence,
            'trend_score': trend_score,
            'reason': reason,
            'current_price': current_price,
            'current_rsi': current_rsi,
            'current_macd': current_macd,
            'sma_20': current_sma
        }

    async def run_batch_analysis_with_progress(self, symbols: List[str], batch_size: int = 20) -> Dict[str, Any]:
        """バッチ分析（進捗表示付き）"""
        self.logger.info(f"Starting batch analysis for {len(symbols)} symbols (batch size: {batch_size})")
        
        task_id = f"batch_analysis_{int(time.time())}"
        self.progress_monitor.start_progress(task_id, len(symbols))
        
        all_results = {}
        start_time = time.time()
        
        # バッチごとに処理
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            # 現在のバッチを並列実行
            batch_results = await run_parallel_analysis(
                batch_symbols,
                self._analyze_symbol_async,
                max_concurrent=min(batch_size, 10)
            )
            
            all_results.update(batch_results)
            
            # 進捗更新
            completed_count = min(i + batch_size, len(symbols))
            self.progress_monitor.update_progress(task_id, completed_count)
            
            # 進捗表示
            progress_info = self.progress_monitor.get_progress(task_id)
            print(f"Progress: {progress_info['percentage']:.1f}% ({completed_count}/{len(symbols)})")
            
            # バッチ間の短い休憩（リソース保護）
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.1)
        
        execution_time = time.time() - start_time
        
        return {
            'total_symbols': len(symbols),
            'batch_size': batch_size,
            'execution_time': execution_time,
            'results': all_results,
            'progress_id': task_id
        }

    def schedule_periodic_analysis(self, symbols: List[str], interval_minutes: int = 60) -> str:
        """定期分析スケジュール"""
        from datetime import timedelta
        
        def periodic_analysis():
            # 非同期分析を同期実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.run_async_multi_analysis(symbols))
                self.logger.info(f"Periodic analysis completed: {result['successful_analyses']} successful")
            finally:
                loop.close()
        
        schedule_id = self.scheduler.schedule_periodic(
            periodic_analysis,
            timedelta(minutes=interval_minutes)
        )
        
        self.logger.info(f"Scheduled periodic analysis: {schedule_id} (every {interval_minutes} minutes)")
        return schedule_id

    async def get_worker_performance(self) -> Dict[str, Any]:
        """ワーカーパフォーマンス取得"""
        worker_stats = self.async_executor.get_worker_stats()
        
        performance_summary = {
            'total_workers': len(worker_stats),
            'workers': {},
            'system_summary': {
                'total_tasks_completed': 0,
                'total_tasks_failed': 0,
                'average_execution_time': 0.0,
                'overall_success_rate': 0.0
            }
        }
        
        total_completed = 0
        total_failed = 0
        total_execution_times = []
        
        for worker_id, stats in worker_stats.items():
            performance_summary['workers'][worker_id] = {
                'worker_type': stats.worker_type.value,
                'total_tasks': stats.total_tasks,
                'completed_tasks': stats.completed_tasks,
                'failed_tasks': stats.failed_tasks,
                'success_rate': (stats.completed_tasks / stats.total_tasks * 100) if stats.total_tasks > 0 else 0,
                'avg_execution_time': stats.avg_execution_time,
                'current_load': stats.current_load,
                'cpu_usage': stats.cpu_usage,
                'memory_usage': stats.memory_usage
            }
            
            total_completed += stats.completed_tasks
            total_failed += stats.failed_tasks
            if stats.avg_execution_time > 0:
                total_execution_times.append(stats.avg_execution_time)
        
        # システムサマリー計算
        system_summary = performance_summary['system_summary']
        system_summary['total_tasks_completed'] = total_completed
        system_summary['total_tasks_failed'] = total_failed
        
        if total_execution_times:
            system_summary['average_execution_time'] = sum(total_execution_times) / len(total_execution_times)
        
        total_tasks = total_completed + total_failed
        if total_tasks > 0:
            system_summary['overall_success_rate'] = (total_completed / total_tasks) * 100
        
        return performance_summary

    def print_performance_summary(self, performance: Dict[str, Any]):
        """パフォーマンスサマリー表示"""
        print("\n" + "="*60)
        print("ASYNC PERFORMANCE SUMMARY")
        print("="*60)
        
        summary = performance['system_summary']
        print(f"Total Tasks Completed: {summary['total_tasks_completed']}")
        print(f"Total Tasks Failed: {summary['total_tasks_failed']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Average Execution Time: {summary['average_execution_time']:.3f}s")
        
        print(f"\nWorker Details:")
        for worker_id, stats in performance['workers'].items():
            print(f"  {worker_id}:")
            print(f"    Type: {stats['worker_type']}")
            print(f"    Success Rate: {stats['success_rate']:.1f}%")
            print(f"    Avg Time: {stats['avg_execution_time']:.3f}s")
            print(f"    CPU Usage: {stats['cpu_usage']:.1f}%")
            print(f"    Memory Usage: {stats['memory_usage']:.1f}%")
        
        print("="*60)


# ファクトリー関数
def create_async_stock_analysis_application(debug: bool = False, use_cache: bool = True) -> AsyncStockAnalysisApplication:
    """非同期株価分析アプリケーションインスタンスを作成"""
    return AsyncStockAnalysisApplication(debug=debug, use_cache=use_cache)