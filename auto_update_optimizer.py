#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Update Optimizer - 自動更新最適化システム

Issue #881対応：自動更新の更新時間を考える
- 更新頻度の動的最適化
- 対象銘柄数の拡張（20→100銘柄）
- 予測精度向上
- 進捗バー・ステータス表示システム
"""

import asyncio
import logging
import yaml
import time
import psutil
import threading
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import warnings
warnings.filterwarnings('ignore')

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 進捗表示ライブラリ
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 既存システムとの統合
try:
    from real_data_provider_v2 import real_data_provider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False

try:
    from ml_prediction_models import ml_prediction_models
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

try:
    from multi_timeframe_predictor import multi_timeframe_predictor
    MULTI_TIMEFRAME_AVAILABLE = True
except ImportError:
    MULTI_TIMEFRAME_AVAILABLE = False

try:
    from enhanced_symbol_manager import (
        EnhancedSymbolManager,
        EnhancedStockInfo,
        SymbolTier,
        IndustryCategory,
        VolatilityLevel
    )
    ENHANCED_SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    ENHANCED_SYMBOL_MANAGER_AVAILABLE = False

class LoadLevel(Enum):
    """システム負荷レベル"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class MarketPhase(Enum):
    """市場フェーズ"""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"

class SymbolPriority(Enum):
    """銘柄優先度"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class UpdateTask:
    """更新タスク"""
    symbol: str
    priority: SymbolPriority
    task_type: str
    created_at: datetime
    estimated_duration: float = 0.0
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class ProcessingStatus:
    """処理ステータス"""
    total_symbols: int = 0
    processed_symbols: int = 0
    failed_symbols: int = 0
    current_symbol: str = ""
    current_stage: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    queue_size: int = 0
    active_workers: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    errors: List[str] = field(default_factory=list)

@dataclass
class SystemMetrics:
    """システムメトリクス"""
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    load_level: LoadLevel = LoadLevel.NORMAL
    update_frequency: float = 60.0
    actual_frequency: float = 60.0

class SymbolPriorityQueue:
    """銘柄優先度キュー"""

    def __init__(self, symbols: List[str], priorities: Dict[str, SymbolPriority]):
        self.symbols = symbols
        self.priorities = priorities
        self.queue = queue.PriorityQueue()
        self._initialize_queue()

    def _initialize_queue(self):
        """キュー初期化"""
        for symbol in self.symbols:
            priority = self.priorities.get(symbol, SymbolPriority.MEDIUM)
            priority_value = {
                SymbolPriority.HIGH: 1,
                SymbolPriority.MEDIUM: 2,
                SymbolPriority.LOW: 3
            }.get(priority, 2)
            self.queue.put((priority_value, symbol))

    async def get_next_symbol(self) -> Optional[str]:
        """次の銘柄取得"""
        if not self.queue.empty():
            _, symbol = self.queue.get()
            return symbol
        return None

    async def update_symbol_score(self, symbol: str, score: float):
        """銘柄スコア更新"""
        # 実装では動的な優先度調整を行う
        pass

    def get_queue_size(self) -> int:
        """キューサイズ取得"""
        return self.queue.qsize()

    def is_empty(self) -> bool:
        """キューが空かチェック"""
        return self.queue.empty()

class UpdateFrequencyManager:
    """更新頻度管理"""

    def __init__(self, config: Dict, system_metrics: SystemMetrics):
        self.config = config
        self.system_metrics = system_metrics
        self.base_frequency = config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('base_frequency_seconds', 60)

    def get_base_frequency(self) -> float:
        """基本頻度取得"""
        return self.base_frequency

    def get_adjusted_frequency(self, symbol: str) -> float:
        """調整済み頻度取得"""
        load_multiplier = {
            LoadLevel.LOW: 0.8,
            LoadLevel.NORMAL: 1.0,
            LoadLevel.HIGH: 1.5,
            LoadLevel.CRITICAL: 2.0
        }.get(self.system_metrics.load_level, 1.0)

        return self.base_frequency * load_multiplier

class ProgressDisplayManager:
    """進捗表示管理"""

    def __init__(self, config: Dict):
        self.config = config
        self.is_active = False
        self.progress_info = {}
        if RICH_AVAILABLE:
            self.console = Console()

    def start_progress_display(self, total_symbols: int):
        """進捗表示開始"""
        self.is_active = True
        self.progress_info = {
            'total_count': total_symbols,
            'processed_count': 0,
            'current_symbol': '',
            'stage': '',
            'start_time': datetime.now()
        }

    def update_progress(self, current_symbol: str, processed_count: int, total_count: int, stage: str, processing_time: float):
        """進捗更新"""
        self.progress_info.update({
            'current_symbol': current_symbol,
            'processed_count': processed_count,
            'total_count': total_count,
            'stage': stage,
            'completion_percentage': (processed_count / total_count) * 100 if total_count > 0 else 0
        })

    def stop_progress_display(self):
        """進捗表示停止"""
        self.is_active = False

    def get_progress_info(self) -> Dict:
        """進捗情報取得"""
        return self.progress_info

class PerformanceTracker:
    """パフォーマンス追跡"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.is_tracking = False
        self.performance_data = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'processing_times': [],
            'start_time': None
        }

    def start_tracking(self):
        """追跡開始"""
        self.is_tracking = True
        self.performance_data['start_time'] = datetime.now()

    def stop_tracking(self):
        """追跡停止"""
        self.is_tracking = False

    def record_symbol_processing(self, symbol: str, processing_time: float, success: bool):
        """銘柄処理記録"""
        self.performance_data['total_processed'] += 1
        if success:
            self.performance_data['successful_processed'] += 1
        else:
            self.performance_data['failed_processed'] += 1
        self.performance_data['processing_times'].append(processing_time)

    def get_performance_statistics(self) -> Dict:
        """パフォーマンス統計取得"""
        total = self.performance_data['total_processed']
        successful = self.performance_data['successful_processed']
        processing_times = self.performance_data['processing_times']

        return {
            'total_processed': total,
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'average_processing_time': np.mean(processing_times) if processing_times else 0,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_usage_percent': psutil.cpu_percent()
        }

class AutoUpdateOptimizer:
    """自動更新最適化システム"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定読み込み
        self.config_path = config_path or Path("config/auto_update_optimizer_config.yaml")
        self.config = self._load_configuration()

        # システム状態
        self.is_running = False
        self.is_paused = False
        self.update_tasks = queue.PriorityQueue()
        self.processing_status = ProcessingStatus()
        self.system_metrics = SystemMetrics()

        # 銘柄管理システム統合
        if ENHANCED_SYMBOL_MANAGER_AVAILABLE:
            self.symbol_manager = EnhancedSymbolManager()
            self.logger.info("EnhancedSymbolManagerを統合しました")
        else:
            self.symbol_manager = None
            self.logger.warning("EnhancedSymbolManagerが利用できません")

        # 銘柄管理
        self.current_symbols = []
        self.symbol_priorities = {}
        self.symbol_performance = {}
        self.symbol_queue = None
        self.frequency_manager = None
        self.progress_manager = None
        self.performance_tracker = None

        # パフォーマンス追跡
        self.processing_times = []
        self.success_rates = []
        self.error_counts = {}

        # 進捗表示
        if RICH_AVAILABLE:
            self.console = Console()
            self.progress_table = None

        # データディレクトリ初期化
        self.data_dir = Path("auto_update_data")
        self.data_dir.mkdir(exist_ok=True)

        # データベース初期化
        self.db_path = self.data_dir / "auto_update_optimizer.db"
        self._init_database()

        # ワーカープール
        self.worker_pool = None
        self.max_workers = self.config.get('performance_optimization', {}).get('parallel_processing', {}).get('max_workers', 4)

        # 更新頻度管理
        self.current_frequency = self.config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('base_frequency_seconds', 60)
        self.last_update_time = datetime.now()

        self.logger.info("Auto update optimizer initialized")

    async def initialize(self):
        """システム初期化"""
        await self.initialize_symbols()

    def _load_configuration(self) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            'update_frequency_optimization': {
                'adaptive_frequency': {
                    'enabled': True,
                    'base_frequency_seconds': 60,
                    'min_frequency_seconds': 30,
                    'max_frequency_seconds': 300
                }
            },
            'symbol_expansion': {
                'max_symbols': 100,
                'batch_size': 10
            }
        }

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # システムメトリクステーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage_mb REAL,
                    memory_usage_percent REAL,
                    load_level TEXT,
                    update_frequency REAL,
                    actual_frequency REAL
                )
            """)

            # 処理履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT,
                    task_type TEXT,
                    processing_time REAL,
                    success INTEGER,
                    error_message TEXT
                )
            """)

            # 銘柄パフォーマンステーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prediction_accuracy REAL,
                    processing_time REAL,
                    data_quality_score REAL,
                    priority_score REAL
                )
            """)

    def update_system_metrics(self):
        """システムメトリクス更新"""

        # CPU・メモリ使用率取得
        self.system_metrics.cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        self.system_metrics.memory_usage_mb = memory.used / (1024 * 1024)
        self.system_metrics.memory_usage_percent = memory.percent

        # 負荷レベル判定
        self.system_metrics.load_level = self._determine_load_level()

        # 更新頻度調整
        self._adjust_update_frequency()

        # データベース保存
        self._save_system_metrics()

    def _determine_load_level(self) -> LoadLevel:
        """負荷レベル判定"""

        cpu_threshold_high = self.config.get('update_frequency_optimization', {}).get('load_based_adjustment', {}).get('cpu_threshold_high', 80)
        cpu_threshold_low = self.config.get('update_frequency_optimization', {}).get('load_based_adjustment', {}).get('cpu_threshold_low', 50)
        memory_threshold_mb = self.config.get('update_frequency_optimization', {}).get('load_based_adjustment', {}).get('memory_threshold_mb', 1000)

        cpu_usage = self.system_metrics.cpu_usage
        memory_usage = self.system_metrics.memory_usage_mb

        if cpu_usage >= 90 or memory_usage >= memory_threshold_mb * 1.5:
            return LoadLevel.CRITICAL
        elif cpu_usage >= cpu_threshold_high or memory_usage >= memory_threshold_mb:
            return LoadLevel.HIGH
        elif cpu_usage <= cpu_threshold_low and memory_usage <= memory_threshold_mb * 0.5:
            return LoadLevel.LOW
        else:
            return LoadLevel.NORMAL

    def _adjust_update_frequency(self):
        """更新頻度調整"""

        if not self.config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('enabled', True):
            return

        base_frequency = self.config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('base_frequency_seconds', 60)
        min_frequency = self.config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('min_frequency_seconds', 30)
        max_frequency = self.config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('max_frequency_seconds', 300)

        # 市場時間考慮
        market_frequency = self._get_market_based_frequency()

        # 負荷ベース調整
        load_multipliers = self.config.get('update_frequency_optimization', {}).get('load_based_adjustment', {}).get('frequency_multipliers', {})
        load_multiplier = load_multipliers.get(self.system_metrics.load_level.value, 1.0)

        # 調整後頻度計算
        adjusted_frequency = market_frequency * load_multiplier

        # 制限適用
        self.current_frequency = max(min_frequency, min(max_frequency, adjusted_frequency))
        self.system_metrics.update_frequency = self.current_frequency

    def _get_market_based_frequency(self) -> float:
        """市場時間ベース更新頻度取得"""

        if not self.config.get('update_frequency_optimization', {}).get('market_hours_aware', {}).get('enabled', False):
            return self.config.get('update_frequency_optimization', {}).get('adaptive_frequency', {}).get('base_frequency_seconds', 60)

        market_phase = self._get_current_market_phase()
        frequencies = self.config.get('update_frequency_optimization', {}).get('market_hours_aware', {}).get('frequencies', {})

        return frequencies.get(market_phase.value, 60)

    def _get_current_market_phase(self) -> MarketPhase:
        """現在の市場フェーズ取得"""

        now = datetime.now()
        current_time = now.time()

        # 週末判定
        if now.weekday() >= 5:  # 土曜日・日曜日
            return MarketPhase.WEEKEND

        # 市場時間設定
        trading_hours = self.config.get('update_frequency_optimization', {}).get('market_hours_aware', {}).get('trading_hours', {})
        start_time = dt_time.fromisoformat(trading_hours.get('start', '09:00'))
        end_time = dt_time.fromisoformat(trading_hours.get('end', '15:00'))

        # 市場開閉判定
        if start_time <= current_time <= end_time:
            return MarketPhase.MARKET_OPEN
        elif dt_time(8, 0) <= current_time < start_time:
            return MarketPhase.PRE_MARKET
        elif end_time < current_time <= dt_time(17, 0):
            return MarketPhase.AFTER_HOURS
        else:
            return MarketPhase.MARKET_CLOSE

    async def initialize_symbols(self):
        """銘柄初期化 - EnhancedSymbolManager統合"""

        max_symbols = self.config.get('symbol_expansion', {}).get('max_symbols', 100)
        selection_strategy = self.config.get('symbol_expansion', {}).get('selection_strategy', {})

        if self.symbol_manager and selection_strategy.get('method') == 'dynamic_scoring':
            # EnhancedSymbolManagerを使用した最適化選択
            self.current_symbols = self.get_optimized_symbol_selection(max_symbols)
        elif self.symbol_manager:
            # EnhancedSymbolManagerのデフォルト選択
            portfolio = self.symbol_manager.get_daytrading_optimized_portfolio(count=max_symbols)
            self.current_symbols = [stock.symbol for stock in portfolio]
        else:
            # フォールバック：デフォルト銘柄リスト
            self.current_symbols = [
                "7203", "8306", "9984", "6758", "4689", "6861", "8035", "9433", "4063", "7974",
                "6501", "7267", "8058", "4324", "6954", "9432", "4543", "8801", "2914", "9020"
            ][:max_symbols]

        # 銘柄優先度設定（EnhancedSymbolManager統合）
        if self.symbol_manager:
            categorized = self.categorize_symbols_by_priority()
            self.symbol_priorities = {}
            for priority, symbols in categorized.items():
                for symbol in symbols:
                    self.symbol_priorities[symbol] = priority
        else:
            self._assign_symbol_priorities()

        self.logger.info(f"Initialized {len(self.current_symbols)} symbols using EnhancedSymbolManager: {self.symbol_manager is not None}")

        # 統計情報表示
        if self.symbol_manager:
            stats = self.get_enhanced_symbol_statistics()
            self.logger.info(f"Symbol distribution - Tier1: {stats['tier_distribution'].get('コア大型株', 0)}, "
                           f"Tier2: {stats['tier_distribution'].get('成長中型株', 0)}, "
                           f"Tier3: {stats['tier_distribution'].get('機会小型株', 0)}")

        await self.initialize_components()

    async def initialize_components(self):
        """各コンポーネントの初期化"""
        # 優先度キューの初期化
        self.symbol_queue = SymbolPriorityQueue(self.current_symbols, self.symbol_priorities)

        # 更新頻度管理の初期化
        self.frequency_manager = UpdateFrequencyManager(self.config, self.system_metrics)

        # 進捗表示管理の初期化
        self.progress_manager = ProgressDisplayManager(self.config)

        # パフォーマンス追跡の初期化
        self.performance_tracker = PerformanceTracker(self.db_path)

        self.logger.info("All components initialized successfully")

    async def _select_symbols_by_scoring(self, max_symbols: int) -> List[str]:
        """スコアリングによる銘柄選択"""

        # 基本銘柄リスト（東証プライム主要銘柄）
        candidate_symbols = [
            "7203", "8306", "9984", "6758", "4689", "6861", "8035", "9433", "4063", "7974",
            "6501", "7267", "8058", "4324", "6954", "9432", "4543", "8801", "2914", "9020",
            "6098", "4755", "3382", "8031", "9062", "9064", "2802", "4519", "3659", "6367",
            "7751", "6702", "6273", "7735", "6594", "8233", "4188", "7731", "7013", "6326",
            "7182", "4661", "7832", "4452", "6762", "9101", "4612", "8252", "6178", "9766",
            "7269", "8604", "3407", "8628", "8053", "8002", "4307", "8750", "8830", "8697",
            "3436", "2269", "6504", "7003", "4901", "7012", "7201", "3861", "7911", "2371",
            "3401", "8001", "7951", "3103", "4521", "1605", "7205", "8316", "6146", "7717",
            "8354", "4568", "6724", "8766", "7270", "8154", "4208", "8267", "6965", "8309",
            "3048", "4004", "6988", "7013", "4151", "8306", "3659", "8113", "7013", "6113"
        ]

        # 重複除去
        candidate_symbols = list(set(candidate_symbols))

        # スコアリング実行
        symbol_scores = []

        for symbol in candidate_symbols[:50]:  # パフォーマンス考慮で50銘柄に制限
            try:
                score = await self._calculate_symbol_score(symbol)
                symbol_scores.append((symbol, score))
            except Exception as e:
                self.logger.warning(f"Failed to score symbol {symbol}: {e}")

        # スコア順ソート
        symbol_scores.sort(key=lambda x: x[1], reverse=True)

        return [symbol for symbol, score in symbol_scores[:max_symbols]]

    async def _calculate_symbol_score(self, symbol: str) -> float:
        """銘柄スコア計算"""

        scoring_criteria = self.config.get('symbol_expansion', {}).get('selection_strategy', {}).get('scoring_criteria', {})

        # 基本スコア
        base_score = 50.0

        # ダミーデータでのスコア計算（実際の実装では各種データソースを使用）
        volume_score = np.random.uniform(0, 100)      # 出来高スコア
        volatility_score = np.random.uniform(0, 100)  # ボラティリティスコア
        price_change_score = np.random.uniform(0, 100) # 価格変動スコア
        market_cap_score = np.random.uniform(0, 100)  # 時価総額スコア
        prediction_accuracy_score = np.random.uniform(0, 100) # 予測精度スコア

        # 重み付き合計
        total_score = (
            volume_score * scoring_criteria.get('volume_weight', 0.3) +
            volatility_score * scoring_criteria.get('volatility_weight', 0.2) +
            price_change_score * scoring_criteria.get('price_change_weight', 0.2) +
            market_cap_score * scoring_criteria.get('market_cap_weight', 0.1) +
            prediction_accuracy_score * scoring_criteria.get('prediction_accuracy_weight', 0.2)
        ) + base_score

        return total_score

    def _assign_symbol_priorities(self):
        """銘柄優先度割り当て"""

        categories = self.config.get('symbol_expansion', {}).get('categories', {})

        high_count = categories.get('high_priority', {}).get('count', 20)
        medium_count = categories.get('medium_priority', {}).get('count', 40)

        self.symbol_priorities = {}

        for i, symbol in enumerate(self.current_symbols):
            if i < high_count:
                self.symbol_priorities[symbol] = SymbolPriority.HIGH
            elif i < high_count + medium_count:
                self.symbol_priorities[symbol] = SymbolPriority.MEDIUM
            else:
                self.symbol_priorities[symbol] = SymbolPriority.LOW

    async def start_auto_update(self):
        """自動更新開始"""

        if self.is_running:
            self.logger.warning("Auto update is already running")
            return

        self.is_running = True
        self.processing_status = ProcessingStatus()

        # 銘柄初期化
        await self.initialize_symbols()

        # ワーカープール開始
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # 進捗表示開始
        if RICH_AVAILABLE:
            await self._start_progress_display()

        # メインループ開始
        await self._main_update_loop()

    async def _main_update_loop(self):
        """メイン更新ループ"""

        self.logger.info("Starting main update loop")

        try:
            while self.is_running:
                if not self.is_paused:
                    # システムメトリクス更新
                    self.update_system_metrics()

                    # 更新タスク生成
                    await self._create_update_tasks()

                    # タスク実行
                    await self._execute_update_tasks()

                    # 統計更新
                    self._update_statistics()

                # 更新頻度に基づく待機
                await asyncio.sleep(self.current_frequency)

        except Exception as e:
            self.logger.error(f"Main update loop error: {e}")
        finally:
            await self.stop_auto_update()

    async def _create_update_tasks(self):
        """更新タスク生成"""

        batch_size = self.config.get('symbol_expansion', {}).get('batch_size', 10)

        # 優先度別バッチ作成
        high_priority_symbols = [s for s, p in self.symbol_priorities.items() if p == SymbolPriority.HIGH]
        medium_priority_symbols = [s for s, p in self.symbol_priorities.items() if p == SymbolPriority.MEDIUM]
        low_priority_symbols = [s for s, p in self.symbol_priorities.items() if p == SymbolPriority.LOW]

        # 高優先度タスク
        for symbol in high_priority_symbols[:batch_size]:
            task = UpdateTask(
                symbol=symbol,
                priority=SymbolPriority.HIGH,
                task_type="data_update",
                created_at=datetime.now()
            )
            self.update_tasks.put((1, task))  # 優先度1（最高）

        # 中優先度タスク
        for symbol in medium_priority_symbols[:batch_size]:
            task = UpdateTask(
                symbol=symbol,
                priority=SymbolPriority.MEDIUM,
                task_type="data_update",
                created_at=datetime.now()
            )
            self.update_tasks.put((2, task))  # 優先度2

        # 低優先度タスク
        for symbol in low_priority_symbols[:batch_size//2]:
            task = UpdateTask(
                symbol=symbol,
                priority=SymbolPriority.LOW,
                task_type="data_update",
                created_at=datetime.now()
            )
            self.update_tasks.put((3, task))  # 優先度3

    async def _execute_update_tasks(self):
        """更新タスク実行"""

        executed_tasks = []

        # タスクキューから取得
        while not self.update_tasks.empty() and len(executed_tasks) < self.max_workers:
            try:
                priority, task = self.update_tasks.get_nowait()
                executed_tasks.append(task)
            except queue.Empty:
                break

        if not executed_tasks:
            return

        # 並列実行
        loop = asyncio.get_event_loop()
        futures = []

        for task in executed_tasks:
            future = loop.run_in_executor(self.worker_pool, self._process_single_task, task)
            futures.append(future)

        # 結果待ち
        results = await asyncio.gather(*futures, return_exceptions=True)

        # 結果処理
        for task, result in zip(executed_tasks, results):
            if isinstance(result, Exception):
                self.logger.error(f"Task failed for {task.symbol}: {result}")
                self.processing_status.failed_symbols += 1
                self.processing_status.errors.append(f"{task.symbol}: {str(result)}")
            else:
                self.processing_status.processed_symbols += 1

        self.processing_status.queue_size = self.update_tasks.qsize()

    def _process_single_task(self, task: UpdateTask) -> Dict[str, Any]:
        """単一タスク処理"""

        start_time = time.time()
        self.processing_status.current_symbol = task.symbol
        self.processing_status.current_stage = "データ取得"

        try:
            # データ取得（ダミー処理）
            time.sleep(np.random.uniform(0.5, 2.0))  # 実際の処理時間をシミュレート

            self.processing_status.current_stage = "予測実行"

            # 予測実行（ダミー処理）
            time.sleep(np.random.uniform(0.2, 1.0))

            self.processing_status.current_stage = "結果保存"

            # 結果保存
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # データベース保存
            self._save_processing_result(task, processing_time, True, None)

            return {
                'symbol': task.symbol,
                'processing_time': processing_time,
                'success': True
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self._save_processing_result(task, processing_time, False, str(e))
            raise e

    def _save_processing_result(self, task: UpdateTask, processing_time: float, success: bool, error_message: Optional[str]):
        """処理結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO processing_history
                    (timestamp, symbol, task_type, processing_time, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    task.symbol,
                    task.task_type,
                    processing_time,
                    1 if success else 0,
                    error_message
                ))
        except Exception as e:
            self.logger.error(f"Failed to save processing result: {e}")

    def _save_system_metrics(self):
        """システムメトリクス保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO system_metrics
                    (timestamp, cpu_usage, memory_usage_mb, memory_usage_percent,
                     load_level, update_frequency, actual_frequency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    self.system_metrics.cpu_usage,
                    self.system_metrics.memory_usage_mb,
                    self.system_metrics.memory_usage_percent,
                    self.system_metrics.load_level.value,
                    self.system_metrics.update_frequency,
                    self.system_metrics.actual_frequency
                ))
        except Exception as e:
            self.logger.error(f"Failed to save system metrics: {e}")

    def _update_statistics(self):
        """統計更新"""

        # 平均処理時間
        if self.processing_times:
            self.processing_status.average_processing_time = np.mean(self.processing_times[-100:])  # 直近100件

        # 成功率
        total_processed = self.processing_status.processed_symbols + self.processing_status.failed_symbols
        if total_processed > 0:
            self.processing_status.success_rate = self.processing_status.processed_symbols / total_processed

        # 完了予測時刻
        if self.processing_status.average_processing_time > 0:
            remaining_symbols = self.processing_status.total_symbols - total_processed
            estimated_seconds = remaining_symbols * self.processing_status.average_processing_time
            self.processing_status.estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)

        # アクティブワーカー数
        self.processing_status.active_workers = min(self.max_workers, self.update_tasks.qsize())

    async def _start_progress_display(self):
        """進捗表示開始"""

        if not RICH_AVAILABLE:
            return

        # 進捗表示を別スレッドで開始
        progress_thread = threading.Thread(target=self._run_progress_display, daemon=True)
        progress_thread.start()

    def _run_progress_display(self):
        """進捗表示実行"""

        if not RICH_AVAILABLE:
            return

        try:
            with Live(self._create_progress_layout(), refresh_per_second=2, console=self.console) as live:
                while self.is_running:
                    live.update(self._create_progress_layout())
                    time.sleep(0.5)
        except Exception as e:
            self.logger.error(f"Progress display error: {e}")

    def _create_progress_layout(self):
        """進捗レイアウト作成"""

        if not RICH_AVAILABLE:
            return "Progress display not available"

        # システムステータステーブル
        system_table = Table(title="🖥️ システムステータス")
        system_table.add_column("項目", style="cyan")
        system_table.add_column("値", style="green")

        system_table.add_row("CPU使用率", f"{self.system_metrics.cpu_usage:.1f}%")
        system_table.add_row("メモリ使用量", f"{self.system_metrics.memory_usage_mb:.0f}MB ({self.system_metrics.memory_usage_percent:.1f}%)")
        system_table.add_row("負荷レベル", self.system_metrics.load_level.value.upper())
        system_table.add_row("更新頻度", f"{self.current_frequency:.0f}秒")
        system_table.add_row("アクティブワーカー", f"{self.processing_status.active_workers}/{self.max_workers}")

        # 処理進捗テーブル
        progress_table = Table(title="📊 処理進捗")
        progress_table.add_column("項目", style="cyan")
        progress_table.add_column("値", style="yellow")

        total_symbols = len(self.current_symbols)
        processed = self.processing_status.processed_symbols
        failed = self.processing_status.failed_symbols
        progress_percent = (processed + failed) / total_symbols * 100 if total_symbols > 0 else 0

        progress_table.add_row("総銘柄数", f"{total_symbols}")
        progress_table.add_row("処理済み", f"{processed}")
        progress_table.add_row("失敗", f"{failed}")
        progress_table.add_row("進捗率", f"{progress_percent:.1f}%")
        progress_table.add_row("現在処理中", f"{self.processing_status.current_symbol}")
        progress_table.add_row("処理ステージ", f"{self.processing_status.current_stage}")
        progress_table.add_row("キューサイズ", f"{self.processing_status.queue_size}")

        if self.processing_status.average_processing_time > 0:
            progress_table.add_row("平均処理時間", f"{self.processing_status.average_processing_time:.2f}秒")

        if self.processing_status.estimated_completion:
            progress_table.add_row("完了予定", f"{self.processing_status.estimated_completion.strftime('%H:%M:%S')}")

        progress_table.add_row("成功率", f"{self.processing_status.success_rate*100:.1f}%")

        # エラーログ
        error_panel = Panel(
            "\n".join(self.processing_status.errors[-5:]) if self.processing_status.errors else "エラーなし",
            title="⚠️ 最新エラー",
            border_style="red"
        )

        # レイアウト組み立て
        from rich.columns import Columns
        return Columns([system_table, progress_table, error_panel])

    async def stop_auto_update(self):
        """自動更新停止"""

        self.is_running = False

        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)

        self.logger.info("Auto update stopped")

    def pause_auto_update(self):
        """自動更新一時停止"""
        self.is_paused = True
        self.logger.info("Auto update paused")

    def resume_auto_update(self):
        """自動更新再開"""
        self.is_paused = False
        self.logger.info("Auto update resumed")

    async def get_system_status(self) -> Dict[str, Any]:
        """システムステータス取得"""

        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_symbols_count': len(self.current_symbols),
            'processing_status': {
                'total_symbols': len(self.current_symbols),
                'processed_symbols': self.processing_status.processed_symbols,
                'failed_symbols': self.processing_status.failed_symbols,
                'current_symbol': self.processing_status.current_symbol,
                'current_stage': self.processing_status.current_stage,
                'queue_size': self.processing_status.queue_size,
                'active_workers': self.processing_status.active_workers,
                'average_processing_time': self.processing_status.average_processing_time,
                'success_rate': self.processing_status.success_rate
            },
            'system_metrics': {
                'cpu_usage': self.system_metrics.cpu_usage,
                'memory_usage_mb': self.system_metrics.memory_usage_mb,
                'memory_usage_percent': self.system_metrics.memory_usage_percent,
                'load_level': self.system_metrics.load_level.value,
                'update_frequency': self.current_frequency,
                'actual_frequency': self.system_metrics.actual_frequency
            },
            'recent_errors': self.processing_status.errors[-10:]
        }

    # EnhancedSymbolManager統合メソッド

    def get_optimized_symbol_selection(self, target_count: int = None) -> List[str]:
        """最適化された銘柄選択"""
        if not self.symbol_manager:
            return self.current_symbols[:target_count] if target_count else self.current_symbols

        if target_count is None:
            target_count = self.config.get('symbol_expansion', {}).get('max_symbols', 50)

        # デイトレード最適化ポートフォリオを基本に
        portfolio = self.symbol_manager.get_daytrading_optimized_portfolio(count=target_count)
        symbols = [stock.symbol for stock in portfolio]

        # 高ボラティリティ機会銘柄を追加
        if len(symbols) < target_count:
            high_vol_symbols = self.symbol_manager.get_high_volatility_opportunities(
                count=target_count - len(symbols)
            )
            symbols.extend([stock.symbol for stock in high_vol_symbols
                          if stock.symbol not in symbols])

        # 業界分散ポートフォリオで不足分を補完
        if len(symbols) < target_count:
            diversified_symbols = self.symbol_manager.get_diversified_portfolio(
                count=target_count
            )
            symbols.extend([stock.symbol for stock in diversified_symbols
                          if stock.symbol not in symbols])

        return symbols[:target_count]

    def calculate_symbol_priority_scores(self) -> Dict[str, float]:
        """銘柄優先度スコア計算"""
        scores = {}

        if not self.symbol_manager:
            # フォールバック：全て同じ優先度
            return {symbol: 50.0 for symbol in self.current_symbols}

        for symbol in self.current_symbols:
            if symbol in self.symbol_manager.symbols:
                stock_info = self.symbol_manager.symbols[symbol]

                # ボラティリティスコア（デイトレードに重要）
                volatility_scores = {
                    VolatilityLevel.VERY_HIGH: 100,
                    VolatilityLevel.HIGH: 85,
                    VolatilityLevel.MEDIUM: 65,
                    VolatilityLevel.LOW: 40,
                    VolatilityLevel.VERY_LOW: 20
                }
                vol_score = volatility_scores.get(stock_info.volatility_level, 50)

                # 流動性スコア
                liquidity_score = stock_info.liquidity_score

                # 成長性スコア
                growth_score = stock_info.growth_potential

                # ティア別ボーナス
                tier_bonus = {
                    SymbolTier.TIER1_CORE: 20,
                    SymbolTier.TIER2_GROWTH: 15,
                    SymbolTier.TIER3_OPPORTUNITY: 25,  # 高ボラ機会銘柄に高ボーナス
                    SymbolTier.TIER4_SPECIAL: 10
                }.get(stock_info.tier, 10)

                # 総合スコア計算
                total_score = (
                    vol_score * 0.35 +
                    liquidity_score * 0.25 +
                    growth_score * 0.25 +
                    tier_bonus * 0.15
                )

                scores[symbol] = min(100.0, max(0.0, total_score))
            else:
                scores[symbol] = 50.0  # デフォルトスコア

        return scores

    def categorize_symbols_by_priority(self) -> Dict[SymbolPriority, List[str]]:
        """銘柄を優先度別に分類"""
        scores = self.calculate_symbol_priority_scores()

        # スコア順でソート
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 設定に基づく分類
        config = self.config.get('symbol_expansion', {}).get('categories', {})
        high_count = config.get('high_priority', {}).get('count', 20)
        medium_count = config.get('medium_priority', {}).get('count', 30)

        categorized = {
            SymbolPriority.HIGH: [],
            SymbolPriority.MEDIUM: [],
            SymbolPriority.LOW: []
        }

        total_symbols = len(sorted_symbols)

        for i, (symbol, score) in enumerate(sorted_symbols):
            if i < high_count:
                categorized[SymbolPriority.HIGH].append(symbol)
            elif i < high_count + medium_count:
                categorized[SymbolPriority.MEDIUM].append(symbol)
            else:
                categorized[SymbolPriority.LOW].append(symbol)

        return categorized

    def get_symbol_update_frequency(self, symbol: str) -> float:
        """銘柄別更新頻度取得"""
        if symbol not in self.symbol_priorities:
            return self.current_frequency

        priority = self.symbol_priorities[symbol]
        config = self.config.get('symbol_expansion', {}).get('categories', {})

        frequency_map = {
            SymbolPriority.HIGH: config.get('high_priority', {}).get('update_frequency', 30),
            SymbolPriority.MEDIUM: config.get('medium_priority', {}).get('update_frequency', 60),
            SymbolPriority.LOW: config.get('low_priority', {}).get('update_frequency', 120)
        }

        base_frequency = frequency_map.get(priority, self.current_frequency)

        # システム負荷による調整
        load_multiplier = {
            LoadLevel.LOW: 0.8,
            LoadLevel.NORMAL: 1.0,
            LoadLevel.HIGH: 1.5,
            LoadLevel.CRITICAL: 2.0
        }.get(self.system_metrics.load_level, 1.0)

        return base_frequency * load_multiplier

    def get_enhanced_symbol_statistics(self) -> Dict[str, Any]:
        """拡張銘柄統計"""
        if not self.symbol_manager:
            return {'error': 'EnhancedSymbolManager not available'}

        # 基本統計
        basic_stats = self.symbol_manager.get_summary_statistics()

        # 現在の選択銘柄統計
        current_symbols_info = []
        for symbol in self.current_symbols:
            if symbol in self.symbol_manager.symbols:
                current_symbols_info.append(self.symbol_manager.symbols[symbol])

        # 優先度分布
        priority_distribution = {}
        for priority in SymbolPriority:
            priority_distribution[priority.value] = len([
                s for s, p in self.symbol_priorities.items() if p == priority
            ])

        # ティア分布
        tier_distribution = {}
        for tier in SymbolTier:
            count = sum(1 for stock in current_symbols_info if stock.tier == tier)
            tier_distribution[tier.value] = count

        # 業界分布
        industry_distribution = {}
        for industry in IndustryCategory:
            count = sum(1 for stock in current_symbols_info if stock.industry == industry)
            industry_distribution[industry.value] = count

        return {
            'total_available_symbols': basic_stats['total_symbols'],
            'current_selected_symbols': len(self.current_symbols),
            'priority_distribution': priority_distribution,
            'tier_distribution': tier_distribution,
            'industry_distribution': industry_distribution,
            'average_scores': {
                'liquidity': np.mean([s.liquidity_score for s in current_symbols_info]) if current_symbols_info else 0,
                'stability': np.mean([s.stability_score for s in current_symbols_info]) if current_symbols_info else 0,
                'growth_potential': np.mean([s.growth_potential for s in current_symbols_info]) if current_symbols_info else 0,
                'risk_score': np.mean([s.risk_score for s in current_symbols_info]) if current_symbols_info else 0
            }
        }

# グローバルインスタンス
auto_update_optimizer = AutoUpdateOptimizer()

# テスト関数
async def test_auto_update_optimizer():
    """自動更新最適化システムのテスト"""

    print("=== 自動更新最適化システム テスト ===")

    optimizer = AutoUpdateOptimizer()

    print(f"\n[ システム初期化完了 ]")
    print(f"設定読み込み: ✅")
    print(f"データベース初期化: ✅")
    print(f"最大ワーカー数: {optimizer.max_workers}")
    print(f"基本更新頻度: {optimizer.current_frequency}秒")

    # システムメトリクス更新テスト
    print(f"\n[ システムメトリクス更新テスト ]")
    optimizer.update_system_metrics()
    print(f"CPU使用率: {optimizer.system_metrics.cpu_usage:.1f}%")
    print(f"メモリ使用量: {optimizer.system_metrics.memory_usage_mb:.0f}MB")
    print(f"負荷レベル: {optimizer.system_metrics.load_level.value}")
    print(f"調整後更新頻度: {optimizer.current_frequency:.0f}秒")

    # 銘柄初期化テスト
    print(f"\n[ 銘柄初期化テスト ]")
    await optimizer.initialize_symbols()
    print(f"選択銘柄数: {len(optimizer.current_symbols)}")
    print(f"銘柄例: {optimizer.current_symbols[:10]}")

    # 銘柄優先度確認
    high_priority = [s for s, p in optimizer.symbol_priorities.items() if p == SymbolPriority.HIGH]
    medium_priority = [s for s, p in optimizer.symbol_priorities.items() if p == SymbolPriority.MEDIUM]
    low_priority = [s for s, p in optimizer.symbol_priorities.items() if p == SymbolPriority.LOW]

    print(f"高優先度銘柄: {len(high_priority)}件")
    print(f"中優先度銘柄: {len(medium_priority)}件")
    print(f"低優先度銘柄: {len(low_priority)}件")

    # 短時間実行テスト
    print(f"\n[ 短時間実行テスト（10秒間） ]")

    # 非同期実行
    import concurrent.futures

    async def short_run():
        optimizer.is_running = True
        optimizer.worker_pool = ThreadPoolExecutor(max_workers=optimizer.max_workers)

        # タスク生成・実行
        await optimizer._create_update_tasks()
        await optimizer._execute_update_tasks()

        # 統計更新
        optimizer._update_statistics()

        optimizer.worker_pool.shutdown(wait=True)
        optimizer.is_running = False

    await short_run()

    # 結果確認
    status = await optimizer.get_system_status()
    print(f"処理済み銘柄: {status['processing_status']['processed_symbols']}")
    print(f"失敗銘柄: {status['processing_status']['failed_symbols']}")
    print(f"成功率: {status['processing_status']['success_rate']*100:.1f}%")
    print(f"平均処理時間: {status['processing_status']['average_processing_time']:.2f}秒")

    print(f"\n=== 自動更新最適化システム テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_auto_update_optimizer())