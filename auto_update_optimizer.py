#!/usr/bin/env python3
"""
自動更新時間最適化システム
Issue #881: 自動更新の更新時間を考える

市場状況と銘柄特性に基づいて動的に更新頻度を最適化
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


class MarketCondition(Enum):
    """市場状況"""
    OPENING = "opening"      # 開場直後
    NORMAL = "normal"        # 通常取引
    LUNCH = "lunch"          # 昼休み
    CLOSING = "closing"      # 引け前
    HIGH_VOLATILITY = "high_vol"  # 高ボラティリティ
    LOW_VOLATILITY = "low_vol"    # 低ボラティリティ


class SymbolPriority(Enum):
    """銘柄優先度"""
    CRITICAL = "critical"    # 最重要銘柄
    HIGH = "high"           # 高優先度
    MEDIUM = "medium"       # 中優先度
    LOW = "low"             # 低優先度
    MINIMAL = "minimal"     # 最小監視


@dataclass
class UpdateSchedule:
    """更新スケジュール"""
    symbol: str
    interval_seconds: float
    next_update: datetime
    priority: SymbolPriority
    volatility_score: float = 0.0
    volume_score: float = 0.0
    prediction_accuracy: float = 0.0
    last_significant_change: Optional[datetime] = None


@dataclass
class OptimizationMetrics:
    """最適化メトリクス"""
    total_updates: int = 0
    successful_predictions: int = 0
    api_calls_saved: int = 0
    accuracy_improvement: float = 0.0
    processing_time_saved: float = 0.0
    symbols_processed: int = 0


class AutoUpdateOptimizer:
    """自動更新時間最適化システム"""

    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.schedules: Dict[str, UpdateSchedule] = {}
        self.metrics = OptimizationMetrics()
        self.market_condition = MarketCondition.NORMAL
        self.optimization_active = True

        # 動的更新間隔設定
        self.update_intervals = {
            MarketCondition.OPENING: {
                SymbolPriority.CRITICAL: 10,    # 10秒
                SymbolPriority.HIGH: 15,        # 15秒
                SymbolPriority.MEDIUM: 30,      # 30秒
                SymbolPriority.LOW: 60,         # 1分
                SymbolPriority.MINIMAL: 300     # 5分
            },
            MarketCondition.NORMAL: {
                SymbolPriority.CRITICAL: 20,    # 20秒
                SymbolPriority.HIGH: 30,        # 30秒
                SymbolPriority.MEDIUM: 60,      # 1分
                SymbolPriority.LOW: 120,        # 2分
                SymbolPriority.MINIMAL: 600     # 10分
            },
            MarketCondition.HIGH_VOLATILITY: {
                SymbolPriority.CRITICAL: 5,     # 5秒
                SymbolPriority.HIGH: 10,        # 10秒
                SymbolPriority.MEDIUM: 20,      # 20秒
                SymbolPriority.LOW: 45,         # 45秒
                SymbolPriority.MINIMAL: 180     # 3分
            },
            MarketCondition.LOW_VOLATILITY: {
                SymbolPriority.CRITICAL: 45,    # 45秒
                SymbolPriority.HIGH: 90,        # 1.5分
                SymbolPriority.MEDIUM: 180,     # 3分
                SymbolPriority.LOW: 300,        # 5分
                SymbolPriority.MINIMAL: 900     # 15分
            },
            MarketCondition.LUNCH: {
                SymbolPriority.CRITICAL: 120,   # 2分
                SymbolPriority.HIGH: 300,       # 5分
                SymbolPriority.MEDIUM: 600,     # 10分
                SymbolPriority.LOW: 1200,       # 20分
                SymbolPriority.MINIMAL: 1800    # 30分
            },
            MarketCondition.CLOSING: {
                SymbolPriority.CRITICAL: 8,     # 8秒
                SymbolPriority.HIGH: 12,        # 12秒
                SymbolPriority.MEDIUM: 25,      # 25秒
                SymbolPriority.LOW: 50,         # 50秒
                SymbolPriority.MINIMAL: 150     # 2.5分
            }
        }

        # 進捗追跡
        self.progress_bar: Optional[tqdm] = None
        self.setup_logging()

    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/auto_update_optimizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> dict:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"設定ファイルが見つかりません: {self.config_path}")
            return {}

    def initialize_schedules(self) -> None:
        """更新スケジュール初期化"""
        symbols = self.config.get('watchlist', {}).get('symbols', [])
        self.logger.info(f"更新スケジュール初期化開始: {len(symbols)}銘柄")

        self.progress_bar = tqdm(
            total=len(symbols),
            desc="スケジュール初期化",
            unit="銘柄"
        )

        for symbol_data in symbols:
            symbol = symbol_data['code']
            priority = self._determine_priority(symbol_data)

            schedule = UpdateSchedule(
                symbol=symbol,
                interval_seconds=self._get_optimal_interval(priority),
                next_update=datetime.now(),
                priority=priority
            )

            self.schedules[symbol] = schedule
            self.progress_bar.update(1)

        self.progress_bar.close()
        self.logger.info(f"更新スケジュール初期化完了: {len(self.schedules)}銘柄")

    def _determine_priority(self, symbol_data: dict) -> SymbolPriority:
        """銘柄優先度決定"""
        config_priority = symbol_data.get('priority', 'medium')
        sector = symbol_data.get('sector', '')

        # デイトレード・バイオテック銘柄は高優先度
        if sector in ['DayTrading', 'BioTech']:
            return SymbolPriority.CRITICAL

        # 設定ベース優先度
        priority_map = {
            'high': SymbolPriority.HIGH,
            'medium': SymbolPriority.MEDIUM,
            'low': SymbolPriority.LOW
        }

        return priority_map.get(config_priority, SymbolPriority.MEDIUM)

    def _get_optimal_interval(self, priority: SymbolPriority) -> float:
        """最適更新間隔取得"""
        intervals = self.update_intervals.get(
            self.market_condition,
            self.update_intervals[MarketCondition.NORMAL]
        )
        return intervals.get(priority, 60)

    def detect_market_condition(self) -> MarketCondition:
        """市場状況検知"""
        now = datetime.now()
        current_time = now.time()

        market_hours = self.config.get('watchlist', {}).get('market_hours', {})
        start_time = datetime.strptime(market_hours.get('start', '09:00'), '%H:%M').time()
        lunch_start = datetime.strptime(market_hours.get('lunch_start', '11:30'), '%H:%M').time()
        lunch_end = datetime.strptime(market_hours.get('lunch_end', '12:30'), '%H:%M').time()
        end_time = datetime.strptime(market_hours.get('end', '15:00'), '%H:%M').time()

        # 開場直後（30分間）
        opening_end = (datetime.combine(now.date(), start_time) + timedelta(minutes=30)).time()
        if start_time <= current_time <= opening_end:
            return MarketCondition.OPENING

        # 昼休み
        if lunch_start <= current_time <= lunch_end:
            return MarketCondition.LUNCH

        # 引け前（30分間）
        closing_start = (datetime.combine(now.date(), end_time) - timedelta(minutes=30)).time()
        if closing_start <= current_time <= end_time:
            return MarketCondition.CLOSING

        # ボラティリティベース判定（今後実装）
        # TODO: 実際のボラティリティデータ分析

        return MarketCondition.NORMAL

    async def optimize_update_frequency(self) -> None:
        """更新頻度最適化"""
        self.logger.info("更新頻度最適化開始")
        previous_condition = self.market_condition
        self.market_condition = self.detect_market_condition()

        if previous_condition != self.market_condition:
            self.logger.info(f"市場状況変化: {previous_condition.value} -> {self.market_condition.value}")
            await self._adjust_all_schedules()

        # 個別銘柄最適化
        await self._optimize_individual_symbols()

    async def _adjust_all_schedules(self) -> None:
        """全スケジュール調整"""
        self.logger.info("全銘柄スケジュール調整開始")

        progress_bar = tqdm(
            total=len(self.schedules),
            desc="スケジュール調整",
            unit="銘柄"
        )

        for symbol, schedule in self.schedules.items():
            new_interval = self._get_optimal_interval(schedule.priority)
            if new_interval != schedule.interval_seconds:
                schedule.interval_seconds = new_interval
                self.logger.debug(f"{symbol}: 更新間隔変更 {schedule.interval_seconds}秒")

            progress_bar.update(1)

        progress_bar.close()
        self.logger.info("全銘柄スケジュール調整完了")

    async def _optimize_individual_symbols(self) -> None:
        """個別銘柄最適化"""
        # ボラティリティ・出来高分析による動的調整
        for symbol, schedule in self.schedules.items():
            # TODO: 実際の価格データ分析
            # 仮想的なボラティリティスコア
            volatility_score = np.random.uniform(0.1, 2.0)

            if volatility_score > 1.5:
                # 高ボラティリティ時は更新頻度アップ
                schedule.interval_seconds *= 0.7
            elif volatility_score < 0.3:
                # 低ボラティリティ時は更新頻度ダウン
                schedule.interval_seconds *= 1.5

            schedule.volatility_score = volatility_score

    def get_next_updates(self, limit: int = 10) -> List[Tuple[str, datetime]]:
        """次回更新リスト取得"""
        now = datetime.now()
        updates = []

        for symbol, schedule in self.schedules.items():
            if schedule.next_update <= now:
                # 次回更新時刻計算
                schedule.next_update = now + timedelta(seconds=schedule.interval_seconds)
                updates.append((symbol, schedule.next_update))

        return sorted(updates, key=lambda x: x[1])[:limit]

    def update_prediction_accuracy(self, symbol: str, accuracy: float) -> None:
        """予測精度更新"""
        if symbol in self.schedules:
            self.schedules[symbol].prediction_accuracy = accuracy

            # 高精度銘柄は更新頻度を最適化
            if accuracy > 0.8:
                self.schedules[symbol].interval_seconds *= 1.2
            elif accuracy < 0.5:
                self.schedules[symbol].interval_seconds *= 0.8

            self.metrics.successful_predictions += 1 if accuracy > 0.7 else 0

    def generate_progress_report(self) -> dict:
        """進捗レポート生成"""
        now = datetime.now()

        # 優先度別銘柄数集計
        priority_counts = {}
        total_symbols = len(self.schedules)

        for priority in SymbolPriority:
            count = sum(1 for s in self.schedules.values() if s.priority == priority)
            priority_counts[priority.value] = count

        # 市場状況別統計
        avg_interval = np.mean([s.interval_seconds for s in self.schedules.values()])

        return {
            "timestamp": now.isoformat(),
            "market_condition": self.market_condition.value,
            "total_symbols": total_symbols,
            "priority_distribution": priority_counts,
            "average_update_interval": round(avg_interval, 2),
            "optimization_metrics": {
                "total_updates": self.metrics.total_updates,
                "successful_predictions": self.metrics.successful_predictions,
                "api_calls_saved": self.metrics.api_calls_saved,
                "accuracy_improvement": round(self.metrics.accuracy_improvement, 3),
                "processing_time_saved": round(self.metrics.processing_time_saved, 2)
            },
            "next_updates": self.get_next_updates(5)
        }

    def save_optimization_report(self, filepath: str = "reports/auto_update_optimization.json") -> None:
        """最適化レポート保存"""
        report = self.generate_progress_report()

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"最適化レポート保存: {filepath}")

    async def run_optimization_cycle(self) -> None:
        """最適化サイクル実行"""
        self.logger.info("自動更新最適化サイクル開始")

        # 初期化
        if not self.schedules:
            self.initialize_schedules()

        # メインループ
        while self.optimization_active:
            try:
                await self.optimize_update_frequency()

                # 次回更新予定表示
                next_updates = self.get_next_updates(3)
                for symbol, update_time in next_updates:
                    self.logger.debug(f"次回更新: {symbol} at {update_time}")

                # メトリクス更新
                self.metrics.total_updates += 1

                # レポート生成（1分毎）
                if self.metrics.total_updates % 60 == 0:
                    self.save_optimization_report()

                await asyncio.sleep(1)  # 1秒待機

            except KeyboardInterrupt:
                self.logger.info("最適化停止要求受信")
                self.optimization_active = False
            except Exception as e:
                self.logger.error(f"最適化エラー: {e}")
                await asyncio.sleep(5)  # エラー時は5秒待機

        self.logger.info("自動更新最適化サイクル終了")


def main():
    """メイン実行"""
    optimizer = AutoUpdateOptimizer()

    try:
        asyncio.run(optimizer.run_optimization_cycle())
    except KeyboardInterrupt:
        print("\n自動更新最適化システム停止")
    except Exception as e:
        print(f"システムエラー: {e}")


if __name__ == "__main__":
    main()