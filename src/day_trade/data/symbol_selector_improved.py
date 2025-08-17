#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Symbol Selector - 改善版銘柄選択システム
Issue #854対応：銘柄選択システムの柔軟性と堅牢性強化

主要改善点:
1. エラーハンドリングとロギングの強化
2. SQLクエリ構築の柔軟性向上と保守性
3. TOPIX500MasterManagerとの結合度の低減
4. 銘柄選択基準の外部設定化
5. セクター分散銘柄選択ロジックの明確化
"""

import asyncio
import sqlite3
import pandas as pd
import numpy as np
import logging
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
from abc import ABC, abstractmethod

# Windows環境での文字化け対策
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

# 堅牢なロギング設定
def get_robust_logger(name: str) -> logging.Logger:
    """堅牢なロガー取得"""
    try:
        # 通常のロギング設定を試行
        from ..utils.logging_config import get_context_logger
        return get_context_logger(name)
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        # フォールバック：標準ロギング
        sys.stderr.write(f"Warning: Failed to import context logger ({e}), using standard logging\n")
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    except Exception as e:
        # 致命的エラー：最低限のロギング
        sys.stderr.write(f"Critical: Logger initialization failed ({e}), using stderr output\n")

        class StderrLogger:
            def info(self, msg): sys.stderr.write(f"INFO: {msg}\n")
            def warning(self, msg): sys.stderr.write(f"WARNING: {msg}\n")
            def error(self, msg): sys.stderr.write(f"ERROR: {msg}\n")
            def debug(self, msg): sys.stderr.write(f"DEBUG: {msg}\n")

        return StderrLogger()


class MarketSegment(Enum):
    """市場セグメント"""
    MEGA_CAP = "mega_cap"
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"


class SectorType(Enum):
    """セクター種別"""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    CONSUMER = "consumer"
    INDUSTRIAL = "industrial"
    ENERGY = "energy"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"


@dataclass
class SymbolSelectionCriteria:
    """銘柄選択基準"""
    min_volume: int = 1000000
    max_volume: int = 100000000
    min_price: float = 100.0
    max_price: float = 10000.0
    min_market_cap: float = 1000000000  # 10億円
    max_market_cap: float = 10000000000000  # 10兆円
    min_volatility: float = 0.01  # 1%
    max_volatility: float = 0.05  # 5%
    liquidity_threshold: float = 0.7
    sector_diversification: bool = True
    max_symbols_per_sector: int = 2
    exclude_symbols: List[str] = field(default_factory=list)
    include_symbols: List[str] = field(default_factory=list)


@dataclass
class SymbolInfo:
    """銘柄情報"""
    symbol: str
    name: str
    sector: str
    market_cap: float
    price: float
    volume: int
    volatility: float
    liquidity_score: float
    topix_weight: float
    selection_score: float


@dataclass
class SelectionResult:
    """選択結果"""
    symbols: List[SymbolInfo]
    criteria_used: SymbolSelectionCriteria
    selection_time: datetime
    total_candidates: int
    selected_count: int
    selection_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# 抽象インターフェース定義
@runtime_checkable
class SymbolDataProvider(Protocol):
    """銘柄データプロバイダーインターフェース"""

    async def get_symbols_by_criteria(self, criteria: SymbolSelectionCriteria) -> List[SymbolInfo]:
        """基準による銘柄取得"""
        ...

    async def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """単一銘柄情報取得"""
        ...

    async def get_sectors(self) -> List[str]:
        """セクター一覧取得"""
        ...


class SymbolSelectionConfigManager:
    """銘柄選択設定管理"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = get_robust_logger(__name__)
        self.config_path = config_path or Path("config/symbol_selection_criteria.yaml")
        self.criteria_sets = self._load_criteria_sets()

    def _load_criteria_sets(self) -> Dict[str, SymbolSelectionCriteria]:
        """設定ファイルから基準セット読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                criteria_sets = {}
                for name, data in config_data.get('criteria_sets', {}).items():
                    # exclude_symbols, include_symbolsのデフォルト値設定
                    data.setdefault('exclude_symbols', [])
                    data.setdefault('include_symbols', [])
                    criteria_sets[name] = SymbolSelectionCriteria(**data)

                self.logger.info(f"Loaded {len(criteria_sets)} criteria sets from {self.config_path}")
                return criteria_sets
            else:
                self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_criteria_sets()

        except Exception as e:
            error_msg = f"Failed to load selection criteria: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(
                f"Unable to load symbol selection criteria from {self.config_path}. "
                f"Error: {e}. "
                f"Please check the file exists and has valid YAML format. "
                f"You can create a default config file or use built-in defaults."
            ) from e

    def _get_default_criteria_sets(self) -> Dict[str, SymbolSelectionCriteria]:
        """デフォルト基準セット"""
        return {
            'liquid': SymbolSelectionCriteria(
                min_volume=5000000,
                max_volume=50000000,
                min_price=500.0,
                max_price=5000.0,
                liquidity_threshold=0.8,
                max_symbols_per_sector=3
            ),
            'volatile': SymbolSelectionCriteria(
                min_volume=1000000,
                max_volume=100000000,
                min_volatility=0.02,
                max_volatility=0.08,
                liquidity_threshold=0.6,
                max_symbols_per_sector=2
            ),
            'balanced': SymbolSelectionCriteria(
                min_volume=2000000,
                max_volume=80000000,
                min_price=300.0,
                max_price=8000.0,
                min_volatility=0.015,
                max_volatility=0.06,
                liquidity_threshold=0.7,
                sector_diversification=True,
                max_symbols_per_sector=2
            ),
            'conservative': SymbolSelectionCriteria(
                min_volume=10000000,
                max_volume=200000000,
                min_price=1000.0,
                max_price=3000.0,
                min_market_cap=5000000000,  # 50億円以上
                min_volatility=0.005,
                max_volatility=0.03,
                liquidity_threshold=0.85,
                max_symbols_per_sector=1
            )
        }

    def get_criteria(self, criteria_name: str) -> SymbolSelectionCriteria:
        """基準取得"""
        if criteria_name not in self.criteria_sets:
            available = list(self.criteria_sets.keys())
            raise ValueError(
                f"Unknown criteria set '{criteria_name}'. "
                f"Available criteria sets: {available}. "
                f"Please check your configuration file or use one of the predefined sets."
            )
        return self.criteria_sets[criteria_name]

    def list_criteria_sets(self) -> List[str]:
        """利用可能な基準セット一覧"""
        return list(self.criteria_sets.keys())

    def save_criteria_sets(self):
        """基準セットを設定ファイルに保存"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'criteria_sets': {
                    name: {
                        'min_volume': criteria.min_volume,
                        'max_volume': criteria.max_volume,
                        'min_price': criteria.min_price,
                        'max_price': criteria.max_price,
                        'min_market_cap': criteria.min_market_cap,
                        'max_market_cap': criteria.max_market_cap,
                        'min_volatility': criteria.min_volatility,
                        'max_volatility': criteria.max_volatility,
                        'liquidity_threshold': criteria.liquidity_threshold,
                        'sector_diversification': criteria.sector_diversification,
                        'max_symbols_per_sector': criteria.max_symbols_per_sector,
                        'exclude_symbols': criteria.exclude_symbols,
                        'include_symbols': criteria.include_symbols
                    }
                    for name, criteria in self.criteria_sets.items()
                }
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"Saved criteria sets to {self.config_path}")

        except Exception as e:
            error_msg = f"Failed to save criteria sets: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(f"Unable to save configuration to {self.config_path}: {e}") from e


class SQLQueryBuilder:
    """SQL クエリビルダー（保守性と安全性向上）"""

    def __init__(self):
        self.logger = get_robust_logger(__name__)

    def build_symbol_selection_query(self, criteria: SymbolSelectionCriteria,
                                   limit: Optional[int] = None,
                                   order_by: str = "volume DESC") -> Tuple[str, Dict[str, Any]]:
        """銘柄選択クエリ構築"""

        # ベースクエリ
        base_query = """
        SELECT
            symbol,
            name,
            sector,
            market_cap,
            price,
            volume,
            volatility,
            liquidity_score,
            COALESCE(topix_weight, 0.001) as topix_weight
        FROM symbols
        WHERE 1=1
        """

        # パラメータ辞書
        params = {}
        conditions = []

        # 出来高条件
        if criteria.min_volume > 0:
            conditions.append("volume >= :min_volume")
            params['min_volume'] = criteria.min_volume

        if criteria.max_volume < float('inf'):
            conditions.append("volume <= :max_volume")
            params['max_volume'] = criteria.max_volume

        # 価格条件
        if criteria.min_price > 0:
            conditions.append("price >= :min_price")
            params['min_price'] = criteria.min_price

        if criteria.max_price < float('inf'):
            conditions.append("price <= :max_price")
            params['max_price'] = criteria.max_price

        # 時価総額条件
        if criteria.min_market_cap > 0:
            conditions.append("market_cap >= :min_market_cap")
            params['min_market_cap'] = criteria.min_market_cap

        if criteria.max_market_cap < float('inf'):
            conditions.append("market_cap <= :max_market_cap")
            params['max_market_cap'] = criteria.max_market_cap

        # ボラティリティ条件
        if criteria.min_volatility > 0:
            conditions.append("volatility >= :min_volatility")
            params['min_volatility'] = criteria.min_volatility

        if criteria.max_volatility < float('inf'):
            conditions.append("volatility <= :max_volatility")
            params['max_volatility'] = criteria.max_volatility

        # 流動性条件
        if criteria.liquidity_threshold > 0:
            conditions.append("liquidity_score >= :liquidity_threshold")
            params['liquidity_threshold'] = criteria.liquidity_threshold

        # 除外銘柄
        if criteria.exclude_symbols:
            placeholders = ','.join(f':exclude_{i}' for i in range(len(criteria.exclude_symbols)))
            conditions.append(f"symbol NOT IN ({placeholders})")
            for i, symbol in enumerate(criteria.exclude_symbols):
                params[f'exclude_{i}'] = symbol

        # 含む銘柄（他の条件より優先）
        if criteria.include_symbols:
            placeholders = ','.join(f':include_{i}' for i in range(len(criteria.include_symbols)))
            include_condition = f"symbol IN ({placeholders})"
            for i, symbol in enumerate(criteria.include_symbols):
                params[f'include_{i}'] = symbol

            # 含む銘柄は他の条件と OR で結合
            if conditions:
                all_conditions = f"({' AND '.join(conditions)}) OR ({include_condition})"
            else:
                all_conditions = include_condition
        else:
            all_conditions = ' AND '.join(conditions) if conditions else "1=1"

        # クエリ組み立て
        query = f"{base_query} AND {all_conditions}"

        # ソート
        query += f" ORDER BY {order_by}"

        # 制限
        if limit and limit > 0:
            query += " LIMIT :limit"
            params['limit'] = limit

        self.logger.debug(f"Built query with {len(conditions)} conditions")
        return query, params

    def build_sector_diversified_query(self, criteria: SymbolSelectionCriteria,
                                     limit: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """セクター分散クエリ構築（改善版）"""

        # ウィンドウ関数を使用したセクター別上位選択
        query = """
        WITH sector_ranked AS (
            SELECT
                symbol,
                name,
                sector,
                market_cap,
                price,
                volume,
                volatility,
                liquidity_score,
                COALESCE(topix_weight, 0.001) as topix_weight,
                ROW_NUMBER() OVER (
                    PARTITION BY sector
                    ORDER BY volume * liquidity_score * COALESCE(topix_weight, 0.001) DESC
                ) as sector_rank
            FROM symbols
            WHERE 1=1
        """

        # 基本条件を追加
        base_query, params = self.build_symbol_selection_query(criteria)

        # 基本条件をWITH句に統合
        base_conditions = base_query.split("WHERE 1=1")[1].split("ORDER BY")[0].strip()
        if base_conditions and base_conditions != "AND":
            query += base_conditions

        # セクター別選択条件
        query += f"""
        )
        SELECT *
        FROM sector_ranked
        WHERE sector_rank <= :max_per_sector
        ORDER BY sector, sector_rank
        """

        params['max_per_sector'] = criteria.max_symbols_per_sector

        # 全体制限
        if limit and limit > 0:
            query += " LIMIT :total_limit"
            params['total_limit'] = limit

        return query, params


class TOPIX500DataProvider:
    """TOPIX500データプロバイダー（インターフェース実装）"""

    def __init__(self, db_path: str = "data/stock_master.db"):
        self.logger = get_robust_logger(__name__)
        self.db_path = db_path
        self.query_builder = SQLQueryBuilder()

    async def get_symbols_by_criteria(self, criteria: SymbolSelectionCriteria) -> List[SymbolInfo]:
        """基準による銘柄取得"""
        try:
            if criteria.sector_diversification:
                query, params = self.query_builder.build_sector_diversified_query(criteria, limit=50)
            else:
                query, params = self.query_builder.build_symbol_selection_query(criteria, limit=50)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(query, params)
                rows = cursor.fetchall()

                symbols = []
                for row in rows:
                    symbol_info = SymbolInfo(
                        symbol=row['symbol'],
                        name=row.get('name', ''),
                        sector=row.get('sector', ''),
                        market_cap=float(row.get('market_cap', 0)),
                        price=float(row.get('price', 0)),
                        volume=int(row.get('volume', 0)),
                        volatility=float(row.get('volatility', 0)),
                        liquidity_score=float(row.get('liquidity_score', 0)),
                        topix_weight=float(row.get('topix_weight', 0.001)),
                        selection_score=0.0  # 後で計算
                    )
                    symbols.append(symbol_info)

                # 選択スコア計算
                self._calculate_selection_scores(symbols, criteria)

                self.logger.info(f"Retrieved {len(symbols)} symbols using criteria")
                return symbols

        except sqlite3.Error as e:
            error_msg = (
                f"Database error while retrieving symbols: {e}. "
                f"Please check if the database file '{self.db_path}' exists and is accessible. "
                f"If the database is corrupted, try regenerating the symbol database."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        except Exception as e:
            error_msg = (
                f"Unexpected error during symbol retrieval: {e}. "
                f"This may be due to invalid criteria parameters or system issues. "
                f"Please check the criteria values and system status."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """単一銘柄情報取得"""
        try:
            query = """
            SELECT
                symbol, name, sector, market_cap, price, volume,
                volatility, liquidity_score, COALESCE(topix_weight, 0.001) as topix_weight
            FROM symbols
            WHERE symbol = ?
            """

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(query, (symbol,))
                row = cursor.fetchone()

                if row:
                    return SymbolInfo(
                        symbol=row['symbol'],
                        name=row.get('name', ''),
                        sector=row.get('sector', ''),
                        market_cap=float(row.get('market_cap', 0)),
                        price=float(row.get('price', 0)),
                        volume=int(row.get('volume', 0)),
                        volatility=float(row.get('volatility', 0)),
                        liquidity_score=float(row.get('liquidity_score', 0)),
                        topix_weight=float(row.get('topix_weight', 0.001)),
                        selection_score=0.0
                    )

                return None

        except Exception as e:
            error_msg = f"Failed to retrieve symbol info for '{symbol}': {e}"
            self.logger.error(error_msg)
            return None

    async def get_sectors(self) -> List[str]:
        """セクター一覧取得"""
        try:
            query = "SELECT DISTINCT sector FROM symbols WHERE sector IS NOT NULL ORDER BY sector"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()

                return [row[0] for row in rows]

        except Exception as e:
            error_msg = f"Failed to retrieve sectors: {e}"
            self.logger.error(error_msg)
            return []

    def _calculate_selection_scores(self, symbols: List[SymbolInfo], criteria: SymbolSelectionCriteria):
        """選択スコア計算"""
        if not symbols:
            return

        # 正規化用の最大・最小値
        volumes = [s.volume for s in symbols]
        liquidity_scores = [s.liquidity_score for s in symbols]
        topix_weights = [s.topix_weight for s in symbols]

        max_volume = max(volumes) if volumes else 1
        max_liquidity = max(liquidity_scores) if liquidity_scores else 1
        max_topix = max(topix_weights) if topix_weights else 1

        for symbol in symbols:
            # 各要素を0-1に正規化
            volume_score = symbol.volume / max_volume
            liquidity_score = symbol.liquidity_score / max_liquidity
            topix_score = symbol.topix_weight / max_topix

            # 重み付き合計（設定可能）
            symbol.selection_score = (
                volume_score * 0.4 +
                liquidity_score * 0.4 +
                topix_score * 0.2
            ) * 100  # 0-100スケール


class ImprovedSymbolSelector:
    """改善版銘柄選択システム"""

    def __init__(self,
                 data_provider: Optional[SymbolDataProvider] = None,
                 config_path: Optional[Path] = None):
        self.logger = get_robust_logger(__name__)

        # 依存性注入：プロバイダーが指定されない場合はデフォルト使用
        self.data_provider = data_provider or TOPIX500DataProvider()

        # 設定管理
        self.config_manager = SymbolSelectionConfigManager(config_path)

        # 選択履歴
        self.selection_history: List[SelectionResult] = []

        self.logger.info("Improved Symbol Selector initialized")

    async def select_daytrading_symbols(self,
                                      criteria_name: str = "balanced",
                                      limit: int = 20,
                                      custom_criteria: Optional[SymbolSelectionCriteria] = None) -> SelectionResult:
        """デイトレード用銘柄選択（改善版）"""

        start_time = datetime.now()

        try:
            # 基準取得
            if custom_criteria:
                criteria = custom_criteria
                method = f"custom_{criteria_name}"
            else:
                try:
                    criteria = self.config_manager.get_criteria(criteria_name)
                    method = criteria_name
                except ValueError as e:
                    # 利用可能な基準セットを含む詳細エラー
                    available_sets = self.config_manager.list_criteria_sets()
                    raise ValueError(
                        f"Invalid criteria name '{criteria_name}'. "
                        f"Available criteria sets: {available_sets}. "
                        f"Original error: {e}"
                    ) from e

            self.logger.info(f"Starting symbol selection with criteria: {method}")

            # 銘柄取得
            all_symbols = await self.data_provider.get_symbols_by_criteria(criteria)

            if not all_symbols:
                self.logger.warning("No symbols found matching the criteria")
                return SelectionResult(
                    symbols=[],
                    criteria_used=criteria,
                    selection_time=start_time,
                    total_candidates=0,
                    selected_count=0,
                    selection_method=method,
                    metadata={'warning': 'No symbols matched the criteria'}
                )

            # 制限適用
            selected_symbols = all_symbols[:limit] if limit > 0 else all_symbols

            # 結果作成
            result = SelectionResult(
                symbols=selected_symbols,
                criteria_used=criteria,
                selection_time=start_time,
                total_candidates=len(all_symbols),
                selected_count=len(selected_symbols),
                selection_method=method,
                metadata={
                    'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'criteria_name': criteria_name,
                    'limit_applied': limit,
                    'sector_diversification': criteria.sector_diversification
                }
            )

            # 履歴に追加
            self.selection_history.append(result)

            self.logger.info(
                f"Selected {len(selected_symbols)} symbols from {len(all_symbols)} candidates "
                f"using {method} criteria in {result.metadata['processing_time_ms']:.1f}ms"
            )

            return result

        except Exception as e:
            error_msg = (
                f"Symbol selection failed for criteria '{criteria_name}': {e}. "
                f"This may be due to database connectivity issues, invalid criteria, "
                f"or system configuration problems. Please check the logs for more details."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    async def get_liquid_symbols(self, limit: int = 10) -> SelectionResult:
        """流動性重視銘柄選択"""
        return await self.select_daytrading_symbols("liquid", limit)

    async def get_volatile_symbols(self, limit: int = 10) -> SelectionResult:
        """ボラティリティ重視銘柄選択"""
        return await self.select_daytrading_symbols("volatile", limit)

    async def get_balanced_portfolio(self, limit: int = 20) -> SelectionResult:
        """バランス型ポートフォリオ銘柄選択"""
        return await self.select_daytrading_symbols("balanced", limit)

    async def get_conservative_symbols(self, limit: int = 15) -> SelectionResult:
        """保守的銘柄選択"""
        return await self.select_daytrading_symbols("conservative", limit)

    async def get_sector_diversified_symbols(self,
                                           limit: int = 30,
                                           max_per_sector: int = 2) -> SelectionResult:
        """セクター分散銘柄選択（ロジック明確化）"""

        try:
            # カスタム基準作成
            custom_criteria = SymbolSelectionCriteria(
                sector_diversification=True,
                max_symbols_per_sector=max_per_sector,
                min_volume=2000000,
                liquidity_threshold=0.7
            )

            result = await self.select_daytrading_symbols(
                "sector_diversified",
                limit,
                custom_criteria
            )

            # セクター分散の詳細情報を追加
            sector_count = {}
            for symbol in result.symbols:
                sector_count[symbol.sector] = sector_count.get(symbol.sector, 0) + 1

            result.metadata.update({
                'sector_distribution': sector_count,
                'sectors_count': len(sector_count),
                'max_per_sector_limit': max_per_sector,
                'diversification_applied': True
            })

            self.logger.info(
                f"Sector diversified selection: {len(sector_count)} sectors, "
                f"max {max(sector_count.values()) if sector_count else 0} symbols per sector"
            )

            return result

        except Exception as e:
            error_msg = (
                f"Sector diversified selection failed: {e}. "
                f"This may be due to insufficient data or database issues. "
                f"Please check if sector information is available in the database."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_selection_history(self, limit: int = 10) -> List[SelectionResult]:
        """選択履歴取得"""
        return self.selection_history[-limit:] if limit > 0 else self.selection_history

    def get_criteria_sets(self) -> List[str]:
        """利用可能な基準セット一覧"""
        return self.config_manager.list_criteria_sets()

    async def validate_symbol(self, symbol: str) -> bool:
        """銘柄存在確認"""
        try:
            symbol_info = await self.data_provider.get_symbol_info(symbol)
            return symbol_info is not None
        except Exception as e:
            self.logger.error(f"Symbol validation failed for {symbol}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        if not self.selection_history:
            return {'total_selections': 0}

        total_selections = len(self.selection_history)
        total_symbols = sum(r.selected_count for r in self.selection_history)
        avg_processing_time = np.mean([
            r.metadata.get('processing_time_ms', 0)
            for r in self.selection_history
        ])

        method_counts = {}
        for result in self.selection_history:
            method = result.selection_method
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'total_selections': total_selections,
            'total_symbols_selected': total_symbols,
            'avg_symbols_per_selection': total_symbols / total_selections,
            'avg_processing_time_ms': avg_processing_time,
            'method_usage': method_counts,
            'last_selection': self.selection_history[-1].selection_time.isoformat()
        }


# テスト関数
async def test_improved_symbol_selector():
    """改善版銘柄選択システムのテスト"""
    print("=== Improved Symbol Selector Test ===")

    try:
        # モックデータプロバイダー
        class MockDataProvider:
            async def get_symbols_by_criteria(self, criteria):
                # テスト用のサンプルデータ
                return [
                    SymbolInfo(
                        symbol=f"TEST{i:04d}",
                        name=f"テスト銘柄{i}",
                        sector=f"セクター{i%5}",
                        market_cap=1000000000 + i * 100000000,
                        price=1000 + i * 100,
                        volume=1000000 + i * 100000,
                        volatility=0.02 + i * 0.001,
                        liquidity_score=0.7 + i * 0.01,
                        topix_weight=0.001 + i * 0.0001,
                        selection_score=0.0
                    )
                    for i in range(20)
                ]

            async def get_symbol_info(self, symbol):
                return SymbolInfo(
                    symbol=symbol, name=f"テスト{symbol}", sector="テスト",
                    market_cap=1000000000, price=1000, volume=1000000,
                    volatility=0.02, liquidity_score=0.7, topix_weight=0.001,
                    selection_score=0.0
                )

            async def get_sectors(self):
                return ["セクター0", "セクター1", "セクター2", "セクター3", "セクター4"]

        # テスト用選択器初期化
        selector = ImprovedSymbolSelector(data_provider=MockDataProvider())
        print("✓ Symbol selector initialized with mock data provider")

        # 利用可能な基準セット確認
        criteria_sets = selector.get_criteria_sets()
        print(f"✓ Available criteria sets: {', '.join(criteria_sets)}")

        # 各種選択テスト
        test_methods = [
            ("liquid", selector.get_liquid_symbols),
            ("volatile", selector.get_volatile_symbols),
            ("balanced", selector.get_balanced_portfolio),
            ("conservative", selector.get_conservative_symbols)
        ]

        for method_name, method in test_methods:
            result = await method(10)
            print(f"✓ {method_name} selection: {result.selected_count} symbols selected "
                  f"from {result.total_candidates} candidates "
                  f"({result.metadata.get('processing_time_ms', 0):.1f}ms)")

        # セクター分散選択テスト
        sector_result = await selector.get_sector_diversified_symbols(20, 2)
        print(f"✓ Sector diversified selection: {sector_result.selected_count} symbols, "
              f"{sector_result.metadata.get('sectors_count', 0)} sectors")

        # 統計情報確認
        stats = selector.get_statistics()
        print(f"✓ Statistics: {stats['total_selections']} selections, "
              f"avg {stats['avg_processing_time_ms']:.1f}ms")

        # 銘柄バリデーション
        is_valid = await selector.validate_symbol("TEST0001")
        print(f"✓ Symbol validation test: {is_valid}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    asyncio.run(test_improved_symbol_selector())