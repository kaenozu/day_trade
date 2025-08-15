#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動的銘柄選択システム（改善版）
デイトレード向けの適切な銘柄を動的に選択する

改善項目:
1. エラーハンドリングとロギングの強化
2. SQLクエリ構築の柔軟性向上
3. TOPIX500MasterManagerとの結合度低減
4. 外部設定ファイル対応
5. ロジック明確化
6. テストコード分離
"""

import sqlite3
import yaml
from typing import List, Dict, Optional, Any, Protocol
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

# 共通ユーティリティ
try:
    from ..utils.logging_config import get_context_logger
    from ..utils.encoding_utils import setup_windows_encoding
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

    def get_context_logger(name):
        import logging
        return logging.getLogger(name)

    def setup_windows_encoding():
        """Windows環境でのエンコーディング設定"""
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except Exception:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Windows環境対応
setup_windows_encoding()

# ロギング設定
logger = get_context_logger(__name__)


@dataclass
class SymbolSelectionCriteria:
    """銘柄選択基準"""
    min_market_cap: Optional[float] = 100_000_000_000  # 1000億円以上
    max_market_cap: Optional[float] = None
    min_liquidity_score: Optional[float] = 0.5  # 流動性スコア
    excluded_sectors: List[str] = field(default_factory=list)  # 除外セクター
    preferred_sectors: List[str] = field(default_factory=list)  # 優先セクター
    max_symbols: int = 50  # 最大銘柄数
    sort_criteria: str = "market_cap_desc"  # ソート基準
    additional_filters: List[str] = field(default_factory=list)  # 追加フィルター


@dataclass
class QueryComponents:
    """SQLクエリ構成要素"""
    select_clause: str
    from_clause: str
    where_conditions: List[str] = field(default_factory=list)
    parameters: List[Any] = field(default_factory=list)
    order_clause: Optional[str] = None
    limit_clause: Optional[str] = None


class DatabaseProvider(Protocol):
    """データベースプロバイダーインターフェース"""

    @property
    def db_path(self) -> str:
        """データベースパス"""
        ...

    def get_connection(self) -> sqlite3.Connection:
        """データベース接続を取得"""
        ...


class ConfigurationManager:
    """設定管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/symbol_selection_config.yaml")
        self.config = {}
        self._load_configuration()

    def _load_configuration(self):
        """設定ファイルを読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self._create_default_config()

        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")
            self._load_default_settings()

    def _create_default_config(self):
        """デフォルト設定ファイルを作成"""
        default_config = {
            'system': {'log_level': 'INFO', 'db_path': 'data/topix500_master.db'},
            'default_criteria': {
                'min_market_cap': 100_000_000_000,
                'max_symbols': 50
            },
            'strategies': {
                'liquid_trading': {
                    'min_market_cap': 300_000_000_000,
                    'max_symbols': 20,
                    'excluded_sectors': ['REIT', 'ETF']
                }
            }
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"デフォルト設定ファイルを作成: {self.config_path}")
            self.config = default_config
        except Exception as e:
            logger.error(f"デフォルト設定ファイル作成エラー: {e}")
            self._load_default_settings()

    def _load_default_settings(self):
        """デフォルト設定を読み込み"""
        self.config = {
            'system': {'log_level': 'INFO', 'db_path': 'data/topix500_master.db'},
            'default_criteria': {'min_market_cap': 100_000_000_000, 'max_symbols': 50},
            'error_handling': {'max_retries': 3, 'fallback_to_defaults': True}
        }
        logger.info("デフォルト設定を使用します")

    def get_criteria_for_strategy(self, strategy: str) -> SymbolSelectionCriteria:
        """戦略に応じた選択基準を取得"""
        try:
            strategy_config = self.config.get('strategies', {}).get(strategy, {})
            default_config = self.config.get('default_criteria', {})

            # デフォルト値とマージ
            merged_config = {**default_config, **strategy_config}

            return SymbolSelectionCriteria(
                min_market_cap=merged_config.get('min_market_cap'),
                max_market_cap=merged_config.get('max_market_cap'),
                min_liquidity_score=merged_config.get('min_liquidity_score'),
                excluded_sectors=merged_config.get('excluded_sectors', []),
                preferred_sectors=merged_config.get('preferred_sectors', []),
                max_symbols=merged_config.get('max_symbols', 50),
                sort_criteria=merged_config.get('sort_criteria', 'market_cap_desc')
            )

        except Exception as e:
            logger.error(f"戦略設定取得エラー ({strategy}): {e}")
            return SymbolSelectionCriteria()

    def get_system_setting(self, key: str, default=None):
        """システム設定値を取得"""
        return self.config.get('system', {}).get(key, default)


class QueryBuilder:
    """SQLクエリビルダー"""

    def __init__(self):
        self.reset()

    def reset(self):
        """クエリコンポーネントをリセット"""
        self.components = QueryComponents(
            select_clause="SELECT code, name, market_cap, topix_weight, sector_code, sector_name",
            from_clause="FROM topix500_master"
        )
        return self

    def add_base_filters(self) -> 'QueryBuilder':
        """基本フィルターを追加"""
        self.components.where_conditions.append("is_active = TRUE")
        return self

    def add_market_cap_filter(self, min_cap: Optional[float], max_cap: Optional[float]) -> 'QueryBuilder':
        """時価総額フィルターを追加"""
        if min_cap is not None:
            self.components.where_conditions.append("market_cap >= ?")
            self.components.parameters.append(min_cap)

        if max_cap is not None:
            self.components.where_conditions.append("market_cap <= ?")
            self.components.parameters.append(max_cap)

        return self

    def add_sector_filters(self, excluded: List[str], preferred: List[str]) -> 'QueryBuilder':
        """セクターフィルターを追加"""
        if excluded:
            placeholders = ",".join(["?"] * len(excluded))
            self.components.where_conditions.append(f"sector_code NOT IN ({placeholders})")
            self.components.parameters.extend(excluded)

        if preferred:
            placeholders = ",".join(["?"] * len(preferred))
            self.components.where_conditions.append(f"sector_code IN ({placeholders})")
            self.components.parameters.extend(preferred)

        return self

    def add_custom_filter(self, condition: str, params: List[Any] = None) -> 'QueryBuilder':
        """カスタムフィルターを追加"""
        self.components.where_conditions.append(condition)
        if params:
            self.components.parameters.extend(params)
        return self

    def set_order(self, sort_criteria: str) -> 'QueryBuilder':
        """ソート条件を設定"""
        sort_map = {
            "market_cap_desc": "market_cap DESC",
            "topix_weight_desc": "topix_weight DESC",
            "combined_score_desc": "(market_cap * COALESCE(topix_weight, 0.001)) DESC",
            "market_cap_asc": "market_cap ASC"
        }

        order_clause = sort_map.get(sort_criteria, sort_map["combined_score_desc"])
        self.components.order_clause = f"ORDER BY {order_clause}"
        return self

    def set_limit(self, limit: int) -> 'QueryBuilder':
        """LIMIT句を設定"""
        if limit > 0:
            self.components.limit_clause = f"LIMIT {limit}"
        return self

    def build(self) -> tuple[str, List[Any]]:
        """最終的なクエリを構築"""
        query_parts = [
            self.components.select_clause,
            self.components.from_clause
        ]

        if self.components.where_conditions:
            where_clause = "WHERE " + " AND ".join(self.components.where_conditions)
            query_parts.append(where_clause)

        if self.components.order_clause:
            query_parts.append(self.components.order_clause)

        if self.components.limit_clause:
            query_parts.append(self.components.limit_clause)

        query = "\n".join(query_parts)
        return query, self.components.parameters


class TOPIX500DatabaseProvider:
    """TOPIX500データベースプロバイダー実装"""

    def __init__(self, db_path: str):
        self._db_path = db_path

    @property
    def db_path(self) -> str:
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)


class DynamicSymbolSelector:
    """動的銘柄選択システム（改善版）"""

    def __init__(self,
                 db_provider: Optional[DatabaseProvider] = None,
                 config_manager: Optional[ConfigurationManager] = None):
        """
        初期化

        Args:
            db_provider: データベースプロバイダー
            config_manager: 設定管理
        """
        self.config_manager = config_manager or ConfigurationManager()

        # データベースプロバイダーの設定
        if db_provider is None:
            db_path = self.config_manager.get_system_setting('db_path', 'data/topix500_master.db')
            self.db_provider = TOPIX500DatabaseProvider(db_path)
        else:
            self.db_provider = db_provider

        self.query_builder = QueryBuilder()

        # エラーハンドリング設定
        error_config = self.config_manager.config.get('error_handling', {})
        self.max_retries = error_config.get('max_retries', 3)
        self.fallback_to_defaults = error_config.get('fallback_to_defaults', True)

        logger.info("動的銘柄選択システム初期化完了")

    @contextmanager
    def _get_db_connection(self):
        """データベース接続の管理"""
        conn = None
        try:
            conn = self.db_provider.get_connection()
            yield conn
        except Exception as e:
            logger.error(f"データベース接続エラー: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def _execute_query_with_retry(self, query: str, params: List[Any]) -> List[tuple]:
        """リトライ機能付きクエリ実行"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    results = cursor.fetchall()

                    if not results and attempt == 0:
                        logger.warning(f"クエリ結果が空です (試行 {attempt + 1})")

                    return results

            except Exception as e:
                last_exception = e
                logger.warning(f"クエリ実行エラー (試行 {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(1 * (attempt + 1))  # 指数バックオフ

        # すべての試行が失敗した場合
        error_msg = f"クエリ実行に失敗しました: {last_exception}"
        logger.error(error_msg)

        if self.fallback_to_defaults:
            logger.info("フォールバック処理を実行します")
            return []
        else:
            raise RuntimeError(error_msg) from last_exception

    def select_symbols_by_criteria(self, criteria: SymbolSelectionCriteria) -> List[str]:
        """
        指定基準による銘柄選択

        Args:
            criteria: 選択基準

        Returns:
            選択された銘柄コードリスト
        """
        try:
            # クエリビルダーを使用してクエリ構築
            query, params = (self.query_builder
                             .reset()
                             .add_base_filters()
                             .add_market_cap_filter(criteria.min_market_cap, criteria.max_market_cap)
                             .add_sector_filters(criteria.excluded_sectors, criteria.preferred_sectors)
                             .set_order(criteria.sort_criteria)
                             .set_limit(criteria.max_symbols)
                             .build())

            # 追加フィルターの適用
            for filter_condition in criteria.additional_filters:
                query += f" AND {filter_condition}"

            logger.debug(f"実行クエリ: {query}")
            logger.debug(f"パラメーター: {params}")

            # クエリ実行
            results = self._execute_query_with_retry(query, params)
            symbols = [row[0] for row in results]

            logger.info(f"銘柄選択完了: {len(symbols)}銘柄")
            if symbols:
                logger.info(f"選択銘柄例: {', '.join(symbols[:5])}")

            return symbols

        except Exception as e:
            logger.error(f"銘柄選択エラー: {e}")
            if self.fallback_to_defaults:
                logger.info("デフォルト銘柄リストを返します")
                return ["7203", "8306", "4751"]  # フォールバック
            else:
                raise RuntimeError(f"動的銘柄選択に失敗しました: {e}") from e

    def get_liquid_symbols(self, limit: int = 20) -> List[str]:
        """
        高流動性銘柄取得（デイトレード特化）

        Args:
            limit: 取得件数

        Returns:
            高流動性銘柄リスト
        """
        criteria = self.config_manager.get_criteria_for_strategy('liquid_trading')
        criteria.max_symbols = limit
        return self.select_symbols_by_criteria(criteria)

    def get_volatile_symbols(self, limit: int = 10) -> List[str]:
        """
        高ボラティリティ銘柄取得（デイトレード機会重視）

        Args:
            limit: 取得件数

        Returns:
            高ボラティリティ銘柄リスト
        """
        criteria = self.config_manager.get_criteria_for_strategy('volatile_trading')
        criteria.max_symbols = limit
        return self.select_symbols_by_criteria(criteria)

    def get_balanced_portfolio(self, limit: int = 15) -> List[str]:
        """
        バランス型ポートフォリオ銘柄（安定性重視）

        Args:
            limit: 取得件数

        Returns:
            バランス型銘柄リスト
        """
        criteria = self.config_manager.get_criteria_for_strategy('balanced_portfolio')
        criteria.max_symbols = limit
        return self.select_symbols_by_criteria(criteria)

    def get_sector_diversified_symbols(self, limit: int = 20) -> List[str]:
        """
        セクター分散された銘柄選択（改善版）

        各セクターから上位銘柄を均等に選択し、セクター分散を実現

        Args:
            limit: 取得件数

        Returns:
            セクター分散銘柄リスト
        """
        try:
            # セクター分散設定を取得
            diversification_config = self.config_manager.config.get('sector_diversification', {})
            max_per_sector = diversification_config.get('max_symbols_per_sector', 2)
            min_sectors = diversification_config.get('min_sectors', 5)
            market_cap_threshold = diversification_config.get('market_cap_threshold', 100_000_000_000)

            # セクター別上位銘柄を取得するクエリ
            query = """
                WITH ranked_symbols AS (
                    SELECT
                        code,
                        sector_code,
                        sector_name,
                        market_cap,
                        topix_weight,
                        ROW_NUMBER() OVER (
                            PARTITION BY sector_code
                            ORDER BY (market_cap * COALESCE(topix_weight, 0.001)) DESC
                        ) as sector_rank
                    FROM topix500_master
                    WHERE is_active = TRUE
                      AND market_cap >= ?
                ),
                sector_counts AS (
                    SELECT
                        sector_code,
                        COUNT(*) as symbol_count
                    FROM ranked_symbols
                    WHERE sector_rank <= ?
                    GROUP BY sector_code
                    HAVING COUNT(*) >= 1
                    ORDER BY COUNT(*) DESC
                )
                SELECT rs.code
                FROM ranked_symbols rs
                INNER JOIN sector_counts sc ON rs.sector_code = sc.sector_code
                WHERE rs.sector_rank <= ?
                ORDER BY sc.symbol_count DESC, rs.sector_code, rs.sector_rank
                LIMIT ?
            """

            params = [market_cap_threshold, max_per_sector, max_per_sector, limit]

            logger.debug(f"セクター分散クエリ実行: max_per_sector={max_per_sector}, limit={limit}")

            results = self._execute_query_with_retry(query, params)
            symbols = [row[0] for row in results]

            # 最小セクター数チェック
            if len(set(self._get_sectors_for_symbols(symbols))) < min_sectors:
                logger.warning(f"セクター数が最小要件({min_sectors})を下回りました")

            logger.info(f"セクター分散銘柄選択完了: {len(symbols)}銘柄")

            if not symbols:
                logger.warning("セクター分散銘柄が見つかりません。フォールバックを実行します")
                return self._get_fallback_diversified_symbols(limit)

            return symbols

        except Exception as e:
            logger.error(f"セクター分散銘柄選択エラー: {e}")

            if self.fallback_to_defaults:
                return self._get_fallback_diversified_symbols(limit)
            else:
                raise RuntimeError(f"セクター分散銘柄選択に失敗しました: {e}") from e

    def _get_sectors_for_symbols(self, symbols: List[str]) -> List[str]:
        """銘柄のセクター情報を取得"""
        try:
            if not symbols:
                return []

            placeholders = ",".join(["?"] * len(symbols))
            query = f"SELECT DISTINCT sector_code FROM topix500_master WHERE code IN ({placeholders})"

            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, symbols)
                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"セクター情報取得エラー: {e}")
            return []

    def _get_fallback_diversified_symbols(self, limit: int) -> List[str]:
        """フォールバック用のセクター分散銘柄"""
        logger.info("フォールバック: 簡易セクター分散銘柄を返します")

        # 基本的な高流動性銘柄を取得
        fallback_criteria = SymbolSelectionCriteria(
            min_market_cap=200_000_000_000,
            max_symbols=limit,
            sort_criteria="combined_score_desc"
        )

        return self.select_symbols_by_criteria(fallback_criteria)

    def get_symbols_by_strategy(self, strategy_name: str, limit: Optional[int] = None) -> List[str]:
        """
        戦略名による銘柄選択

        Args:
            strategy_name: 戦略名
            limit: 取得件数（Noneの場合は設定値を使用）

        Returns:
            選択された銘柄リスト
        """
        try:
            criteria = self.config_manager.get_criteria_for_strategy(strategy_name)

            if limit is not None:
                criteria.max_symbols = limit

            return self.select_symbols_by_criteria(criteria)

        except Exception as e:
            logger.error(f"戦略別銘柄選択エラー ({strategy_name}): {e}")
            return []

    def validate_symbol_selection(self, symbols: List[str]) -> Dict[str, Any]:
        """
        銘柄選択結果の検証

        Args:
            symbols: 検証対象銘柄リスト

        Returns:
            検証結果の詳細情報
        """
        try:
            if not symbols:
                return {"valid": False, "reason": "銘柄リストが空です"}

            # 銘柄の詳細情報を取得
            placeholders = ",".join(["?"] * len(symbols))
            query = f"""
                SELECT code, name, market_cap, sector_code, sector_name
                FROM topix500_master
                WHERE code IN ({placeholders}) AND is_active = TRUE
            """

            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, symbols)
                results = cursor.fetchall()

            # 検証結果の構築
            valid_symbols = [row[0] for row in results]
            invalid_symbols = [symbol for symbol in symbols if symbol not in valid_symbols]

            sectors = list(set(row[3] for row in results))
            avg_market_cap = sum(row[2] or 0 for row in results) / len(results) if results else 0

            validation_result = {
                "valid": len(invalid_symbols) == 0,
                "total_symbols": len(symbols),
                "valid_symbols": len(valid_symbols),
                "invalid_symbols": invalid_symbols,
                "sector_count": len(sectors),
                "sectors": sectors,
                "avg_market_cap": avg_market_cap,
                "details": results
            }

            logger.info(f"銘柄選択検証完了: {validation_result['valid_symbols']}/{validation_result['total_symbols']} 有効")

            return validation_result

        except Exception as e:
            logger.error(f"銘柄選択検証エラー: {e}")
            return {"valid": False, "reason": f"検証エラー: {e}"}


# ファクトリー関数
def create_symbol_selector(config_path: Optional[str] = None,
                           db_path: Optional[str] = None) -> DynamicSymbolSelector:
    """
    DynamicSymbolSelectorインスタンスの作成

    Args:
        config_path: 設定ファイルパス
        db_path: データベースファイルパス

    Returns:
        DynamicSymbolSelectorインスタンス
    """
    config_manager = ConfigurationManager(Path(config_path) if config_path else None)

    if db_path:
        db_provider = TOPIX500DatabaseProvider(db_path)
        return DynamicSymbolSelector(db_provider=db_provider, config_manager=config_manager)
    else:
        return DynamicSymbolSelector(config_manager=config_manager)


if __name__ == "__main__":
    # 基本動作確認は別ファイルに分離
    pass