#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Symbol Manager - 動的監視対象銘柄管理

監視対象銘柄の動的選択と管理を行うモジュール
SymbolSelectorとの連携により、市場状況に応じた銘柄選択を実現
"""

import logging
from datetime import datetime
from typing import List, Set
from .config import EnhancedPerformanceConfigManager

logger = logging.getLogger(__name__)

# 外部システム連携チェック
try:
    from symbol_selector import SymbolSelector
    SYMBOL_SELECTOR_AVAILABLE = True
except ImportError:
    SYMBOL_SELECTOR_AVAILABLE = False


class DynamicSymbolManager:
    """動的監視対象銘柄管理クラス
    
    監視対象銘柄の動的選択と管理を行います。
    設定ファイルからの静的銘柄とSymbolSelectorからの動的銘柄を統合して管理します。
    
    Attributes:
        config_manager: 設定管理インスタンス
        symbol_selector: SymbolSelectorインスタンス
        last_update: 最終更新時刻
        current_symbols: 現在の監視対象銘柄セット
    """

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager
        self.symbol_selector = None
        self.last_update = None
        self.current_symbols = set()

        # symbol_selector連携
        if SYMBOL_SELECTOR_AVAILABLE:
            try:
                self.symbol_selector = SymbolSelector()
                logger.info("SymbolSelectorとの連携を開始しました")
            except Exception as e:
                logger.warning(f"SymbolSelector初期化失敗: {e}")

    def get_monitoring_symbols(self) -> List[str]:
        """現在の監視対象銘柄を取得
        
        静的設定と動的選択を組み合わせて監視対象銘柄リストを生成します。
        
        Returns:
            監視対象銘柄のリスト
        """
        config_symbols = self.config_manager.config.get('monitoring_symbols', {})

        # 基本監視銘柄
        symbols = []
        symbols.extend(config_symbols.get('primary_symbols', []))
        symbols.extend(config_symbols.get('secondary_symbols', []))

        # 動的銘柄選択が有効な場合
        dynamic_config = config_symbols.get('dynamic_selection', {})
        if dynamic_config.get('enabled', False) and self.symbol_selector:
            try:
                # 動的銘柄選択実行
                criteria = dynamic_config.get('selection_criteria', {})
                dynamic_symbols = self._get_dynamic_symbols()
                symbols.extend(dynamic_symbols)
                logger.info(f"動的銘柄選択で{len(dynamic_symbols)}銘柄を追加")
            except Exception as e:
                logger.error(f"動的銘柄選択エラー: {e}")

        # 監視設定からも取得
        monitoring_config = self.config_manager.get_monitoring_config()
        default_symbols = monitoring_config.get('default_symbols', ["7203", "8306", "4751"])
        symbols.extend(default_symbols)

        # 重複除去してソート
        unique_symbols = sorted(list(set(symbols)))
        logger.debug(f"監視対象銘柄: {unique_symbols}")
        return unique_symbols

    def _get_dynamic_symbols(self) -> Set[str]:
        """動的監視銘柄を取得
        
        SymbolSelectorを使用して動的に銘柄を選択します。
        
        Returns:
            動的に選択された銘柄のセット
        """
        dynamic_symbols = set()

        try:
            if self.symbol_selector:
                # symbol_selectorから活発な銘柄を取得
                if hasattr(self.symbol_selector, 'get_recommended_symbols'):
                    active_symbols = self.symbol_selector.get_recommended_symbols(limit=5)
                    dynamic_symbols.update(active_symbols)
                    logger.debug(f"推奨銘柄: {active_symbols}")

                # 高ボラティリティ銘柄を追加
                if hasattr(self.symbol_selector, 'get_high_volatility_symbols'):
                    volatile_symbols = self.symbol_selector.get_high_volatility_symbols(limit=3)
                    dynamic_symbols.update(volatile_symbols)
                    logger.debug(f"高ボラティリティ銘柄: {volatile_symbols}")

        except Exception as e:
            logger.error(f"動的監視銘柄取得エラー: {e}")

        return dynamic_symbols

    def should_update_symbols(self) -> bool:
        """監視銘柄を更新すべきかチェック
        
        最終更新時刻と設定の更新間隔を比較して、更新の必要性を判定します。
        
        Returns:
            更新が必要な場合はTrue
        """
        if self.last_update is None:
            return True

        monitoring_config = self.config_manager.get_monitoring_config()
        dynamic_config = monitoring_config.get('dynamic_monitoring', {})
        update_interval = dynamic_config.get('update_interval_hours', 24)

        elapsed = datetime.now() - self.last_update
        return elapsed.total_seconds() > (update_interval * 3600)

    def update_symbols(self) -> List[str]:
        """監視銘柄を更新
        
        現在時刻で監視対象銘柄リストを更新します。
        
        Returns:
            更新された監視対象銘柄のリスト
        """
        symbols = self.get_monitoring_symbols()
        self.current_symbols = set(symbols)
        self.last_update = datetime.now()
        logger.info(f"監視銘柄を更新しました: {len(symbols)}銘柄")
        return symbols

    def add_symbol(self, symbol: str) -> bool:
        """監視銘柄を追加
        
        Args:
            symbol: 追加する銘柄コード
            
        Returns:
            追加に成功した場合はTrue
        """
        try:
            if symbol not in self.current_symbols:
                self.current_symbols.add(symbol)
                logger.info(f"銘柄 {symbol} を監視対象に追加しました")
                return True
            else:
                logger.debug(f"銘柄 {symbol} は既に監視対象です")
                return False
        except Exception as e:
            logger.error(f"銘柄追加エラー {symbol}: {e}")
            return False

    def remove_symbol(self, symbol: str) -> bool:
        """監視銘柄を削除
        
        Args:
            symbol: 削除する銘柄コード
            
        Returns:
            削除に成功した場合はTrue
        """
        try:
            if symbol in self.current_symbols:
                self.current_symbols.remove(symbol)
                logger.info(f"銘柄 {symbol} を監視対象から削除しました")
                return True
            else:
                logger.debug(f"銘柄 {symbol} は監視対象ではありません")
                return False
        except Exception as e:
            logger.error(f"銘柄削除エラー {symbol}: {e}")
            return False

    def get_current_symbols(self) -> List[str]:
        """現在の監視銘柄を取得
        
        Returns:
            現在の監視対象銘柄のリスト
        """
        return sorted(list(self.current_symbols))

    def is_symbol_monitored(self, symbol: str) -> bool:
        """銘柄が監視対象かチェック
        
        Args:
            symbol: チェックする銘柄コード
            
        Returns:
            監視対象の場合はTrue
        """
        return symbol in self.current_symbols

    def get_symbol_categories(self) -> dict:
        """銘柄のカテゴリ別分類を取得
        
        Returns:
            カテゴリ別の銘柄辞書
        """
        symbols = self.get_current_symbols()
        categories = {
            'primary': [],
            'secondary': [],
            'dynamic': []
        }

        config_symbols = self.config_manager.config.get('monitoring_symbols', {})
        primary_symbols = set(config_symbols.get('primary_symbols', []))
        secondary_symbols = set(config_symbols.get('secondary_symbols', []))

        for symbol in symbols:
            if symbol in primary_symbols:
                categories['primary'].append(symbol)
            elif symbol in secondary_symbols:
                categories['secondary'].append(symbol)
            else:
                categories['dynamic'].append(symbol)

        return categories