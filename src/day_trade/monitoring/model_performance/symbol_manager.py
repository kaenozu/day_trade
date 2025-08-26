#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Performance Monitor - Symbol Manager
動的監視対象銘柄管理クラス
"""

import logging
from datetime import datetime
from typing import List, Set
from .config_manager import EnhancedPerformanceConfigManager

logger = logging.getLogger(__name__)

# symbol_selector連携
try:
    from symbol_selector import SymbolSelector
    SYMBOL_SELECTOR_AVAILABLE = True
except ImportError:
    SYMBOL_SELECTOR_AVAILABLE = False


class DynamicSymbolManager:
    """動的監視対象銘柄管理クラス"""

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
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
        """現在の監視対象銘柄を取得"""
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
        default_symbols = monitoring_config.get(
            'default_symbols', ["7203", "8306", "4751"]
        )
        symbols.extend(default_symbols)

        # 重複除去してソート
        return sorted(list(set(symbols)))

    def _get_dynamic_symbols(self) -> Set[str]:
        """動的監視銘柄を取得"""
        dynamic_symbols = set()

        try:
            if self.symbol_selector:
                # symbol_selectorから活発な銘柄を取得
                if hasattr(self.symbol_selector, 'get_recommended_symbols'):
                    active_symbols = self.symbol_selector.get_recommended_symbols(
                        limit=5
                    )
                    dynamic_symbols.update(active_symbols)

                # 高ボラティリティ銘柄を追加
                if hasattr(self.symbol_selector, 'get_high_volatility_symbols'):
                    volatile_symbols = (
                        self.symbol_selector.get_high_volatility_symbols(
                            limit=3
                        )
                    )
                    dynamic_symbols.update(volatile_symbols)

        except Exception as e:
            logger.error(f"動的監視銘柄取得エラー: {e}")

        return dynamic_symbols

    def should_update_symbols(self) -> bool:
        """監視銘柄を更新すべきかチェック"""
        if self.last_update is None:
            return True

        monitoring_config = self.config_manager.get_monitoring_config()
        dynamic_config = monitoring_config.get('dynamic_monitoring', {})
        update_interval = dynamic_config.get('update_interval_hours', 24)

        elapsed = datetime.now() - self.last_update
        return elapsed.total_seconds() > (update_interval * 3600)