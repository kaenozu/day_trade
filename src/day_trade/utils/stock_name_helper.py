#!/usr/bin/env python3
"""
銘柄名表示ヘルパー

Issue #450: 銘柄コードから日本語会社名を取得・表示する機能
"""

import json
import os
from typing import Dict, Optional, Union
from pathlib import Path

from .logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockNameHelper:
    """銘柄名表示ヘルパークラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス（デフォルトは config/settings.json）
        """
        self._stock_info_cache = {}
        self._config_loaded = False
        
        if config_path is None:
            # プロジェクトルートからの相対パス
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "settings.json"
        
        self.config_path = Path(config_path)
        self._load_stock_info()
    
    def _load_stock_info(self):
        """設定ファイルから銘柄情報を読み込み"""
        try:
            if not self.config_path.exists():
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # watchlist.symbols から銘柄情報を抽出
            watchlist = config.get('watchlist', {})
            symbols = watchlist.get('symbols', [])
            
            for symbol_info in symbols:
                if isinstance(symbol_info, dict):
                    code = symbol_info.get('code', '')
                    if code:
                        self._stock_info_cache[code] = {
                            'name': symbol_info.get('name', ''),
                            'group': symbol_info.get('group', ''),
                            'sector': symbol_info.get('sector', ''),
                            'priority': symbol_info.get('priority', 'medium')
                        }
            
            self._config_loaded = True
            logger.info(f"銘柄情報読み込み完了: {len(self._stock_info_cache)} 銘柄")
            
        except Exception as e:
            logger.error(f"銘柄情報読み込みエラー: {e}")
            self._config_loaded = False
    
    def get_stock_name(self, symbol: Union[str, int]) -> str:
        """
        銘柄コードから日本語会社名を取得
        
        Args:
            symbol: 銘柄コード（文字列または数値）
        
        Returns:
            日本語会社名（見つからない場合は銘柄コードをそのまま返す）
        """
        symbol_str = str(symbol).strip()
        
        if not self._config_loaded:
            return symbol_str
        
        stock_info = self._stock_info_cache.get(symbol_str)
        if stock_info and stock_info.get('name'):
            return stock_info['name']
        
        return symbol_str
    
    def get_stock_info(self, symbol: Union[str, int]) -> Dict[str, str]:
        """
        銘柄コードから詳細情報を取得
        
        Args:
            symbol: 銘柄コード（文字列または数値）
        
        Returns:
            銘柄詳細情報（辞書形式）
        """
        symbol_str = str(symbol).strip()
        
        default_info = {
            'code': symbol_str,
            'name': symbol_str,
            'group': '不明',
            'sector': '不明',
            'priority': 'medium'
        }
        
        if not self._config_loaded:
            return default_info
        
        stock_info = self._stock_info_cache.get(symbol_str, {})
        result = default_info.copy()
        result.update(stock_info)
        result['code'] = symbol_str  # コードは常に設定
        
        return result
    
    def format_stock_display(self, symbol: Union[str, int], include_code: bool = True) -> str:
        """
        銘柄表示用の文字列をフォーマット
        
        Args:
            symbol: 銘柄コード
            include_code: コードを含めるかどうか
        
        Returns:
            フォーマットされた表示文字列
        """
        symbol_str = str(symbol).strip()
        stock_info = self.get_stock_info(symbol_str)
        
        if include_code:
            if stock_info['name'] != symbol_str:
                return f"{symbol_str}({stock_info['name']})"
            else:
                return symbol_str
        else:
            return stock_info['name']
    
    def get_all_symbols(self) -> Dict[str, Dict[str, str]]:
        """
        全銘柄情報を取得
        
        Returns:
            全銘柄の辞書（キー: 銘柄コード、値: 銘柄情報）
        """
        if not self._config_loaded:
            return {}
        
        return self._stock_info_cache.copy()
    
    def search_by_name(self, name_part: str) -> Dict[str, Dict[str, str]]:
        """
        会社名の一部で検索
        
        Args:
            name_part: 検索する会社名の一部
        
        Returns:
            マッチした銘柄の辞書
        """
        if not self._config_loaded:
            return {}
        
        results = {}
        name_part_lower = name_part.lower()
        
        for code, info in self._stock_info_cache.items():
            name = info.get('name', '').lower()
            if name_part_lower in name:
                results[code] = info.copy()
                results[code]['code'] = code
        
        return results
    
    def reload_config(self):
        """設定ファイルを再読み込み"""
        self._stock_info_cache.clear()
        self._config_loaded = False
        self._load_stock_info()


# グローバルインスタンス（シングルトンパターン）
_global_stock_helper = None


def get_stock_helper() -> StockNameHelper:
    """グローバル銘柄名ヘルパーを取得"""
    global _global_stock_helper
    if _global_stock_helper is None:
        _global_stock_helper = StockNameHelper()
    return _global_stock_helper


def get_stock_name(symbol: Union[str, int]) -> str:
    """
    銘柄コードから日本語会社名を取得（便利関数）
    
    Args:
        symbol: 銘柄コード
    
    Returns:
        日本語会社名
    """
    return get_stock_helper().get_stock_name(symbol)


def format_stock_display(symbol: Union[str, int], include_code: bool = True) -> str:
    """
    銘柄表示用文字列をフォーマット（便利関数）
    
    Args:
        symbol: 銘柄コード
        include_code: コードを含めるかどうか
    
    Returns:
        フォーマットされた表示文字列
    """
    return get_stock_helper().format_stock_display(symbol, include_code)


if __name__ == "__main__":
    # テスト実行
    print("=== 銘柄名ヘルパーテスト ===")
    
    helper = StockNameHelper()
    
    # テスト銘柄
    test_symbols = ["7203", "8306", "9984", "1234"]  # 1234は存在しない銘柄
    
    for symbol in test_symbols:
        name = helper.get_stock_name(symbol)
        display = helper.format_stock_display(symbol)
        info = helper.get_stock_info(symbol)
        
        print(f"銘柄コード: {symbol}")
        print(f"  会社名: {name}")
        print(f"  表示形式: {display}")
        print(f"  詳細情報: {info}")
        print()
    
    # 検索テスト
    print("=== 検索テスト ===")
    search_results = helper.search_by_name("トヨタ")
    print(f"'トヨタ'で検索: {search_results}")
    
    search_results = helper.search_by_name("銀行")
    print(f"'銀行'で検索: {search_results}")
    
    print(f"\n総銘柄数: {len(helper.get_all_symbols())}")