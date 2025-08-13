#!/usr/bin/env python3
"""
銘柄名表示ヘルパー

Issue #450: 銘柄コードから日本語会社名を取得・表示する機能
"""

import json
import os
import threading
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

        # Issue #606対応: 設定パスの堅牢性改善
        self.config_path = self._resolve_config_path(config_path)
        self._load_stock_info()
    
    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """
        設定ファイルパスの解決 - Issue #606対応
        
        Args:
            config_path: 指定された設定パス
            
        Returns:
            解決された設定ファイルパス
        """
        if config_path is not None:
            # 明示的に指定されたパスを使用
            return Path(config_path)
        
        # デフォルトパスの探索と選択
        possible_paths = [
            # 1. プロジェクトルートからの相対パス
            Path(__file__).parent.parent.parent.parent / "config" / "settings.json",
            # 2. 現在のワーキングディレクトリからの相対パス
            Path.cwd() / "config" / "settings.json",
            # 3. 環境変数からのパス
            Path(os.getenv("STOCK_CONFIG_PATH", "")) if os.getenv("STOCK_CONFIG_PATH") else None
        ]
        
        # Noneを除外して最初に存在するファイルを使用
        for path in possible_paths:
            if path is not None and path.exists():
                logger.debug(f"設定ファイルを検出: {path}")
                return path
        
        # どれも存在しない場合はデフォルトを返す
        default_path = Path(__file__).parent.parent.parent.parent / "config" / "settings.json"
        logger.debug(f"デフォルト設定パスを使用: {default_path}")
        return default_path

    def _load_stock_info(self):
        """
        Issue #607対応: 株式情報読み込みロジック改善
        エラーハンドリング強化とデータ検証
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                # Issue #607対応: 空のキャッシュで継続
                self._stock_info_cache = {}
                self._config_loaded = True  # 空でも読み込み完了とみなす
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Issue #607対応: 複数の設定形式に対応
            symbols = []
            if 'watchlist' in config and 'symbols' in config['watchlist']:
                symbols = config['watchlist']['symbols']
            elif 'symbols' in config:
                symbols = config['symbols']
            elif 'stock_info' in config:
                symbols = config['stock_info']

            if not symbols:
                logger.warning("設定ファイルに銘柄情報がありません")
                self._stock_info_cache = {}
                self._config_loaded = True
                return

            # データ検証付きキャッシュ構築
            loaded_count = 0
            for symbol_info in symbols:
                if isinstance(symbol_info, dict):
                    code = symbol_info.get('code', '')
                    name = symbol_info.get('name', '')
                    
                    # Issue #607対応: 必須フィールド検証
                    if code and name:
                        self._stock_info_cache[code] = {
                            'name': name,
                            'group': symbol_info.get('group', 'その他'),
                            'sector': symbol_info.get('sector', '未分類'),
                            'priority': symbol_info.get('priority', 'medium')
                        }
                        loaded_count += 1
                    else:
                        logger.debug(f"不完全な銘柄データをスキップ: {symbol_info}")
                elif isinstance(symbol_info, str):
                    # Issue #607対応: 文字列のみの場合の対応
                    self._stock_info_cache[symbol_info] = {
                        'name': symbol_info,
                        'group': 'その他',
                        'sector': '未分類',
                        'priority': 'medium'
                    }
                    loaded_count += 1

            self._config_loaded = True
            logger.info(f"銘柄情報読み込み完了: {loaded_count} 銘柄")

        except json.JSONDecodeError as e:
            logger.error(f"JSONパースエラー: {e}")
            self._stock_info_cache = {}
            self._config_loaded = True  # エラーでも継続
        except Exception as e:
            logger.error(f"銘柄情報読み込みエラー: {e}")
            self._stock_info_cache = {}
            self._config_loaded = True

    def get_stock_name(
        self, 
        symbol: Union[str, int], 
        fallback_format: str = "{symbol}"
    ) -> str:
        """
        Issue #608対応: 銘柄コードから日本語会社名を取得（フォールバック強化）

        Args:
            symbol: 銘柄コード（文字列または数値）
            fallback_format: 未知銘柄時のフォーマット文字列

        Returns:
            日本語会社名または適切なフォールバック文字列
        """
        symbol_str = str(symbol).strip()

        if not self._config_loaded:
            logger.debug(f"設定未読み込み状態で銘柄名要求: {symbol_str}")
            return fallback_format.format(symbol=symbol_str)

        stock_info = self._stock_info_cache.get(symbol_str)
        if stock_info and stock_info.get('name'):
            return stock_info['name']
        
        # Issue #608対応: 未知銘柄への拡張対応
        # 1. 4桁の数値コードの場合の推測
        if symbol_str.isdigit() and len(symbol_str) == 4:
            sector_map = {
                "1": "水産・農林業", "2": "鉱業", "3": "建設業", "4": "食料品",
                "5": "繊維製品", "6": "パルプ・紙", "7": "化学", "8": "医薬品", "9": "石油・石炭製品"
            }
            first_digit = symbol_str[0]
            sector_hint = sector_map.get(first_digit, "その他業種")
            return fallback_format.format(symbol=f"{symbol_str}({sector_hint})")
        
        # 2. アルファベット含みの場合（外国株等）
        if any(c.isalpha() for c in symbol_str):
            return fallback_format.format(symbol=f"{symbol_str}(外国株等)")
        
        # 3. デフォルトフォールバック
        return fallback_format.format(symbol=symbol_str)

    def get_stock_info(self, symbol: Union[str, int]) -> Dict[str, str]:
        """
        銘柄コードから詳細情報を取得

        Args:
            symbol: 銘柄コード

        Returns:
            銘柄情報の辞書
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
        default_info.update(stock_info)

        return default_info

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
        stock_name = self.get_stock_name(symbol_str)

        if include_code:
            if stock_name != symbol_str:
                return f"{symbol_str}({stock_name})"
            else:
                return symbol_str
        else:
            return stock_name

    def get_all_symbols(self) -> Dict[str, Dict[str, str]]:
        """
        全銘柄情報を取得

        Returns:
            全銘柄の辞書（キー: 銘柄コード、値: 銘柄情報）
        """
        if not self._config_loaded:
            return {}

        return self._stock_info_cache.copy()
    
    # Issue #611対応: シングルトンパターン再検討と改善
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'StockNameHelper':
        """
        Issue #611対応: スレッドセーフなシングルトンインスタンス取得
        
        Args:
            config_path: 設定ファイルパス（初回のみ有効）
            
        Returns:
            StockNameHelperのシングルトンインスタンス
        """
        if cls._instance is None:
            with cls._lock:
                # ダブルチェックロッキングパターン
                if cls._instance is None:
                    cls._instance = cls(config_path)
                    logger.debug("StockNameHelper シングルトンインスタンスを作成しました")
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """
        Issue #611対応: シングルトンインスタンスのリセット（テスト用）
        """
        with cls._lock:
            cls._instance = None
            logger.debug("StockNameHelper シングルトンインスタンスをリセットしました")


# Issue #612対応: ユーティリティ関数の再配置と最適化
def format_symbol_display(symbol: Union[str, int], include_sector: bool = False) -> str:
    """
    Issue #612対応: 銘柄表示用のグローバルユーティリティ関数
    
    Args:
        symbol: 銘柄コード
        include_sector: セクター情報を含めるか
        
    Returns:
        フォーマットされた銘柄表示文字列
    """
    helper = StockNameHelper.get_instance()
    
    if include_sector:
        stock_info = helper.get_stock_info(symbol)
        sector = stock_info.get('sector', '')
        name = helper.get_stock_name(symbol)
        if sector and sector != '未分類':
            return f"{symbol}({name})[{sector}]"
        else:
            return f"{symbol}({name})"
    else:
        return helper.format_stock_display(symbol)


def get_stock_name_quick(symbol: Union[str, int]) -> str:
    """
    Issue #612対応: 銘柄名取得のクイックアクセス関数
    
    Args:
        symbol: 銘柄コード
        
    Returns:
        銘柄名
    """
    helper = StockNameHelper.get_instance()
    return helper.get_stock_name(symbol)


def validate_symbol_format(symbol: Union[str, int]) -> bool:
    """
    Issue #612対応: 銘柄コード形式検証ユーティリティ
    
    Args:
        symbol: 銘柄コード
        
    Returns:
        有効な形式かどうか
    """
    symbol_str = str(symbol).strip()
    
    # 4桁の数値（日本株）
    if symbol_str.isdigit() and len(symbol_str) == 4:
        return True
    
    # アルファベット含み（外国株等）
    if any(c.isalpha() for c in symbol_str) and len(symbol_str) <= 10:
        return True
    
    return False