#!/usr/bin/env python3
"""
遅延インポートモジュール

パフォーマンス最適化のための遅延インポート実装
"""

import importlib
from typing import Any, Dict, Optional


class LazyImport:
    """遅延インポートクラス"""

    def __init__(self, module_name: str, attr_name: Optional[str] = None):
        self.module_name = module_name
        self.attr_name = attr_name
        self._module = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = importlib.import_module(self.module_name)

        if self.attr_name:
            attr = getattr(self._module, self.attr_name)
            return getattr(attr, name)
        else:
            return getattr(self._module, name)


class OptimizedImports:
    """最適化されたインポート管理"""

    # 重いライブラリの遅延インポート
    numpy = LazyImport('numpy')
    pandas = LazyImport('pandas')
    sklearn = LazyImport('sklearn')
    tensorflow = LazyImport('tensorflow')
    torch = LazyImport('torch')

    # よく使用される軽量インポート
    @staticmethod
    def get_datetime():
        from datetime import datetime
        return datetime

    @staticmethod
    def get_json():
        import json
        return json

    @staticmethod
    def get_pathlib():
        from pathlib import Path
        return Path


# グローバル最適化インスタンス
optimized_imports = OptimizedImports()

# 使用例
# np = optimized_imports.numpy  # 使用時に初めてnumpyがインポートされる
# pd = optimized_imports.pandas  # 使用時に初めてpandasがインポートされる


class ImportManager:
    """インポート管理クラス"""
    
    def __init__(self):
        self._imported_modules: Dict[str, Any] = {}
        self._lazy_imports: Dict[str, LazyImport] = {}
    
    def register_lazy_import(self, name: str, module_name: str, 
                           attr_name: Optional[str] = None):
        """遅延インポートを登録"""
        self._lazy_imports[name] = LazyImport(module_name, attr_name)
    
    def get_module(self, name: str) -> Any:
        """モジュール取得（遅延ロード）"""
        if name in self._imported_modules:
            return self._imported_modules[name]
        
        if name in self._lazy_imports:
            module = self._lazy_imports[name]
            self._imported_modules[name] = module
            return module
        
        raise ImportError(f"Module '{name}' not registered")
    
    def preload_critical_modules(self):
        """重要なモジュールの事前ロード"""
        critical_modules = ['datetime', 'json', 'pathlib']
        
        for module_name in critical_modules:
            try:
                if module_name == 'datetime':
                    self._imported_modules[module_name] = self.get_datetime()
                elif module_name == 'json':
                    self._imported_modules[module_name] = self.get_json()
                elif module_name == 'pathlib':
                    self._imported_modules[module_name] = self.get_pathlib()
            except ImportError:
                pass  # スキップ


# グローバルインポートマネージャー
import_manager = ImportManager()

# 基本モジュールを事前登録
import_manager.register_lazy_import('numpy', 'numpy')
import_manager.register_lazy_import('pandas', 'pandas')
import_manager.register_lazy_import('sklearn', 'sklearn')
import_manager.register_lazy_import('matplotlib', 'matplotlib.pyplot')
import_manager.register_lazy_import('seaborn', 'seaborn')