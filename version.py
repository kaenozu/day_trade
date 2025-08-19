#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade Personal - Version Management
統一バージョン管理システム
"""

# メインバージョン情報
__version__ = "2.1.0"
__version_info__ = (2, 1, 0)
__release_name__ = "Extended"
__build_date__ = "2025-08-18"

# 統合バージョン文字列
__version_extended__ = f"{__version__}_{__release_name__.lower()}"
__version_full__ = f"Day Trade Personal v{__version__} {__release_name__} (Build {__build_date__})"

# API用バージョン情報
def get_version_info():
    """API用のバージョン情報を返す"""
    return {
        "version": __version__,
        "version_extended": __version_extended__,
        "release_name": __release_name__,
        "build_date": __build_date__,
        "version_info": __version_info__
    }

# コンポーネント別バージョン
COMPONENTS = {
    "core": __version__,
    "web_server": __version_extended__,
    "cli": __version_extended__,
    "api": __version__,
}

# 互換性情報
COMPATIBILITY = {
    "min_python_version": "3.8",
    "recommended_python_version": "3.11",
    "supported_os": ["Windows", "macOS", "Linux"],
}

if __name__ == "__main__":
    print(__version_full__)
    print(f"Components: {COMPONENTS}")
    print(f"Compatibility: {COMPATIBILITY}")