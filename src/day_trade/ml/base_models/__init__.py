#!/usr/bin/env python3
"""
Base Models for Ensemble Learning

Issue #462: アンサンブル学習システムの実装
各種ベースモデルの統一インターフェース
"""

from .random_forest_model import RandomForestModel
from .gradient_boosting_model import GradientBoostingModel
from .svr_model import SVRModel
from .base_model_interface import BaseModelInterface

# 高性能ブースティングモデル（Issue #462精度向上）
try:
    from .xgboost_model import XGBoostModel, create_xgboost_model
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBoostModel = None
    create_xgboost_model = None
    XGBOOST_AVAILABLE = False

try:
    from .catboost_model import CatBoostModel, create_catboost_model
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostModel = None
    create_catboost_model = None
    CATBOOST_AVAILABLE = False

__all__ = [
    'RandomForestModel',
    'GradientBoostingModel',
    'SVRModel',
    'BaseModelInterface'
]

# 利用可能な場合のみエクスポート
if XGBOOST_AVAILABLE:
    __all__.extend(['XGBoostModel', 'create_xgboost_model'])

if CATBOOST_AVAILABLE:
    __all__.extend(['CatBoostModel', 'create_catboost_model'])