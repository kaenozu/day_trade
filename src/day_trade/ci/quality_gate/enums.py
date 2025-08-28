#!/usr/bin/env python3
"""
品質ゲートシステム - Enum定義

エンタープライズレベルのコード品質管理における品質メトリクスおよび
レベル定義を提供するモジュール。
"""

from enum import Enum


class QualityMetric(Enum):
    """品質メトリクス種類
    
    コード品質評価で使用される各種メトリクスを定義。
    各メトリクスは独立した評価軸として機能し、
    総合的な品質判定に寄与する。
    """

    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    TYPING = "typing"
    SECURITY = "security"
    DEPENDENCIES = "dependencies"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


class QualityLevel(Enum):
    """品質レベル
    
    品質評価の結果として返されるレベル定義。
    各レベルは明確な基準に基づいて判定され、
    CI/CDパイプラインでの意思決定に使用される。
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    CRITICAL = "critical"