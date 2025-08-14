#!/usr/bin/env python3
"""
包括的テスト自動化フレームワーク
Comprehensive Test Automation Framework

Issue #760: 包括的テスト自動化と検証フレームワークの構築
"""

from .framework import (
    TestFramework,
    TestConfig,
    TestResult,
    TestSuite,
    TestRunner
)

from .fixtures import (
    TestDataManager,
    MockDataGenerator,
    FixtureRegistry,
    CommonFixtures
)

from .assertions import (
    PerformanceAssertions,
    MLModelAssertions,
    DataQualityAssertions,
    IntegrationAssertions
)

from .reporters import (
    TestReporter,
    PerformanceReporter,
    CoverageReporter,
    HTMLReporter
)

__version__ = "1.0.0"
__author__ = "Day Trade ML System"

# モジュール公開API
__all__ = [
    # Core Framework
    "TestFramework",
    "TestConfig",
    "TestResult",
    "TestSuite",
    "TestRunner",

    # Fixtures and Data
    "TestDataManager",
    "MockDataGenerator",
    "FixtureRegistry",
    "CommonFixtures",

    # Assertions
    "PerformanceAssertions",
    "MLModelAssertions",
    "DataQualityAssertions",
    "IntegrationAssertions",

    # Reporting
    "TestReporter",
    "PerformanceReporter",
    "CoverageReporter",
    "HTMLReporter"
]


def get_framework_info() -> dict:
    """
    テストフレームワーク情報取得

    Returns:
        フレームワーク情報辞書
    """
    return {
        "module": "day_trade.testing",
        "version": __version__,
        "components": [
            "TestFramework",
            "TestDataManager",
            "PerformanceAssertions",
            "TestReporter"
        ],
        "features": [
            "統合テスト自動化",
            "パフォーマンステスト",
            "MLモデル検証",
            "データ品質検証",
            "カバレッジ分析",
            "CI/CD統合"
        ],
        "supported_frameworks": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-benchmark"
        ],
        "targets": {
            "test_execution_time": "<30分",
            "code_coverage": ">90%",
            "test_success_rate": ">99%",
            "flake_test_rate": "<1%"
        }
    }