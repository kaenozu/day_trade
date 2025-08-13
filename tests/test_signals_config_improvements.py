#!/usr/bin/env python3
"""
Issues #647, #648, #649, #655 テストケース

signals.pyの設定系改善をテスト
"""

import sys
from pathlib import Path
import tempfile
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.analysis.signals import (
    SignalRulesConfig,
    TradingSignalGenerator,
    _get_shared_config
)

def test_issue_647_config_path_robustness():
    """Issue #647: 設定ファイルパスの堅牢性改善テスト"""
    print("=== Issue #647: 設定ファイルパス堅牢性テスト ===")

    # カスタムパスでの設定テスト
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_config = {
            "default_buy_rules": [],
            "default_sell_rules": [],
            "signal_generation_settings": {"min_data_period": 30}
        }
        json.dump(test_config, f)
        temp_path = f.name

    try:
        # カスタムパスでの初期化
        config = SignalRulesConfig(temp_path)
        settings = config.get_signal_settings()

        if settings.get("min_data_period") == 30:
            print("  [PASS] カスタム設定ファイルが正しく読み込まれました")
        else:
            print("  [FAIL] カスタム設定ファイルの読み込みに失敗")

        # _resolve_config_pathメソッドのテスト
        resolved_path = config._resolve_config_path(None)
        if isinstance(resolved_path, Path):
            print(f"  [PASS] デフォルトパスが正しく解決されました: {resolved_path}")
        else:
            print("  [FAIL] デフォルトパス解決に失敗")

    finally:
        Path(temp_path).unlink()

    print()

def test_issue_648_consolidate_default_values():
    """Issue #648: デフォルト設定値の統合テスト"""
    print("=== Issue #648: デフォルト設定値統合テスト ===")

    config = SignalRulesConfig()

    # 統合されたデフォルト設定の確認
    default_config = config._create_default_config_structure()

    expected_sections = [
        'default_buy_rules',
        'default_sell_rules',
        'signal_generation_settings',
        'rsi_default_thresholds',
        'macd_default_settings',
        'pattern_breakout_settings',
        'golden_dead_cross_settings',
        'volume_spike_settings'
    ]

    all_present = True
    for section in expected_sections:
        if section in default_config:
            print(f"  [PASS] セクション '{section}' が存在します")
        else:
            print(f"  [FAIL] セクション '{section}' が見つかりません")
            all_present = False

    # 信号設定の詳細確認
    signal_settings = config._get_default_signal_settings()
    if 'confidence_multipliers' in signal_settings and 'strength_thresholds' in signal_settings:
        print("  [PASS] シグナル設定が正しく構造化されています")
    else:
        print("  [FAIL] シグナル設定の構造に問題があります")

    if all_present:
        print("  [PASS] デフォルト設定値が統合されました")
    print()

def test_issue_649_optimize_config_handling():
    """Issue #649: 設定ハンドリングの最適化テスト"""
    print("=== Issue #649: 設定ハンドリング最適化テスト ===")

    # 共有設定インスタンスのテスト
    config1 = _get_shared_config()
    config2 = _get_shared_config()

    if config1 is config2:
        print("  [PASS] 共有設定インスタンスが正しく機能しています")
    else:
        print("  [FAIL] 共有設定インスタンスが機能していません")

    # 設定の再利用確認
    print(f"  共有設定インスタンスID: {id(config1)}")
    print(f"  再取得設定インスタンスID: {id(config2)}")

    # 設定値の一貫性確認
    rsi_thresholds1 = config1.get_rsi_thresholds()
    rsi_thresholds2 = config2.get_rsi_thresholds()

    if rsi_thresholds1 == rsi_thresholds2:
        print("  [PASS] 設定値の一貫性が保たれています")
    else:
        print("  [FAIL] 設定値の一貫性に問題があります")

    print()

def test_issue_655_externalize_rule_map():
    """Issue #655: rule_mapの外部化テスト"""
    print("=== Issue #655: rule_map外部化テスト ===")

    generator = TradingSignalGenerator()

    # rule_mappingメソッドのテスト
    rule_mapping = generator._get_rule_mapping()

    expected_rules = [
        'RSIOversoldRule',
        'RSIOverboughtRule',
        'MACDCrossoverRule',
        'MACDDeathCrossRule',
        'BollingerBandRule',
        'PatternBreakoutRule',
        'GoldenCrossRule',
        'DeadCrossRule',
        'VolumeSpikeBuyRule'
    ]

    all_present = True
    for rule_name in expected_rules:
        if rule_name in rule_mapping:
            print(f"  [PASS] ルール '{rule_name}' がマッピングに存在します")
        else:
            print(f"  [FAIL] ルール '{rule_name}' がマッピングに見つかりません")
            all_present = False

    # マッピングの型確認
    if all(isinstance(rule_class, type) for rule_class in rule_mapping.values()):
        print("  [PASS] ルールマッピングの型が正しく設定されています")
    else:
        print("  [FAIL] ルールマッピングの型に問題があります")

    if all_present:
        print("  [PASS] rule_mapが外部化されました")
    print()

def test_integration():
    """統合テスト"""
    print("=== 統合テスト ===")

    try:
        # TradingSignalGeneratorの初期化テスト
        generator = TradingSignalGenerator()

        print(f"  買いルール数: {len(generator.buy_rules)}")
        print(f"  売りルール数: {len(generator.sell_rules)}")

        # 設定アクセステスト
        config = generator.config
        rsi_thresholds = config.get_rsi_thresholds()
        macd_settings = config.get_macd_settings()

        print(f"  RSI閾値: {rsi_thresholds}")
        print(f"  MACD設定: {macd_settings}")

        print("  [PASS] 統合テストが成功しました")

    except Exception as e:
        print(f"  [FAIL] 統合テストでエラーが発生: {e}")

    print()

def run_all_tests():
    """全テストを実行"""
    print("signals.py 設定系改善テスト開始\\n")

    test_issue_647_config_path_robustness()
    test_issue_648_consolidate_default_values()
    test_issue_649_optimize_config_handling()
    test_issue_655_externalize_rule_map()
    test_integration()

    print("全テスト完了")

if __name__ == "__main__":
    run_all_tests()