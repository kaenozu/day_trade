#!/usr/bin/env python3
"""
Issue #583: ML Score Calculation Robustness Test
MLスコア計算の堅牢性改善テスト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def test_safe_score_extraction():
    """安全なスコア値抽出のテスト"""
    print("=== Issue #583 ML Score Robustness Test ===")

    try:
        from daytrade import _safe_get_score_value

        print("1. Testing _safe_get_score_value function...")

        # Mock score object for testing
        class MockScore:
            def __init__(self, value):
                self.score_value = value

        # Normal case
        normal_score = MockScore(75.5)
        result = _safe_get_score_value(normal_score, "test")
        print(f"  Normal score (75.5): {result}")
        assert result == 75.5

        # None case
        result = _safe_get_score_value(None, "test")
        print(f"  None score: {result}")
        assert result is None

        # Invalid value case
        invalid_score = MockScore(float('inf'))
        result = _safe_get_score_value(invalid_score, "test")
        print(f"  Invalid score (inf): {result}")
        assert result is None

        # Out of range case (negative)
        negative_score = MockScore(-10.5)
        result = _safe_get_score_value(negative_score, "test")
        print(f"  Negative score (-10.5): {result}")
        assert result == 0.0

        # Out of range case (too high)
        high_score = MockScore(150.0)
        result = _safe_get_score_value(high_score, "test")
        print(f"  High score (150.0): {result}")
        assert result == 100.0

        print("  ✓ _safe_get_score_value tests passed")

    except Exception as e:
        print(f"  ✗ _safe_get_score_value test error: {e}")
        return False

    return True

def test_educational_analysis_robustness():
    """教育的分析システムの堅牢性テスト"""
    print("\n2. Testing educational analysis robustness...")

    try:
        from src.day_trade.analysis.educational_analysis import EducationalMarketAnalyzer

        # 初期化テスト
        analyzer = EducationalMarketAnalyzer()

        # スコア検証メソッドのテスト
        print("  Testing score validation methods...")

        # 正常値
        normal_score = analyzer._validate_and_normalize_score(75.3, "test")
        print(f"    Normal score (75.3): {normal_score}")
        assert normal_score == 75.3

        # None値
        none_score = analyzer._validate_and_normalize_score(None, "test")
        print(f"    None score: {none_score}")
        assert none_score == 50.0

        # 文字列値
        str_score = analyzer._validate_and_normalize_score("invalid", "test")
        print(f"    String score: {str_score}")
        assert str_score == 50.0

        # 信頼度検証
        conf_normal = analyzer._validate_confidence(0.75)
        print(f"    Normal confidence (0.75): {conf_normal}")
        assert conf_normal == 0.75

        conf_none = analyzer._validate_confidence(None)
        print(f"    None confidence: {conf_none}")
        assert conf_none == 0.5

        print("  ✓ Educational analysis robustness tests passed")

    except Exception as e:
        print(f"  ✗ Educational analysis test error: {e}")
        return False

    return True

def test_ml_score_generation_safety():
    """MLスコア生成の安全性テスト"""
    print("\n3. Testing ML score generation safety...")

    try:
        from src.day_trade.analysis.educational_analysis import EducationalMarketAnalyzer

        analyzer = EducationalMarketAnalyzer()

        # フォールバックMLスコア生成テスト
        fallback_scores = analyzer._generate_ml_technical_scores_fallback("TEST")

        print(f"  Fallback scores generated: {len(fallback_scores)}")

        # 各スコアの検証
        for score in fallback_scores:
            print(f"    {score.score_name}: {score.score_value} (conf: {score.confidence_level})")

            # スコア値の範囲チェック
            assert 0 <= score.score_value <= 100, f"Score out of range: {score.score_value}"
            # 信頼度の範囲チェック
            assert 0 <= score.confidence_level <= 1, f"Confidence out of range: {score.confidence_level}"

        print("  ✓ ML score generation safety tests passed")

    except Exception as e:
        print(f"  ✗ ML score generation test error: {e}")
        return False

    return True

def main():
    """メインテスト実行"""
    print("Issue #583 ML Score Calculation Robustness Test\n")

    all_tests = [
        test_safe_score_extraction,
        test_educational_analysis_robustness,
        test_ml_score_generation_safety
    ]

    passed = 0
    failed = 0

    for test_func in all_tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            failed += 1

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("✅ Issue #583 ML Score Robustness: ALL TESTS PASSED")
        return True
    else:
        print("❌ Issue #583 ML Score Robustness: SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)