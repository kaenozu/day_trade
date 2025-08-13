#!/usr/bin/env python3
"""
Issue #643: OptimizationConfig テンプレート生成改善テスト
JSON標準準拠のテンプレート生成機能とスキーマ検証のテスト
"""

import sys
import json
import tempfile
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def test_standard_json_template():
    """標準JSONテンプレート生成テスト"""
    print("=== Issue #643 Standard JSON Template Test ===\n")

    try:
        from src.day_trade.core.optimization_strategy import OptimizationStrategyFactory

        print("標準JSONテンプレート生成テスト:")
        print("-" * 60)

        results = []

        # テスト用一時ファイル
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 標準JSONテンプレートの生成
            print("1. 標準JSONテンプレート生成テスト:")
            OptimizationStrategyFactory.create_config_template(temp_path, include_comments=False)

            # ファイルが作成されているか確認
            template_created = os.path.exists(temp_path)
            results.append(template_created)

            status = "OK PASS" if template_created else "NG FAIL"
            print(f"   テンプレートファイル作成: {status}")

            if template_created:
                # JSONとして正しく読み込めるか確認
                print("\n2. JSON形式妥当性テスト:")
                try:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        template_data = json.load(f)

                    json_valid = isinstance(template_data, dict)
                    results.append(json_valid)

                    status = "OK PASS" if json_valid else "NG FAIL"
                    print(f"   JSON形式妥当性: {status}")

                    if json_valid:
                        print(f"   読み込み成功: {len(template_data)}個のフィールド")
                        print(f"   フィールド一覧: {list(template_data.keys())}")

                        # _commentsフィールドがないことを確認
                        print("\n3. _commentsフィールド除外テスト:")
                        no_comments = "_comments" not in template_data
                        results.append(no_comments)

                        status = "OK PASS" if no_comments else "NG FAIL"
                        print(f"   _commentsフィールド除外: {status}")

                        # 必要なフィールドが含まれているかチェック
                        print("\n4. 必須フィールド存在テスト:")
                        required_fields = ["level", "auto_fallback", "performance_monitoring", "cache_enabled"]
                        fields_present = all(field in template_data for field in required_fields)
                        results.append(fields_present)

                        status = "OK PASS" if fields_present else "NG FAIL"
                        print(f"   必須フィールド存在: {status}")
                        print(f"   存在フィールド: {[f for f in required_fields if f in template_data]}")

                except json.JSONDecodeError as e:
                    print(f"   JSON読み込みエラー: {e}")
                    results.append(False)

        finally:
            # 一時ファイルクリーンアップ
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        all_passed = all(results)
        print(f"\n標準JSONテンプレート生成: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"標準JSONテンプレート生成テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_commented_template():
    """コメント付きテンプレート生成テスト"""
    print("\n=== Issue #643 Commented Template Test ===\n")

    try:
        from src.day_trade.core.optimization_strategy import OptimizationStrategyFactory

        print("コメント付きテンプレート生成テスト:")
        print("-" * 60)

        results = []

        # テスト用一時ファイル
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonc', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # コメント付きテンプレートの生成
            print("1. コメント付きテンプレート生成テスト:")
            OptimizationStrategyFactory.create_config_template(temp_path, include_comments=True)

            # ファイルが作成されているか確認
            template_created = os.path.exists(temp_path)
            results.append(template_created)

            status = "OK PASS" if template_created else "NG FAIL"
            print(f"   コメント付きテンプレート作成: {status}")

            if template_created:
                # ファイル内容の確認
                print("\n2. コメント形式テスト:")
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # コメント（//）が含まれているかチェック
                has_comments = "//" in content
                results.append(has_comments)

                status = "OK PASS" if has_comments else "NG FAIL"
                print(f"   コメント存在: {status}")

                # JSON形式の基本構造があるかチェック
                print("\n3. JSON基本構造テスト:")
                has_json_structure = content.strip().startswith('{') and content.strip().endswith('}')
                results.append(has_json_structure)

                status = "OK PASS" if has_json_structure else "NG FAIL"
                print(f"   JSON基本構造: {status}")

                # 特定のコメントが含まれているかチェック
                print("\n4. 説明コメント内容テスト:")
                expected_comments = ["最適化レベル", "フォールバック", "パフォーマンス監視"]
                comments_found = all(comment in content for comment in expected_comments)
                results.append(comments_found)

                status = "OK PASS" if comments_found else "NG FAIL"
                print(f"   説明コメント内容: {status}")

                print(f"\n   生成されたテンプレート（最初の10行）:")
                lines = content.split('\n')[:10]
                for i, line in enumerate(lines, 1):
                    print(f"   {i:2d}: {line}")

        finally:
            # 一時ファイルクリーンアップ
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        all_passed = all(results)
        print(f"\nコメント付きテンプレート生成: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"コメント付きテンプレート生成テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_schema():
    """テンプレートスキーマ機能テスト"""
    print("\n=== Issue #643 Template Schema Test ===\n")

    try:
        from src.day_trade.core.optimization_strategy import OptimizationStrategyFactory

        print("テンプレートスキーマ機能テスト:")
        print("-" * 60)

        results = []

        # スキーマ取得テスト
        print("1. スキーマ取得テスト:")
        schema = OptimizationStrategyFactory.get_template_schema()

        schema_exists = schema is not None and isinstance(schema, dict)
        results.append(schema_exists)

        status = "OK PASS" if schema_exists else "NG FAIL"
        print(f"   スキーマ取得: {status}")

        if schema_exists:
            # スキーマ構造テスト
            print("\n2. スキーマ構造テスト:")
            required_schema_keys = ["type", "properties"]
            schema_structure_valid = all(key in schema for key in required_schema_keys)
            results.append(schema_structure_valid)

            status = "OK PASS" if schema_structure_valid else "NG FAIL"
            print(f"   スキーマ基本構造: {status}")

            # プロパティ定義テスト
            print("\n3. プロパティ定義テスト:")
            if "properties" in schema:
                properties = schema["properties"]
                expected_properties = ["level", "auto_fallback", "performance_monitoring", "cache_enabled"]
                properties_defined = all(prop in properties for prop in expected_properties)
                results.append(properties_defined)

                status = "OK PASS" if properties_defined else "NG FAIL"
                print(f"   必須プロパティ定義: {status}")
                print(f"   定義プロパティ数: {len(properties)}")

                # 型定義テスト
                print("\n4. 型定義テスト:")
                type_definitions_valid = True
                for prop_name, prop_def in properties.items():
                    if not isinstance(prop_def, dict) or "type" not in prop_def:
                        type_definitions_valid = False
                        break

                results.append(type_definitions_valid)

                status = "OK PASS" if type_definitions_valid else "NG FAIL"
                print(f"   プロパティ型定義: {status}")
            else:
                results.extend([False, False])

        all_passed = all(results)
        print(f"\nテンプレートスキーマ機能: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"テンプレートスキーマ機能テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """後方互換性テスト"""
    print("\n=== Issue #643 Backward Compatibility Test ===\n")

    try:
        from src.day_trade.core.optimization_strategy import OptimizationStrategyFactory

        print("後方互換性テスト:")
        print("-" * 60)

        results = []

        # デフォルト引数でのテンプレート生成テスト
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # 引数省略でのテンプレート生成（デフォルトでコメント付き）
            print("1. デフォルト引数テスト:")
            OptimizationStrategyFactory.create_config_template(temp_path)

            # ファイルが作成されているか確認
            template_created = os.path.exists(temp_path)
            results.append(template_created)

            status = "OK PASS" if template_created else "NG FAIL"
            print(f"   デフォルト引数でのテンプレート作成: {status}")

            if template_created:
                # デフォルトでコメント付きになっているかチェック
                print("\n2. デフォルト動作テスト:")
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                default_has_comments = "//" in content
                results.append(default_has_comments)

                status = "OK PASS" if default_has_comments else "NG FAIL"
                print(f"   デフォルトでコメント付き: {status}")

        finally:
            # 一時ファイルクリーンアップ
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        all_passed = all(results)
        print(f"\n後方互換性: {'OK ALL TESTS PASSED' if all_passed else 'NG SOME TESTS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"後方互換性テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """メインテスト実行"""
    print("Issue #643 OptimizationConfig Template Generation Improvements Test\n")

    tests = [
        ("標準JSONテンプレート生成", test_standard_json_template),
        ("コメント付きテンプレート生成", test_commented_template),
        ("テンプレートスキーマ機能", test_template_schema),
        ("後方互換性", test_backward_compatibility)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"実行中: {test_name}")
            print('='*60)

            if test_func():
                print(f"OK {test_name}: PASS")
                passed += 1
            else:
                print(f"NG {test_name}: FAIL")
                failed += 1

        except Exception as e:
            print(f"NG {test_name}: ERROR - {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"=== Final Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    if failed == 0:
        print("OK Issue #643 OptimizationConfig Template Generation: ALL TESTS PASSED")
        return True
    else:
        print("NG Issue #643 OptimizationConfig Template Generation: SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)