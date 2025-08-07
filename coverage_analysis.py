#!/usr/bin/env python3
"""
Coverage Analysis Script - ASCII Safe Version

Analyzes current test coverage for the analysis-only system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def count_lines_in_file(file_path: Path) -> tuple[int, int]:
    """
    Count total lines and executable lines in a Python file.

    Returns:
        tuple[int, int]: (total_lines, executable_lines)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        executable_lines = 0

        for line in lines:
            stripped = line.strip()
            # Skip empty lines, comments, and docstrings
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith('"""')
                and not stripped.startswith("'''")
                and stripped != '"""'
                and stripped != "'''"
            ):
                executable_lines += 1

        return total_lines, executable_lines
    except Exception:
        return 0, 0


def analyze_directory_coverage(base_path: Path) -> dict:
    """Analyze coverage for a directory."""
    coverage_data = {
        "total_files": 0,
        "total_lines": 0,
        "executable_lines": 0,
        "files": [],
    }

    python_files = list(base_path.glob("**/*.py"))

    for py_file in python_files:
        # Skip test files and __pycache__
        if (
            "test_" in py_file.name
            or "__pycache__" in str(py_file)
            or py_file.name.startswith("test")
        ):
            continue

        total_lines, executable_lines = count_lines_in_file(py_file)

        if total_lines > 0:
            relative_path = py_file.relative_to(base_path)
            coverage_data["files"].append(
                {
                    "file": str(relative_path),
                    "total_lines": total_lines,
                    "executable_lines": executable_lines,
                }
            )

            coverage_data["total_files"] += 1
            coverage_data["total_lines"] += total_lines
            coverage_data["executable_lines"] += executable_lines

    return coverage_data


def count_test_files(base_path: Path) -> dict:
    """Count test files and their coverage."""
    test_data = {"test_files": 0, "total_test_lines": 0, "files": []}

    # Count test files in tests/ directory
    tests_dir = base_path / "tests"
    if tests_dir.exists():
        test_files = list(tests_dir.glob("**/*.py"))
        for test_file in test_files:
            if test_file.name != "__init__.py":
                total_lines, _ = count_lines_in_file(test_file)
                if total_lines > 0:
                    relative_path = test_file.relative_to(base_path)
                    test_data["files"].append(
                        {"file": str(relative_path), "lines": total_lines}
                    )
                    test_data["test_files"] += 1
                    test_data["total_test_lines"] += total_lines

    # Count standalone test files
    standalone_tests = list(base_path.glob("test_*.py"))
    for test_file in standalone_tests:
        total_lines, _ = count_lines_in_file(test_file)
        if total_lines > 0:
            test_data["files"].append({"file": test_file.name, "lines": total_lines})
            test_data["test_files"] += 1
            test_data["total_test_lines"] += total_lines

    return test_data


def analyze_new_analysis_system_coverage() -> dict:
    """Analyze coverage for new analysis-only system components."""
    base_path = Path.cwd()
    src_path = base_path / "src" / "day_trade"

    new_components = [
        "config/trading_mode_config.py",
        "automation/analysis_only_engine.py",
        "automation/trading_engine.py",  # Modified for analysis mode
        "analysis/enhanced_report_manager.py",
        "analysis/market_analysis_system.py",
        "dashboard/analysis_dashboard_server.py",
        "core/integrated_analysis_system.py",
    ]

    analysis_system_data = {
        "total_files": 0,
        "total_lines": 0,
        "executable_lines": 0,
        "files": [],
    }

    for component in new_components:
        file_path = src_path / component
        if file_path.exists():
            total_lines, executable_lines = count_lines_in_file(file_path)
            if total_lines > 0:
                analysis_system_data["files"].append(
                    {
                        "file": component,
                        "total_lines": total_lines,
                        "executable_lines": executable_lines,
                    }
                )
                analysis_system_data["total_files"] += 1
                analysis_system_data["total_lines"] += total_lines
                analysis_system_data["executable_lines"] += executable_lines

    return analysis_system_data


def generate_coverage_report():
    """Generate comprehensive coverage report."""
    print("Coverage Analysis Started")
    print("=" * 80)

    base_path = Path.cwd()
    src_path = base_path / "src" / "day_trade"

    if not src_path.exists():
        print("ERROR: src/day_trade directory not found")
        return

    # Analyze main source code
    print("=== Main Source Code Analysis ===")
    source_coverage = analyze_directory_coverage(src_path)

    print(f"Total Source Files: {source_coverage['total_files']}")
    print(f"Total Lines: {source_coverage['total_lines']}")
    print(f"Executable Lines: {source_coverage['executable_lines']}")

    # Analyze test coverage
    print("\n=== Test Coverage Analysis ===")
    test_data = count_test_files(base_path)

    print(f"Test Files: {test_data['test_files']}")
    print(f"Total Test Lines: {test_data['total_test_lines']}")

    # Calculate test-to-source ratio
    if source_coverage["executable_lines"] > 0:
        test_ratio = (
            test_data["total_test_lines"] / source_coverage["executable_lines"]
        ) * 100
        print(f"Test-to-Source Ratio: {test_ratio:.1f}%")

    # Analyze new analysis system components
    print("\n=== New Analysis System Components ===")
    analysis_coverage = analyze_new_analysis_system_coverage()

    print(f"Analysis System Files: {analysis_coverage['total_files']}")
    print(f"Analysis System Lines: {analysis_coverage['total_lines']}")
    print(f"Analysis System Executable Lines: {analysis_coverage['executable_lines']}")

    # Detailed analysis system breakdown
    if analysis_coverage["files"]:
        print("\nAnalysis System Files:")
        for file_info in analysis_coverage["files"]:
            print(
                f"  - {file_info['file']}: {file_info['total_lines']} lines "
                f"({file_info['executable_lines']} executable)"
            )

    # Estimate current coverage based on existing tests
    print("\n=== Coverage Estimation ===")

    # Count existing test files that test analysis components
    analysis_test_files = [
        "test_coverage_analysis_system.py",
        "test_daytrade_ascii_safe.py",
        "test_analysis_system.py",
        "test_enhanced_system_simple.py",
    ]

    tested_lines = 0
    for test_file in analysis_test_files:
        test_path = base_path / test_file
        if test_path.exists():
            lines, _ = count_lines_in_file(test_path)
            tested_lines += lines * 2  # Estimate: each test line covers ~2 source lines

    if source_coverage["executable_lines"] > 0:
        estimated_coverage = min(
            100, (tested_lines / source_coverage["executable_lines"]) * 100
        )
        print(f"Estimated Coverage: {estimated_coverage:.1f}%")

        if estimated_coverage >= 80:
            quality = "EXCELLENT"
            color = "green"
        elif estimated_coverage >= 60:
            quality = "GOOD"
            color = "yellow"
        elif estimated_coverage >= 40:
            quality = "FAIR"
            color = "orange"
        else:
            quality = "NEEDS IMPROVEMENT"
            color = "red"  # Color for future use

        print(f"Coverage Quality: {quality}")

    # Test file breakdown
    print("\n=== Test File Breakdown ===")
    if test_data["files"]:
        sorted_tests = sorted(
            test_data["files"], key=lambda x: x["lines"], reverse=True
        )
        for test_info in sorted_tests[:10]:  # Top 10 largest test files
            print(f"  - {test_info['file']}: {test_info['lines']} lines")

    # Recommendations
    print("\n=== Recommendations ===")
    if estimated_coverage < 60:
        print("1. Add more comprehensive tests for core components")
        print("2. Focus on testing analysis_only_engine.py and trading_engine.py")
        print("3. Add integration tests for dashboard and report systems")
    elif estimated_coverage < 80:
        print("1. Add edge case testing")
        print("2. Increase test coverage for error handling")
        print("3. Add performance and load testing")
    else:
        print("1. Maintain current high coverage")
        print("2. Add regression tests for new features")
        print("3. Consider mutation testing for quality validation")

    print("\n=== Safe Mode Verification ===")
    try:
        from src.day_trade.config.trading_mode_config import (
            get_current_trading_config,
            is_safe_mode,
        )

        safe_mode = is_safe_mode()
        config = get_current_trading_config()

        print(f"Safe Mode: {'ENABLED' if safe_mode else 'DISABLED'}")
        print(
            f"Automatic Trading: {'DISABLED' if not config.enable_automatic_trading else 'ENABLED'}"
        )
        print(
            f"Order Execution: {'DISABLED' if not config.enable_order_execution else 'ENABLED'}"
        )
        print(f"Order API: {'DISABLED' if config.disable_order_api else 'ENABLED'}")

    except Exception as e:
        print(f"Safe mode verification error: {e}")

    print("\n" + "=" * 80)
    print("Coverage Analysis Completed")
    print("=" * 80)


if __name__ == "__main__":
    generate_coverage_report()
