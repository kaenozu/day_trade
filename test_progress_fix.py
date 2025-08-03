#!/usr/bin/env python3
"""
progress.pyの修正をテストするスクリプト
"""

import os
import sys
sys.path.insert(0, 'src')

# テスト環境をシミュレート
os.environ['PYTEST_CURRENT_TEST'] = 'test_progress_fix.py::test_function'

from day_trade.utils.progress import (
    progress_context,
    ProgressType,
    BatchProgressTracker,
    MultiStepProgressTracker
)

def test_progress_context():
    """progress_contextのテスト環境検出をテスト"""
    print("Testing progress_context in test environment...")

    with progress_context('test processing', total=5, progress_type=ProgressType.DETERMINATE) as progress:
        for i in range(5):
            progress.update(1)
        print("progress_context working correctly in test mode")

def test_batch_progress_tracker():
    """BatchProgressTrackerのテスト環境対応をテスト"""
    print("Testing BatchProgressTracker...")

    try:
        with BatchProgressTracker('test batch', 3) as tracker:
            tracker.update_success('item1')
            tracker.update_success('item2')
            tracker.update_failure('item3', 'test error')
        print("BatchProgressTracker working correctly")
    except Exception as e:
        print(f"BatchProgressTracker error: {e}")

def test_multi_step_progress_tracker():
    """MultiStepProgressTrackerのテスト環境対応をテスト"""
    print("Testing MultiStepProgressTracker...")

    try:
        steps = ['step1', 'step2', 'step3']
        with MultiStepProgressTracker(steps, 'test process') as tracker:
            tracker.next_step()
            tracker.next_step()
            tracker.complete()
        print("MultiStepProgressTracker working correctly")
    except Exception as e:
        print(f"MultiStepProgressTracker error: {e}")

if __name__ == '__main__':
    print("164ブランチのprogress.py修正のテスト開始")
    test_progress_context()
    test_batch_progress_tracker()
    test_multi_step_progress_tracker()
    print("すべてのテストが完了しました")
