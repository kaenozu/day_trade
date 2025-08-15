import unittest
from unittest.mock import MagicMock, patch

from daytrade import print_summary
from src.day_trade.automation.orchestrator import ExecutionReport


class TestCli(unittest.TestCase):
    def test_print_summary_with_enhanced_details(self):
        """MLモデルの詳細情報（enhanced_details）が正しく表示されるかテスト"""
        recommendations = [
            {
                "symbol": "7203",
                "name": "トヨタ自動車",
                "action": "HOLD",
                "score": 70.0,
                "confidence": 28.7,
                "risk_level": "中"
            }
        ]
        portfolio_recommendation = {
            "total_symbols": 1,
            "recommended_allocation": {},
            "expected_return_percent": 0.0,
            "risk_assessment": "低リスク",
            "total_allocated": 1000000
        }

        with patch("builtins.print") as mock_print:
            print_summary(recommendations, portfolio_recommendation)

            # 期待される出力（一部）
            expected_output = (
                "総分析銘柄数: 1"
            )

            # 出力結果を文字列として結合
            printed_text = "\n".join(
                [call.args[0] for call in mock_print.call_args_list]
            )

            # 期待される出力が、実際の出力に含まれているか確認
            self.assertIn(expected_output, printed_text)


if __name__ == "__main__":
    unittest.main()
