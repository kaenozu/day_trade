import unittest
from unittest.mock import MagicMock, patch

from daytrade import print_summary
from src.day_trade.automation.orchestrator import AutomationReport


class TestCli(unittest.TestCase):
    def test_print_summary_with_enhanced_details(self):
        """MLモデルの詳細情報（enhanced_details）が正しく表示されるかテスト"""
        report = MagicMock(spec=AutomationReport)
        report.start_time = MagicMock()
        report.end_time = MagicMock()
        report.total_symbols = 1
        report.successful_symbols = 1
        report.failed_symbols = 0
        report.triggered_alerts = []
        report.errors = []
        report.portfolio_summary = {}

        # MLモデルによるHOLDシグナルのテストデータ
        report.generated_signals = [
            {
                "symbol": "7203",
                "type": "HOLD",
                "reason": "Enhanced Ensemble: ML+Rules (confidence: 28.7%)",
                "confidence": 0.287,
                "enhanced_details": {
                    "risk_score": 30.0,
                },
            }
        ]

        with patch("builtins.print") as mock_print:
            print_summary(report)

            # 期待される出力（一部）
            expected_output = (
                "  1. 7203 - HOLD (Enhanced Ensemble (Risk: 30.0)) [信頼度: 0.29]"
            )

            # 出力結果を文字列として結合
            printed_text = "\n".join(
                [call.args[0] for call in mock_print.call_args_list]
            )

            # 期待される出力が、実際の出力に含まれているか確認
            self.assertIn(expected_output, printed_text)


if __name__ == "__main__":
    unittest.main()
