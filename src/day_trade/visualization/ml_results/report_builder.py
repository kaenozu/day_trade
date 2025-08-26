#!/usr/bin/env python3
"""
機械学習結果可視化システム - レポート作成機能

Issue #315: 高度テクニカル指標・ML機能拡張
分析結果のテキストレポート生成機能
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
import numpy as np

from .types import (
    LSTMResults,
    VolatilityResults, 
    MultiFrameResults,
    TradingAction,
    VisualizationConfig,
    JAPANESE_LABELS,
    TIMEFRAMES
)
from .data_processor import DataProcessor

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ReportBuilder:
    """
    分析レポート作成クラス
    
    機械学習分析結果から詳細なテキストレポートを生成
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        初期化
        
        Args:
            config: 可視化設定
        """
        self.config = config or VisualizationConfig("output/ml_visualizations")
        self.data_processor = DataProcessor()
        logger.info("レポートビルダー初期化完了")
    
    def generate_analysis_report(
        self, 
        symbol: str, 
        results_dict: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        分析レポート生成（テキスト形式）
        
        Args:
            symbol: 銘柄コード
            results_dict: 全分析結果の辞書
            save_path: 保存パス
            
        Returns:
            保存されたファイルパス
        """
        try:
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_analysis_report.txt"
            else:
                save_path = Path(save_path)
            
            report_content = self._build_complete_report(symbol, results_dict)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            logger.info(f"分析レポート保存完了: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"分析レポート生成エラー: {e}")
            return None
    
    def generate_summary_report(
        self,
        symbol: str,
        results_dict: Dict,
        save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        要約レポート生成
        
        Args:
            symbol: 銘柄コード
            results_dict: 全分析結果の辞書
            save_path: 保存パス
            
        Returns:
            保存されたファイルパス
        """
        try:
            if save_path is None:
                save_path = self.config.output_dir / f"{symbol}_summary_report.txt"
            else:
                save_path = Path(save_path)
            
            summary_content = self._build_summary_report(symbol, results_dict)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(summary_content)
            
            logger.info(f"要約レポート保存完了: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"要約レポート生成エラー: {e}")
            return None
    
    def _build_complete_report(self, symbol: str, results_dict: Dict) -> str:
        """完全レポートの構築"""
        sections = []
        
        # ヘッダー
        sections.append(self._build_header(symbol))
        
        # LSTM分析セクション
        if "lstm" in results_dict:
            sections.append(self._build_lstm_section(results_dict["lstm"]))
        
        # ボラティリティ分析セクション
        if "volatility" in results_dict:
            sections.append(self._build_volatility_section(results_dict["volatility"]))
        
        # マルチタイムフレーム分析セクション
        if "multiframe" in results_dict:
            sections.append(self._build_multiframe_section(results_dict["multiframe"]))
        
        # テクニカル分析セクション
        if "technical" in results_dict:
            sections.append(self._build_technical_section(results_dict["technical"]))
        
        # 統合判断セクション
        sections.append(self._build_integrated_assessment(results_dict))
        
        # フッター
        sections.append(self._build_footer())
        
        return "\n\n".join(sections)
    
    def _build_summary_report(self, symbol: str, results_dict: Dict) -> str:
        """要約レポートの構築"""
        sections = []
        
        # 簡易ヘッダー
        sections.append(f"機械学習分析サマリー - {symbol}")
        sections.append("=" * 40)
        sections.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append("")
        
        # 各分析の要約
        summary_data = self._extract_summary_data(results_dict)
        
        # 統合判定
        sections.append("【統合判定】")
        final_assessment = summary_data.get("final_assessment", {})
        if final_assessment:
            sections.append(f"推奨アクション: {final_assessment.get('final_action', 'HOLD')}")
            sections.append(f"信頼度: {final_assessment.get('confidence', 0):.1f}%")
            sections.append("")
        
        # 主要指標サマリー
        sections.append("【主要指標】")
        
        if "lstm" in results_dict:
            lstm_data = results_dict["lstm"]
            if hasattr(lstm_data, 'confidence_score'):
                sections.append(f"LSTM信頼度: {lstm_data.confidence_score:.1f}%")
            else:
                confidence = lstm_data.get('confidence_score', 0)
                sections.append(f"LSTM信頼度: {confidence:.1f}%")
        
        if "volatility" in results_dict:
            vol_data = results_dict["volatility"]
            if hasattr(vol_data, 'risk_assessment'):
                risk_level = vol_data.risk_assessment.risk_level
                sections.append(f"リスクレベル: {risk_level}")
            else:
                risk_assessment = vol_data.get('risk_assessment', {})
                risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
                sections.append(f"リスクレベル: {risk_level}")
        
        if "multiframe" in results_dict:
            mf_data = results_dict["multiframe"]
            if hasattr(mf_data, 'integrated_analysis'):
                overall_trend = mf_data.integrated_analysis.overall_trend
                consistency = mf_data.integrated_analysis.consistency_score
                sections.append(f"総合トレンド: {overall_trend}")
                sections.append(f"時間軸整合性: {consistency:.1f}%")
            else:
                integrated = mf_data.get('integrated_analysis', {})
                overall_trend = integrated.get('overall_trend', 'unknown')
                consistency = integrated.get('consistency_score', 0)
                sections.append(f"総合トレンド: {overall_trend}")
                sections.append(f"時間軸整合性: {consistency:.1f}%")
        
        return "\n".join(sections)
    
    def _build_header(self, symbol: str) -> str:
        """ヘッダーセクションの構築"""
        header = f"""機械学習統合分析レポート
{'=' * 50}

銘柄: {symbol}
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
分析システム: ML Results Visualization System v1.0"""
        return header
    
    def _build_lstm_section(self, lstm_results) -> str:
        """LSTMセクションの構築"""
        lines = ["【LSTM時系列予測】"]
        
        if hasattr(lstm_results, 'confidence_score'):
            # データクラス形式
            lines.append(f"予測信頼度: {lstm_results.confidence_score:.1f}%")
            
            if hasattr(lstm_results, 'predicted_prices') and lstm_results.predicted_prices:
                lines.append(f"予測価格数: {len(lstm_results.predicted_prices)}個")
                price_range = f"{min(lstm_results.predicted_prices):.0f} - {max(lstm_results.predicted_prices):.0f}"
                lines.append(f"価格レンジ: {price_range}")
            
            if hasattr(lstm_results, 'predicted_returns') and lstm_results.predicted_returns:
                avg_return = np.mean(lstm_results.predicted_returns)
                lines.append(f"平均予測リターン: {avg_return:.2f}%")
                
                positive_returns = sum(1 for r in lstm_results.predicted_returns if r > 0)
                lines.append(f"ポジティブリターン: {positive_returns}/{len(lstm_results.predicted_returns)}回")
        else:
            # 辞書形式
            lines.append(f"予測信頼度: {lstm_results.get('confidence_score', 0):.1f}%")
            
            if "predicted_prices" in lstm_results and lstm_results["predicted_prices"]:
                pred_prices = lstm_results["predicted_prices"]
                lines.append(f"予測価格数: {len(pred_prices)}個")
                price_range = f"{min(pred_prices):.0f} - {max(pred_prices):.0f}"
                lines.append(f"価格レンジ: {price_range}")
            
            if "predicted_returns" in lstm_results and lstm_results["predicted_returns"]:
                pred_returns = lstm_results["predicted_returns"]
                avg_return = np.mean(pred_returns)
                lines.append(f"平均予測リターン: {avg_return:.2f}%")
                
                positive_returns = sum(1 for r in pred_returns if r > 0)
                lines.append(f"ポジティブリターン: {positive_returns}/{len(pred_returns)}回")
        
        return "\n".join(lines)
    
    def _build_volatility_section(self, volatility_results) -> str:
        """ボラティリティセクションの構築"""
        lines = ["【ボラティリティ予測】"]
        
        if hasattr(volatility_results, 'current_metrics'):
            # データクラス形式
            current_metrics = volatility_results.current_metrics
            lines.append(f"現在の実現ボラティリティ: {current_metrics.realized_volatility * 100:.1f}%")
            lines.append(f"VIX風指標: {current_metrics.vix_like_indicator:.1f}")
            lines.append(f"ボラティリティレジーム: {current_metrics.volatility_regime}")
            
            if hasattr(volatility_results, 'ensemble_forecast'):
                ensemble = volatility_results.ensemble_forecast
                lines.append(f"アンサンブル予測: {ensemble.ensemble_volatility:.1f}%")
                lines.append(f"予測信頼度: {ensemble.ensemble_confidence:.2f}")
            
            if hasattr(volatility_results, 'risk_assessment'):
                risk = volatility_results.risk_assessment
                lines.append(f"リスクレベル: {risk.risk_level}")
                lines.append(f"リスクスコア: {risk.risk_score}点")
                
                if risk.risk_factors:
                    lines.append("リスク要因:")
                    for factor in risk.risk_factors[:3]:
                        lines.append(f"  • {factor}")
        else:
            # 辞書形式
            current_metrics = volatility_results.get("current_metrics", {})
            if current_metrics:
                lines.append(f"現在の実現ボラティリティ: {current_metrics.get('realized_volatility', 0) * 100:.1f}%")
                lines.append(f"VIX風指標: {current_metrics.get('vix_like_indicator', 0):.1f}")
            
            ensemble = volatility_results.get("ensemble_forecast", {})
            if ensemble:
                lines.append(f"アンサンブル予測: {ensemble.get('ensemble_volatility', 0):.1f}%")
                lines.append(f"予測信頼度: {ensemble.get('ensemble_confidence', 0):.2f}")
            
            risk_assessment = volatility_results.get("risk_assessment", {})
            if risk_assessment:
                lines.append(f"リスクレベル: {risk_assessment.get('risk_level', 'UNKNOWN')}")
                lines.append(f"リスクスコア: {risk_assessment.get('risk_score', 0)}点")
        
        return "\n".join(lines)
    
    def _build_multiframe_section(self, multiframe_results) -> str:
        """マルチタイムフレームセクションの構築"""
        lines = ["【マルチタイムフレーム分析】"]
        
        if hasattr(multiframe_results, 'integrated_analysis'):
            # データクラス形式
            integrated = multiframe_results.integrated_analysis
            lines.append(f"総合トレンド: {integrated.overall_trend}")
            lines.append(f"トレンド信頼度: {integrated.trend_confidence:.1f}%")
            lines.append(f"時間軸整合性: {integrated.consistency_score:.1f}%")
            
            signal = integrated.integrated_signal
            lines.append(f"統合シグナル: {signal.action} ({signal.strength})")
            lines.append(f"シグナルスコア: {signal.signal_score:.1f}")
            
            recommendation = integrated.investment_recommendation
            lines.append(f"投資推奨: {recommendation.recommendation}")
            lines.append(f"ポジションサイズ: {recommendation.position_size}")
            lines.append(f"推奨信頼度: {recommendation.confidence:.1f}%")
            
            if recommendation.reasons:
                lines.append("推奨理由:")
                for reason in recommendation.reasons[:3]:
                    lines.append(f"  • {reason}")
            
            if hasattr(multiframe_results, 'timeframes'):
                lines.append("\n時間軸別分析:")
                for tf_key, tf_data in multiframe_results.timeframes.items():
                    tf_name = tf_data.timeframe if hasattr(tf_data, 'timeframe') else tf_key
                    trend = tf_data.trend_direction if hasattr(tf_data, 'trend_direction') else 'unknown'
                    strength = tf_data.trend_strength if hasattr(tf_data, 'trend_strength') else 0
                    lines.append(f"  {tf_name}: {trend} (強度: {strength:.1f})")
        else:
            # 辞書形式
            integrated = multiframe_results.get("integrated_analysis", {})
            if integrated:
                lines.append(f"総合トレンド: {integrated.get('overall_trend', 'unknown')}")
                lines.append(f"トレンド信頼度: {integrated.get('trend_confidence', 0):.1f}%")
                lines.append(f"時間軸整合性: {integrated.get('consistency_score', 0):.1f}%")
                
                signal = integrated.get("integrated_signal", {})
                if signal:
                    lines.append(f"統合シグナル: {signal.get('action', 'HOLD')}")
                
                recommendation = integrated.get("investment_recommendation", {})
                if recommendation:
                    lines.append(f"投資推奨: {recommendation.get('recommendation', 'HOLD')}")
                    lines.append(f"ポジションサイズ: {recommendation.get('position_size', 'NEUTRAL')}")
        
        return "\n".join(lines)
    
    def _build_technical_section(self, technical_results: Dict) -> str:
        """テクニカル分析セクションの構築"""
        lines = ["【テクニカル分析】"]
        
        # 基本的なテクニカル指標情報を表示
        for indicator, value in technical_results.items():
            if isinstance(value, (int, float)):
                lines.append(f"{indicator}: {value:.2f}")
            else:
                lines.append(f"{indicator}: {value}")
        
        return "\n".join(lines)
    
    def _build_integrated_assessment(self, results_dict: Dict) -> str:
        """統合判断セクションの構築"""
        lines = ["【統合投資判断】"]
        
        # 各分析からの信号を集計
        signals = []
        confidences = []
        
        # LSTM信号
        if "lstm" in results_dict:
            lstm_results = results_dict["lstm"]
            if hasattr(lstm_results, 'predicted_returns'):
                returns = lstm_results.predicted_returns
            else:
                returns = lstm_results.get("predicted_returns", [])
            
            if returns:
                avg_return = np.mean(returns)
                lstm_signal = (
                    "BUY" if avg_return > 1
                    else "SELL" if avg_return < -1 else "HOLD"
                )
                signals.append(lstm_signal)
                
                if hasattr(lstm_results, 'confidence_score'):
                    confidences.append(lstm_results.confidence_score)
                else:
                    confidences.append(lstm_results.get("confidence_score", 0))
        
        # マルチタイムフレーム信号
        if "multiframe" in results_dict:
            mf_results = results_dict["multiframe"]
            if hasattr(mf_results, 'integrated_analysis'):
                integrated = mf_results.integrated_analysis
                signals.append(str(integrated.integrated_signal.action))
                confidences.append(integrated.trend_confidence)
            else:
                integrated = mf_results.get("integrated_analysis", {})
                signal_info = integrated.get("integrated_signal", {})
                signals.append(signal_info.get("action", "HOLD"))
                confidences.append(integrated.get("trend_confidence", 0))
        
        # 統合判定
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL") 
        hold_count = signals.count("HOLD")
        
        if buy_count > sell_count and buy_count > hold_count:
            final_signal = "BUY"
        elif sell_count > buy_count and sell_count > hold_count:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        
        avg_confidence = np.mean(confidences) if confidences else 0
        
        lines.append(f"最終判定: {final_signal}")
        lines.append(f"統合信頼度: {avg_confidence:.1f}%")
        lines.append(f"シグナル分布: BUY({buy_count}) SELL({sell_count}) HOLD({hold_count})")
        
        # リスク評価サマリー
        if "volatility" in results_dict:
            vol_results = results_dict["volatility"]
            if hasattr(vol_results, 'risk_assessment'):
                risk_level = vol_results.risk_assessment.risk_level
            else:
                risk_assessment = vol_results.get("risk_assessment", {})
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
            
            lines.append(f"総合リスクレベル: {risk_level}")
        
        return "\n".join(lines)
    
    def _build_footer(self) -> str:
        """フッターセクションの構築"""
        footer = f"""{'=' * 50}
レポート終了

注意事項:
• 本レポートは機械学習による分析結果であり、投資判断の参考情報です
• 実際の投資では、複数の情報源を総合的に検討してください
• 過去の成績は将来の成果を保証するものではありません

ML Results Visualization System v1.0
生成システム: day_trade_sub"""
        return footer
    
    def _extract_summary_data(self, results_dict: Dict) -> Dict[str, Any]:
        """要約データの抽出"""
        summary = {
            'available_analyses': [],
            'key_metrics': {},
            'final_assessment': {}
        }
        
        # 利用可能な分析の確認
        for analysis_type in ['lstm', 'volatility', 'multiframe', 'technical']:
            if analysis_type in results_dict:
                summary['available_analyses'].append(analysis_type)
        
        # 統合評価の実行（data_processorを使用）
        merged_results = self.data_processor.merge_analysis_results(
            lstm_results=results_dict.get('lstm'),
            volatility_results=results_dict.get('volatility'),
            multiframe_results=results_dict.get('multiframe'),
            technical_results=results_dict.get('technical')
        )
        
        summary['final_assessment'] = merged_results.get('final_assessment', {})
        
        return summary