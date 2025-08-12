"""
プラグインマネージャー

動的プラグイン読み込み・ホットリロード・ライフサイクル管理
"""

import importlib
import importlib.util
import inspect
import json
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ホットリロード機能（オプション）
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object
    Observer = None

from ..utils.logging_config import get_context_logger
from .interfaces import (
    IRiskPlugin,
    MarketData,
    PluginInfo,
    PortfolioData,
    RiskAnalysisResult,
    RiskLevel,
)
from .sandbox import SecuritySandbox

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class PluginEventHandler(FileSystemEventHandler):
    """プラグインファイル変更監視"""

    def __init__(self, manager: "PluginManager"):
        self.manager = manager
        super().__init__()

    def on_modified(self, event):
        """ファイル変更時処理"""
        if event.is_directory or not event.src_path.endswith(".py"):
            return

        plugin_path = Path(event.src_path)
        if plugin_path.stem.startswith("plugin_"):
            logger.info(f"プラグインファイル変更検出: {plugin_path}")
            self.manager._schedule_reload(plugin_path.stem[7:])  # 'plugin_'を除去


class PluginManager:
    """プラグインマネージャー"""

    def __init__(
        self,
        plugins_dir: str = "plugins",
        config_file: Optional[str] = None,
        enable_hot_reload: bool = True,
        security_enabled: bool = True,
    ):
        """
        初期化

        Args:
            plugins_dir: プラグインディレクトリ
            config_file: 設定ファイルパス
            enable_hot_reload: ホットリロード有効
            security_enabled: セキュリティサンドボックス有効
        """
        self.plugins_dir = Path(plugins_dir)
        self.config_file = config_file
        self.enable_hot_reload = enable_hot_reload
        self.security_enabled = security_enabled

        # プラグイン管理
        self._plugins: Dict[str, IRiskPlugin] = {}
        self._plugin_infos: Dict[str, PluginInfo] = {}
        self._plugin_modules: Dict[str, Any] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}

        # ホットリロード管理
        self._observer: Optional[Observer] = None
        self._reload_queue: List[str] = []
        self._reload_lock = threading.Lock()

        # セキュリティサンドボックス
        self.sandbox: Optional[SecuritySandbox] = None
        if security_enabled:
            self.sandbox = SecuritySandbox()

        # 初期化
        self._initialize()

        logger.info("プラグインマネージャー初期化完了")

    def _initialize(self):
        """内部初期化処理"""
        # プラグインディレクトリ作成
        self.plugins_dir.mkdir(parents=True, exist_ok=True)

        # 設定読み込み
        if self.config_file and Path(self.config_file).exists():
            self._load_config()

        # 組み込みプラグイン作成
        self._create_builtin_plugins()

        # プラグイン読み込み
        self._load_all_plugins()

        # ホットリロード開始
        if self.enable_hot_reload:
            self._start_hot_reload()

    def _load_config(self):
        """設定ファイル読み込み"""
        try:
            with open(self.config_file, encoding="utf-8") as f:
                config = json.load(f)
                self._plugin_configs = config.get("plugins", {})
                logger.info(
                    f"プラグイン設定読み込み完了: {len(self._plugin_configs)}件"
                )
        except Exception as e:
            logger.error(f"設定読み込みエラー: {e}")

    def _create_builtin_plugins(self):
        """組み込みプラグイン作成"""
        builtin_plugins = [
            self._create_industry_risk_plugin(),
            self._create_compliance_plugin(),
            self._create_news_sentiment_plugin(),
        ]

        for plugin_code, plugin_name in builtin_plugins:
            plugin_file = self.plugins_dir / f"plugin_{plugin_name}.py"
            if not plugin_file.exists():
                plugin_file.write_text(plugin_code, encoding="utf-8")
                logger.info(f"組み込みプラグイン作成: {plugin_name}")

    def _create_industry_risk_plugin(self) -> tuple:
        """業界リスクプラグイン作成"""
        code = '''"""
業界リスク分析プラグイン
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from src.day_trade.plugins.interfaces import (
    IRiskPlugin, PluginInfo, PluginType, RiskLevel,
    RiskAnalysisResult, MarketData, PortfolioData
)


class IndustryRiskPlugin(IRiskPlugin):
    """業界リスク分析プラグイン"""

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="Industry Risk Analyzer",
            version="1.0.0",
            description="業界セクター別リスク分析",
            author="Day Trade System",
            plugin_type=PluginType.RISK_ANALYZER,
            dependencies=[],
            config_schema={
                "risk_thresholds": {
                    "type": "object",
                    "properties": {
                        "high_correlation_threshold": {"type": "number", "default": 0.8},
                        "sector_concentration_limit": {"type": "number", "default": 0.4}
                    }
                }
            }
        )

    def initialize(self) -> bool:
        try:
            # 業界セクター定義
            self.sector_map = {
                "7203.T": "自動車", "7267.T": "自動車",
                "8306.T": "金融", "8316.T": "金融", "8411.T": "金融",
                "9984.T": "テクノロジー", "6758.T": "テクノロジー",
                "4502.T": "医薬品", "4568.T": "医薬品",
                "9433.T": "通信", "9432.T": "通信"
            }

            self.risk_weights = {
                "自動車": 0.15,
                "金融": 0.20,
                "テクノロジー": 0.25,
                "医薬品": 0.10,
                "通信": 0.12
            }

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            return False

    def analyze_risk(
        self,
        market_data: Union[MarketData, List[MarketData]],
        portfolio_data: Optional[PortfolioData] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAnalysisResult:

        try:
            if isinstance(market_data, MarketData):
                market_data = [market_data]

            # セクター集中度分析
            sector_exposure = self._calculate_sector_exposure(market_data, portfolio_data)

            # セクター相関分析
            sector_correlation = self._analyze_sector_correlation(market_data)

            # リスクスコア計算
            risk_score = self._calculate_risk_score(sector_exposure, sector_correlation)

            # リスクレベル判定
            risk_level = self._determine_risk_level(risk_score)

            # 推奨事項生成
            recommendations = self._generate_recommendations(sector_exposure, sector_correlation)

            # アラート生成
            alerts = self._generate_alerts(sector_exposure, sector_correlation)

            return RiskAnalysisResult(
                plugin_name=self.get_info().name,
                timestamp=datetime.now(),
                risk_level=risk_level,
                risk_score=risk_score,
                confidence=0.85,
                message=f"業界リスク分析完了 - スコア: {risk_score:.2f}",
                details={
                    "sector_exposure": sector_exposure,
                    "sector_correlation": sector_correlation,
                    "max_sector_weight": max(sector_exposure.values()) if sector_exposure else 0
                },
                recommendations=recommendations,
                alerts=alerts,
                metadata={"analysis_type": "industry_risk"}
            )

        except Exception as e:
            return RiskAnalysisResult(
                plugin_name=self.get_info().name,
                timestamp=datetime.now(),
                risk_level=RiskLevel.MODERATE,
                risk_score=0.5,
                confidence=0.0,
                message=f"分析エラー: {str(e)}",
                details={},
                recommendations=[],
                alerts=[f"業界リスク分析エラー: {str(e)}"],
                metadata={"error": True}
            )

    def _calculate_sector_exposure(
        self,
        market_data: List[MarketData],
        portfolio_data: Optional[PortfolioData]
    ) -> Dict[str, float]:
        """セクター別エクスポージャー計算"""
        sector_exposure = {}

        if portfolio_data and portfolio_data.allocation:
            for symbol, weight in portfolio_data.allocation.items():
                sector = self.sector_map.get(symbol, "その他")
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight

        return sector_exposure

    def _analyze_sector_correlation(self, market_data: List[MarketData]) -> float:
        """セクター間相関分析"""
        try:
            if len(market_data) < 2:
                return 0.0

            returns_data = {}
            for data in market_data:
                if not data.price_data.empty:
                    returns = data.price_data['Close'].pct_change().dropna()
                    sector = self.sector_map.get(data.symbol, "その他")
                    returns_data[sector] = returns.tail(252)  # 1年分

            if len(returns_data) < 2:
                return 0.0

            correlation_df = pd.DataFrame(returns_data).corr()
            avg_correlation = correlation_df.values[np.triu_indices_from(correlation_df.values, k=1)].mean()

            return float(avg_correlation)

        except Exception:
            return 0.0

    def _calculate_risk_score(
        self,
        sector_exposure: Dict[str, float],
        sector_correlation: float
    ) -> float:
        """リスクスコア計算"""
        # 集中リスクスコア
        concentration_risk = 0.0
        if sector_exposure:
            max_exposure = max(sector_exposure.values())
            concentration_risk = min(max_exposure / 0.4, 1.0)  # 40%を上限とする

        # 相関リスクスコア
        correlation_risk = min(abs(sector_correlation) / 0.8, 1.0)

        # 総合リスクスコア
        risk_score = (concentration_risk * 0.6 + correlation_risk * 0.4)

        return min(risk_score, 1.0)

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """リスクレベル判定"""
        if risk_score >= 0.8:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MODERATE
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _generate_recommendations(
        self,
        sector_exposure: Dict[str, float],
        sector_correlation: float
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        # 集中リスク対応
        if sector_exposure:
            max_sector = max(sector_exposure, key=sector_exposure.get)
            max_weight = sector_exposure[max_sector]

            if max_weight > 0.4:
                recommendations.append(f"{max_sector}セクターの比重が高すぎます。分散を検討してください。")

        # 相関リスク対応
        if sector_correlation > 0.8:
            recommendations.append("セクター間相関が高すぎます。異なる業界への分散を検討してください。")

        return recommendations

    def _generate_alerts(
        self,
        sector_exposure: Dict[str, float],
        sector_correlation: float
    ) -> List[str]:
        """アラート生成"""
        alerts = []

        if sector_exposure:
            max_weight = max(sector_exposure.values())
            if max_weight > 0.5:
                alerts.append("セクター集中リスク警告: 単一セクターが50%超")

        if sector_correlation > 0.85:
            alerts.append("高相関警告: セクター間相関が85%超")

        return alerts

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """設定検証"""
        try:
            thresholds = config.get('risk_thresholds', {})
            correlation_threshold = thresholds.get('high_correlation_threshold', 0.8)
            concentration_limit = thresholds.get('sector_concentration_limit', 0.4)

            return (0.0 <= correlation_threshold <= 1.0 and
                   0.0 <= concentration_limit <= 1.0)
        except Exception:
            return False
'''
        return code, "industry_risk"

    def _create_compliance_plugin(self) -> tuple:
        """コンプライアンスプラグイン作成"""
        code = '''"""
コンプライアンス分析プラグイン
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from src.day_trade.plugins.interfaces import (
    IRiskPlugin, PluginInfo, PluginType, RiskLevel,
    RiskAnalysisResult, MarketData, PortfolioData
)


class CompliancePlugin(IRiskPlugin):
    """コンプライアンス分析プラグイン"""

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="Compliance Checker",
            version="1.0.0",
            description="取引規制・コンプライアンスチェック",
            author="Day Trade System",
            plugin_type=PluginType.COMPLIANCE_CHECK,
            dependencies=[],
            config_schema={
                "limits": {
                    "type": "object",
                    "properties": {
                        "max_single_position": {"type": "number", "default": 0.3},
                        "max_daily_trades": {"type": "integer", "default": 50},
                        "max_leverage": {"type": "number", "default": 3.0}
                    }
                }
            }
        )

    def initialize(self) -> bool:
        try:
            self.limits = self.config.get('limits', {
                'max_single_position': 0.3,
                'max_daily_trades': 50,
                'max_leverage': 3.0
            })

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            return False

    def analyze_risk(
        self,
        market_data: Union[MarketData, List[MarketData]],
        portfolio_data: Optional[PortfolioData] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAnalysisResult:

        try:
            # ポジションサイズチェック
            position_violations = self._check_position_limits(portfolio_data)

            # 取引頻度チェック
            trading_violations = self._check_trading_frequency(context)

            # レバレッジチェック
            leverage_violations = self._check_leverage(portfolio_data)

            # 総合コンプライアンススコア
            total_violations = len(position_violations) + len(trading_violations) + len(leverage_violations)
            compliance_score = max(0.0, 1.0 - (total_violations * 0.2))

            # リスクレベル判定
            if total_violations == 0:
                risk_level = RiskLevel.VERY_LOW
            elif total_violations <= 2:
                risk_level = RiskLevel.LOW
            elif total_violations <= 4:
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.HIGH

            # アラート・推奨事項
            all_violations = position_violations + trading_violations + leverage_violations
            recommendations = [f"違反解消: {v}" for v in all_violations]

            return RiskAnalysisResult(
                plugin_name=self.get_info().name,
                timestamp=datetime.now(),
                risk_level=risk_level,
                risk_score=1.0 - compliance_score,
                confidence=0.95,
                message=f"コンプライアンス分析 - 違反数: {total_violations}",
                details={
                    "position_violations": position_violations,
                    "trading_violations": trading_violations,
                    "leverage_violations": leverage_violations,
                    "compliance_score": compliance_score
                },
                recommendations=recommendations,
                alerts=all_violations,
                metadata={"analysis_type": "compliance"}
            )

        except Exception as e:
            return RiskAnalysisResult(
                plugin_name=self.get_info().name,
                timestamp=datetime.now(),
                risk_level=RiskLevel.MODERATE,
                risk_score=0.5,
                confidence=0.0,
                message=f"分析エラー: {str(e)}",
                details={},
                recommendations=[],
                alerts=[f"コンプライアンス分析エラー: {str(e)}"],
                metadata={"error": True}
            )

    def _check_position_limits(self, portfolio_data: Optional[PortfolioData]) -> List[str]:
        """ポジション制限チェック"""
        violations = []

        if portfolio_data and portfolio_data.allocation:
            max_position = max(portfolio_data.allocation.values())
            if max_position > self.limits['max_single_position']:
                violations.append(f"単一ポジション制限超過: {max_position:.1%}")

        return violations

    def _check_trading_frequency(self, context: Optional[Dict[str, Any]]) -> List[str]:
        """取引頻度チェック"""
        violations = []

        if context and 'daily_trades' in context:
            daily_trades = context['daily_trades']
            if daily_trades > self.limits['max_daily_trades']:
                violations.append(f"1日取引回数制限超過: {daily_trades}回")

        return violations

    def _check_leverage(self, portfolio_data: Optional[PortfolioData]) -> List[str]:
        """レバレッジチェック"""
        violations = []

        if portfolio_data:
            total_value = portfolio_data.total_value
            available_cash = portfolio_data.available_cash

            if available_cash > 0:
                leverage = total_value / available_cash
                if leverage > self.limits['max_leverage']:
                    violations.append(f"レバレッジ制限超過: {leverage:.1f}倍")

        return violations

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """設定検証"""
        try:
            limits = config.get('limits', {})
            max_position = limits.get('max_single_position', 0.3)
            max_trades = limits.get('max_daily_trades', 50)
            max_leverage = limits.get('max_leverage', 3.0)

            return (0.0 < max_position <= 1.0 and
                   max_trades > 0 and
                   max_leverage > 0)
        except Exception:
            return False
'''
        return code, "compliance"

    def _create_news_sentiment_plugin(self) -> tuple:
        """ニュースセンチメントプラグイン作成"""
        code = '''"""
ニュースセンチメント分析プラグイン
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from src.day_trade.plugins.interfaces import (
    IRiskPlugin, PluginInfo, PluginType, RiskLevel,
    RiskAnalysisResult, MarketData, PortfolioData
)


class NewsSentimentPlugin(IRiskPlugin):
    """ニュースセンチメント分析プラグイン"""

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="News Sentiment Analyzer",
            version="1.0.0",
            description="ニュース・社会情勢センチメント分析",
            author="Day Trade System",
            plugin_type=PluginType.SENTIMENT_ANALYSIS,
            dependencies=[],
            config_schema={
                "sentiment_weights": {
                    "type": "object",
                    "properties": {
                        "negative_threshold": {"type": "number", "default": -0.3},
                        "positive_threshold": {"type": "number", "default": 0.3},
                        "news_decay_days": {"type": "integer", "default": 7}
                    }
                }
            }
        )

    def initialize(self) -> bool:
        try:
            # センチメント辞書設定
            self.positive_keywords = [
                "上昇", "成長", "拡大", "増加", "好調", "上方修正",
                "業績向上", "利益増", "株価上昇", "買い推奨"
            ]

            self.negative_keywords = [
                "下落", "減少", "悪化", "低下", "不調", "下方修正",
                "業績悪化", "損失", "株価下落", "売り推奨", "リスク"
            ]

            self.weights_config = self.config.get('sentiment_weights', {
                'negative_threshold': -0.3,
                'positive_threshold': 0.3,
                'news_decay_days': 7
            })

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"初期化エラー: {e}")
            return False

    def analyze_risk(
        self,
        market_data: Union[MarketData, List[MarketData]],
        portfolio_data: Optional[PortfolioData] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskAnalysisResult:

        try:
            # ニュースデータ収集
            news_data = self._collect_news_data(market_data, context)

            # センチメント分析実行
            sentiment_scores = self._analyze_sentiment(news_data)

            # ポートフォリオ影響度計算
            portfolio_sentiment = self._calculate_portfolio_sentiment(
                sentiment_scores, portfolio_data
            )

            # リスクスコア・レベル判定
            risk_score = self._calculate_sentiment_risk(portfolio_sentiment)
            risk_level = self._determine_risk_level(portfolio_sentiment)

            # 推奨事項・アラート生成
            recommendations = self._generate_recommendations(sentiment_scores, portfolio_sentiment)
            alerts = self._generate_alerts(sentiment_scores, portfolio_sentiment)

            return RiskAnalysisResult(
                plugin_name=self.get_info().name,
                timestamp=datetime.now(),
                risk_level=risk_level,
                risk_score=risk_score,
                confidence=0.75,
                message=f"センチメント分析完了 - スコア: {portfolio_sentiment:.2f}",
                details={
                    "individual_sentiments": sentiment_scores,
                    "portfolio_sentiment": portfolio_sentiment,
                    "news_count": len(news_data),
                    "analysis_period": self.weights_config['news_decay_days']
                },
                recommendations=recommendations,
                alerts=alerts,
                metadata={"analysis_type": "news_sentiment"}
            )

        except Exception as e:
            return RiskAnalysisResult(
                plugin_name=self.get_info().name,
                timestamp=datetime.now(),
                risk_level=RiskLevel.MODERATE,
                risk_score=0.5,
                confidence=0.0,
                message=f"分析エラー: {str(e)}",
                details={},
                recommendations=[],
                alerts=[f"センチメント分析エラー: {str(e)}"],
                metadata={"error": True}
            )

    def _collect_news_data(
        self,
        market_data: Union[MarketData, List[MarketData]],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """ニュースデータ収集"""
        news_data = []

        # 市場データからニュース抽出
        if isinstance(market_data, MarketData):
            market_data = [market_data]

        for data in market_data:
            if data.news_data:
                news_data.extend(data.news_data)

        # コンテキストからニュース抽出
        if context and 'news_data' in context:
            news_data.extend(context['news_data'])

        # 期間フィルタリング
        cutoff_date = datetime.now() - timedelta(days=self.weights_config['news_decay_days'])
        filtered_news = [
            news for news in news_data
            if datetime.fromisoformat(news.get('timestamp', datetime.now().isoformat())) >= cutoff_date
        ]

        return filtered_news

    def _analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """個別センチメント分析"""
        sentiment_scores = {}

        for news in news_data:
            symbol = news.get('symbol', 'GENERAL')
            title = news.get('title', '')
            content = news.get('content', '')

            # テキスト結合
            text = f"{title} {content}".lower()

            # キーワードマッチング
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in keyword in text)

            # センチメントスコア計算
            total_keywords = positive_count + negative_count
            if total_keywords > 0:
                sentiment = (positive_count - negative_count) / total_keywords
            else:
                sentiment = 0.0

            # 銘柄別集計
            if symbol not in sentiment_scores:
                sentiment_scores[symbol] = []
            sentiment_scores[symbol].append(sentiment)

        # 平均化
        for symbol in sentiment_scores:
            sentiment_scores[symbol] = sum(sentiment_scores[symbol]) / len(sentiment_scores[symbol])

        return sentiment_scores

    def _calculate_portfolio_sentiment(
        self,
        sentiment_scores: Dict[str, float],
        portfolio_data: Optional[PortfolioData]
    ) -> float:
        """ポートフォリオ全体センチメント計算"""
        if not portfolio_data or not portfolio_data.allocation:
            # 全市場平均
            if sentiment_scores:
                return sum(sentiment_scores.values()) / len(sentiment_scores)
            return 0.0

        weighted_sentiment = 0.0
        total_weight = 0.0

        for symbol, weight in portfolio_data.allocation.items():
            if symbol in sentiment_scores:
                weighted_sentiment += sentiment_scores[symbol] * weight
                total_weight += weight

        # 一般市場センチメントも考慮
        if 'GENERAL' in sentiment_scores and total_weight < 1.0:
            remaining_weight = 1.0 - total_weight
            weighted_sentiment += sentiment_scores['GENERAL'] * remaining_weight
            total_weight += remaining_weight

        return weighted_sentiment / total_weight if total_weight > 0 else 0.0

    def _calculate_sentiment_risk(self, portfolio_sentiment: float) -> float:
        """センチメントリスクスコア計算"""
        # 負のセンチメントをリスクとして扱う
        if portfolio_sentiment < self.weights_config['negative_threshold']:
            return min(abs(portfolio_sentiment), 1.0)
        elif portfolio_sentiment > self.weights_config['positive_threshold']:
            return 0.1  # ポジティブでも過熱リスク
        else:
            return 0.3  # 中立

    def _determine_risk_level(self, portfolio_sentiment: float) -> RiskLevel:
        """リスクレベル判定"""
        if portfolio_sentiment < -0.6:
            return RiskLevel.VERY_HIGH
        elif portfolio_sentiment < -0.3:
            return RiskLevel.HIGH
        elif portfolio_sentiment < 0.0:
            return RiskLevel.MODERATE
        elif portfolio_sentiment < 0.3:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _generate_recommendations(
        self,
        sentiment_scores: Dict[str, float],
        portfolio_sentiment: float
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        if portfolio_sentiment < -0.4:
            recommendations.append("ネガティブセンチメント: リスクオフ戦略を検討")
        elif portfolio_sentiment > 0.5:
            recommendations.append("過度なポジティブセンチメント: 利確検討")

        # 個別銘柄推奨
        for symbol, sentiment in sentiment_scores.items():
            if sentiment < -0.5:
                recommendations.append(f"{symbol}: 強いネガティブセンチメント")

        return recommendations

    def _generate_alerts(
        self,
        sentiment_scores: Dict[str, float],
        portfolio_sentiment: float
    ) -> List[str]:
        """アラート生成"""
        alerts = []

        if portfolio_sentiment < -0.6:
            alerts.append("緊急センチメント警告: 極度のネガティブ")

        negative_count = sum(1 for s in sentiment_scores.values() if s < -0.4)
        if negative_count >= 3:
            alerts.append(f"複数銘柄ネガティブ: {negative_count}銘柄")

        return alerts

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """設定検証"""
        try:
            weights = config.get('sentiment_weights', {})
            neg_threshold = weights.get('negative_threshold', -0.3)
            pos_threshold = weights.get('positive_threshold', 0.3)
            decay_days = weights.get('news_decay_days', 7)

            return (-1.0 <= neg_threshold < 0 and
                   0 < pos_threshold <= 1.0 and
                   decay_days > 0)
        except Exception:
            return False
'''
        return code, "news_sentiment"

    def _load_all_plugins(self):
        """全プラグイン読み込み"""
        plugin_files = self.plugins_dir.glob("plugin_*.py")

        for plugin_file in plugin_files:
            plugin_name = plugin_file.stem[7:]  # 'plugin_'を除去
            try:
                self._load_plugin(plugin_name, plugin_file)
            except Exception as e:
                logger.error(f"プラグイン読み込み失敗 {plugin_name}: {e}")

    def _load_plugin(self, plugin_name: str, plugin_file: Path):
        """個別プラグイン読み込み"""
        try:
            # セキュリティチェック
            if self.security_enabled and self.sandbox:
                if not self.sandbox.validate_plugin_file(plugin_file):
                    logger.warning(f"セキュリティ検証失敗: {plugin_name}")
                    return

            # モジュール動的読み込み
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_name}", plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # プラグインクラス検索
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, IRiskPlugin)
                    and obj != IRiskPlugin
                    and not name.startswith("I")
                ):
                    plugin_class = obj
                    break

            if not plugin_class:
                logger.error(f"プラグインクラス未発見: {plugin_name}")
                return

            # プラグインインスタンス作成
            plugin_config = self._plugin_configs.get(plugin_name, {})
            plugin_instance = plugin_class(plugin_config)

            # 初期化
            if plugin_instance.initialize():
                self._plugins[plugin_name] = plugin_instance
                self._plugin_infos[plugin_name] = plugin_instance.get_info()
                self._plugin_modules[plugin_name] = module

                logger.info(f"プラグイン読み込み完了: {plugin_name}")
            else:
                logger.error(f"プラグイン初期化失敗: {plugin_name}")

        except Exception as e:
            logger.error(f"プラグイン読み込みエラー {plugin_name}: {e}")

    def _start_hot_reload(self):
        """ホットリロード開始"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdogライブラリ未インストール - ホットリロード無効")
            return

        try:
            self._observer = Observer()
            event_handler = PluginEventHandler(self)
            self._observer.schedule(
                event_handler, str(self.plugins_dir), recursive=False
            )
            self._observer.start()
            logger.info("ホットリロード開始")
        except Exception as e:
            logger.error(f"ホットリロード開始エラー: {e}")

    def _schedule_reload(self, plugin_name: str):
        """プラグインリロード予約"""
        with self._reload_lock:
            if plugin_name not in self._reload_queue:
                self._reload_queue.append(plugin_name)
                # 遅延リロード実行
                threading.Timer(1.0, self._execute_reload).start()

    def _execute_reload(self):
        """プラグインリロード実行"""
        with self._reload_lock:
            while self._reload_queue:
                plugin_name = self._reload_queue.pop(0)
                try:
                    self.reload_plugin(plugin_name)
                except Exception as e:
                    logger.error(f"ホットリロードエラー {plugin_name}: {e}")

    def reload_plugin(self, plugin_name: str) -> bool:
        """プラグインリロード"""
        try:
            # 既存プラグインクリーンアップ
            if plugin_name in self._plugins:
                self._plugins[plugin_name].cleanup()
                del self._plugins[plugin_name]
                del self._plugin_infos[plugin_name]

            # モジュールリロード
            if plugin_name in self._plugin_modules:
                importlib.reload(self._plugin_modules[plugin_name])

            # 再読み込み
            plugin_file = self.plugins_dir / f"plugin_{plugin_name}.py"
            if plugin_file.exists():
                self._load_plugin(plugin_name, plugin_file)
                logger.info(f"プラグインリロード完了: {plugin_name}")
                return True
            else:
                logger.error(f"プラグインファイル未発見: {plugin_file}")
                return False

        except Exception as e:
            logger.error(f"プラグインリロードエラー {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[IRiskPlugin]:
        """プラグインインスタンス取得"""
        return self._plugins.get(plugin_name)

    def list_plugins(self) -> List[PluginInfo]:
        """プラグインリスト取得"""
        return list(self._plugin_infos.values())

    def enable_plugin(self, plugin_name: str) -> bool:
        """プラグイン有効化"""
        if plugin_name in self._plugin_infos:
            self._plugin_infos[plugin_name].enabled = True
            logger.info(f"プラグイン有効化: {plugin_name}")
            return True
        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """プラグイン無効化"""
        if plugin_name in self._plugin_infos:
            self._plugin_infos[plugin_name].enabled = False
            logger.info(f"プラグイン無効化: {plugin_name}")
            return True
        return False

    def analyze_with_all_plugins(
        self,
        market_data: Union[MarketData, List[MarketData]],
        portfolio_data: Optional[PortfolioData] = None,
        context: Optional[Dict[str, Any]] = None,
        plugin_filter: Optional[List[str]] = None,
    ) -> Dict[str, RiskAnalysisResult]:
        """全プラグインで分析実行"""
        results = {}

        target_plugins = plugin_filter or list(self._plugins.keys())

        for plugin_name in target_plugins:
            if plugin_name in self._plugins and self._plugin_infos[plugin_name].enabled:
                try:
                    plugin = self._plugins[plugin_name]
                    result = plugin.analyze_risk(market_data, portfolio_data, context)
                    results[plugin_name] = result

                    plugin._last_analysis_time = datetime.now()

                except Exception as e:
                    logger.error(f"プラグイン分析エラー {plugin_name}: {e}")
                    # エラー結果作成
                    results[plugin_name] = RiskAnalysisResult(
                        plugin_name=plugin_name,
                        timestamp=datetime.now(),
                        risk_level=RiskLevel.MODERATE,
                        risk_score=0.5,
                        confidence=0.0,
                        message=f"分析エラー: {str(e)}",
                        details={},
                        recommendations=[],
                        alerts=[f"プラグインエラー: {str(e)}"],
                        metadata={"error": True},
                    )

        return results

    def get_plugin_status(self) -> Dict[str, Dict[str, Any]]:
        """全プラグイン状態取得"""
        status = {}

        for plugin_name, plugin in self._plugins.items():
            info = self._plugin_infos[plugin_name]
            status[plugin_name] = {
                "info": info,
                "status": plugin.get_status(),
                "enabled": info.enabled,
            }

        return status

    def cleanup(self):
        """クリーンアップ"""
        # ホットリロード停止
        if self._observer and WATCHDOG_AVAILABLE:
            try:
                self._observer.stop()
                self._observer.join()
            except Exception as e:
                logger.error(f"ホットリロード停止エラー: {e}")

        # プラグインクリーンアップ
        for plugin in self._plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"プラグインクリーンアップエラー: {e}")

        logger.info("プラグインマネージャークリーンアップ完了")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
