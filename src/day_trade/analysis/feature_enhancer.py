#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Enhancer - 特徴量拡充モジュール
Issue #939対応: ニュースセンチメント分析
"""

from typing import Dict, List
from default_api import google_web_search

class NewsSentimentAnalyzer:
    """ニュースセンチメント分析器"""

    def __init__(self):
        # 簡単なキーワードベースのセンチメント辞書
        self.positive_keywords = ['上昇', '好調', '期待', '最高', '利益', '拡大', '突破', '提携', '革新']
        self.negative_keywords = ['下落', '不振', '懸念', '最低', '損失', '縮小', '懸念', '独禁法', '問題']

    def get_sentiment_for_symbol(self, symbol: str, company_name: str) -> Dict[str, float]:
        """指定された銘柄のニュースセンチメントスコアを取得"""
        print(f"Fetching news for {company_name} ({symbol})...")
        try:
            # Web検索で関連ニュースを取得
            search_results = google_web_search.search(queries=[f"{company_name} 株価 ニュース"])

            if not search_results or not search_results[0].get('results'):
                print("No news found.")
                return {'sentiment_score': 0.0, 'news_count': 0}

            sentiment_score = 0
            news_count = len(search_results[0]['results'])

            for result in search_results[0]['results']:
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                text_to_analyze = (title + " " + snippet).lower()

                # ポジティブ/ネガティブキーワードをカウント
                for p_word in self.positive_keywords:
                    sentiment_score += text_to_analyze.count(p_word)
                for n_word in self.negative_keywords:
                    sentiment_score -= text_to_analyze.count(n_word)

            # 記事数で正規化（単純化）
            normalized_score = sentiment_score / news_count if news_count > 0 else 0

            print(f"Found {news_count} articles, sentiment score: {normalized_score}")
            return {'sentiment_score': normalized_score, 'news_count': news_count}

        except Exception as e:
            print(f"Error fetching or analyzing news for {symbol}: {e}")
            return {'sentiment_score': 0.0, 'news_count': 0}
