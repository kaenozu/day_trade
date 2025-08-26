#!/usr/bin/env python3
"""
一目均衡表ユーティリティモジュール
Issue #315: 高度テクニカル指標・ML機能拡張

一目均衡表の計算・分析ヘルパー機能:
- TKクロス分析
- 遅行スパン信号分析
- 重み付き総合シグナル計算
"""

import pandas as pd


class IchimokuUtils:
    """一目均衡表ユーティリティクラス"""

    @staticmethod
    def analyze_tk_cross(
        tenkan_sen, kijun_sen, current_tenkan, current_kijun
    ) -> float:
        """TKクロス分析"""
        tk_cross_strength = 0

        if current_tenkan > current_kijun:
            if len(tenkan_sen) > 2:
                # 過去のクロス確認
                prev_tenkan = (
                    tenkan_sen.iloc[-2]
                    if not pd.isna(tenkan_sen.iloc[-2])
                    else current_tenkan
                )
                prev_kijun = (
                    kijun_sen.iloc[-2]
                    if not pd.isna(kijun_sen.iloc[-2])
                    else current_kijun
                )

                if prev_tenkan <= prev_kijun:
                    tk_cross_strength = 0.8  # 新しいクロス
                else:
                    tk_cross_strength = 0.5  # 継続中
            else:
                tk_cross_strength = 0.6
        elif current_tenkan < current_kijun:
            if len(tenkan_sen) > 2:
                prev_tenkan = (
                    tenkan_sen.iloc[-2]
                    if not pd.isna(tenkan_sen.iloc[-2])
                    else current_tenkan
                )
                prev_kijun = (
                    kijun_sen.iloc[-2]
                    if not pd.isna(kijun_sen.iloc[-2])
                    else current_kijun
                )

                if prev_tenkan >= prev_kijun:
                    tk_cross_strength = 0.8  # 新しいクロス
                else:
                    tk_cross_strength = 0.5  # 継続中
            else:
                tk_cross_strength = 0.6

        return tk_cross_strength

    @staticmethod
    def analyze_chikou_signal(current_chikou, current_price):
        """遅行スパン信号分析"""
        price_diff_ratio = abs(current_chikou - current_price) / current_price

        if current_chikou > current_price * 1.01:  # 1%以上の差
            chikou_signal = "bullish"
            chikou_signal_strength = min(0.8, price_diff_ratio * 20)
        elif current_chikou < current_price * 0.99:  # 1%以上の差
            chikou_signal = "bearish"
            chikou_signal_strength = min(0.8, price_diff_ratio * 20)
        else:
            chikou_signal = "neutral"
            chikou_signal_strength = 0.0

        return chikou_signal, chikou_signal_strength

    @staticmethod
    def calculate_weighted_ichimoku_signal(
        price_vs_cloud,
        tk_cross,
        tk_cross_strength,
        chikou_signal,
        chikou_signal_strength,
        cloud_color,
        cloud_thickness,
        current_price,
    ) -> float:
        """重み付き一目総合シグナル計算"""
        signal_weights = []

        # 価格vs雲 (重み 0.3)
        if price_vs_cloud == "above":
            signal_weights.append(0.3)
        elif price_vs_cloud == "below":
            signal_weights.append(-0.3)

        # TKクロス (重み 0.25)
        if tk_cross == "bullish":
            signal_weights.append(0.25 * tk_cross_strength)
        elif tk_cross == "bearish":
            signal_weights.append(-0.25 * tk_cross_strength)

        # 遅行スパン (重み 0.2)
        if chikou_signal == "bullish":
            signal_weights.append(0.2 * chikou_signal_strength)
        elif chikou_signal == "bearish":
            signal_weights.append(-0.2 * chikou_signal_strength)

        # 雲の色 (重み 0.15)
        if cloud_color == "bullish":
            signal_weights.append(0.15)
        elif cloud_color == "bearish":
            signal_weights.append(-0.15)

        # 雲の厚さ (重み 0.1)
        thickness_ratio = cloud_thickness / current_price
        if thickness_ratio > 0.02:  # 厚い雲は強いサポート/レジスタンス
            signal_weights.append(
                0.1 if price_vs_cloud == "above" else -0.1
            )

        return sum(signal_weights)

    @staticmethod
    def calculate_cloud_analysis(current_senkou_a, current_senkou_b, current_price):
        """雲分析"""
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        cloud_thickness = cloud_top - cloud_bottom
        cloud_color = (
            "bullish" if current_senkou_a > current_senkou_b else "bearish"
        )

        # 価格vs雲の詳細位置分析
        if current_price > cloud_top + (cloud_thickness * 0.1):
            price_vs_cloud = "above"
        elif current_price < cloud_bottom - (cloud_thickness * 0.1):
            price_vs_cloud = "below"
        else:
            price_vs_cloud = "in"

        return {
            "cloud_top": cloud_top,
            "cloud_bottom": cloud_bottom,
            "cloud_thickness": cloud_thickness,
            "cloud_color": cloud_color,
            "price_vs_cloud": price_vs_cloud,
        }

    @staticmethod
    def calculate_ichimoku_lines(high, low, close, tenkan_period, kijun_period, senkou_b_period):
        """一目均衡表ライン計算"""
        # 転換線の高速計算
        tenkan_high = high.rolling(
            tenkan_period, min_periods=tenkan_period // 2
        ).max()
        tenkan_low = low.rolling(
            tenkan_period, min_periods=tenkan_period // 2
        ).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # 基準線の高速計算
        kijun_high = high.rolling(
            kijun_period, min_periods=kijun_period // 2
        ).max()
        kijun_low = low.rolling(
            kijun_period, min_periods=kijun_period // 2
        ).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # 先行スパンA
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)

        # 先行スパンB
        senkou_b_high = high.rolling(
            senkou_b_period, min_periods=senkou_b_period // 2
        ).max()
        senkou_b_low = low.rolling(
            senkou_b_period, min_periods=senkou_b_period // 2
        ).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(
            kijun_period
        )

        # 遅行スパン
        chikou_span = close.shift(-kijun_period)

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }