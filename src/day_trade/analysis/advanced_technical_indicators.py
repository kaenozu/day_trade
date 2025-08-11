    def calculate_advanced_bollinger_bands(
        self,
        data: pd.DataFrame,
        periods: Optional[List[int]] = None,
        std_devs: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        高度ボリンジャーバンド分析

        Args:
            data: OHLCV価格データ
            periods: SMA期間のリスト (デフォルト: [20, 50])
            std_devs: 標準偏差の倍数リスト (デフォルト: [1.0, 2.0, 3.0])

        Returns:
            計算済みデータフレーム (bb_upper, bb_lower, bb_width, bb_percent_b, bb_squeeze)
        """
        try:
            df = data.copy()

            if periods is None:
                periods = [20, 50]
            if std_devs is None:
                std_devs = [1.0, 2.0, 3.0]

            for period in periods:
                sma = df["Close"].rolling(window=period).mean()
                std = df["Close"].rolling(window=period).std()
                df[f"bb_upper_{period}"] = sma + (
                    std * std_devs[-1]
                )  # 最大標準偏差を使用
                df[f"bb_lower_{period}"] = sma - (std * std_devs[-1])
                df[f"bb_width_{period}"] = (
                    df["bb_upper_{period}"] - df["bb_lower_{period}"]
                ) / sma
                df[f"bb_percent_b_{period}"] = (
                    df["Close"] - df["bb_lower_{period}"]
                ) / (df["bb_upper_{period}"] - df["bb_lower_{period}"])

            # スクイーズ検出
            df["bb_squeeze"] = (
                df["bb_upper_20"] - df["bb_lower_20"]
                < df["bb_upper_50"] - df["bb_lower_50"]
            )  # 20期間が50期間より狭い

            # ボリンジャーバンド戦略シグナル生成
            df["bb_signal"] = 0
            for i in range(1, len(df)):
                # ボラティリティブレイクアウト
                if (
                    df["bb_percent_b_20"].iloc[i] > 1
                    and df["bb_percent_b_20"].iloc[i - 1] < 1
                ):  # 上抜け
                    df["bb_signal"].iloc[i] = 1  # 買いシグナル
                elif (
                    df["bb_percent_b_20"].iloc[i] < 0
                    and df["bb_percent_b_20"].iloc[i - 1] > 0
                ):  # 下抜け
                    df["bb_signal"].iloc[i] = -1  # 売りシグナル

                # スクイーズブレイクアウト
                if "bb_squeeze" in df.columns:
                    if df["bb_squeeze"].iloc[i - 1] and not df["bb_squeeze"].iloc[i]:
                        # ブレイクアウト方向を判定
                        if df["Close"].iloc[i] > df["Close"].iloc[i - 5 : i].mean():
                            pass  # breakout_buy のロジック
                        else:
                            pass  # breakout_sell のロジック
                        # シグナルを付与 (例: 2 for buy, -2 for sell)
                        # df["bb_signal"].iloc[i] = 2 if signal == "breakout_buy" else -2

            # 不要な中間列を削除
            cols_to_drop = [
                col
                for col in df.columns
                if "SMA" in col
                or "StdDev" in col
                or "EMA_Fast" in col
                or "EMA_Slow" in col
            ]
            df = df.drop(columns=cols_to_drop, errors="ignore")

            logger.info("高度ボリンジャーバンド分析完了")
            return df

        except Exception as e:
            logger.error(f"高度ボリンジャーバンド分析エラー: {e}")
            return data