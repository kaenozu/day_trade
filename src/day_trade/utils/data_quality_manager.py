# 品質トレンド集計
        all_scores = []
        for key in self.quality_history:
            if self.quality_history[key]:
                latest_metric = self.quality_history[key][-1]
                all_scores.append(latest_metric.overall_score)

                # 品質レベル分布更新
                level = latest_metric.quality_level.value
                report["overall_statistics"]["quality_distribution"][level] += 1

        # 全体統計
        if all_scores:
            report["overall_statistics"]["total_assessments"] = len(all_scores)
            report["overall_statistics"]["average_quality_score"] = np.mean(all_scores)

        # 個別トレンド（上位10件）
        for key in list(self.quality_history.keys())[:10]:
            symbol, data_type = key.split("_", 1)
            report["quality_trends"][key] = self.get_quality_trend(symbol, data_type)

        return report