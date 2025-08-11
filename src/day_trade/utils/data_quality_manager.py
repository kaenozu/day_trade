        print(
            f"   評価済みデータセット: {quality_report['overall_statistics']['total_assessments']}"
        )
        print(
            f"   平均品質スコア: {quality_report['overall_statistics']['average_quality_score']:.3f}"
        )
        print(f"   バックフィルキュー: {quality_report['backfill_queue_size']}")

        print("
✅ データ品質管理システムテスト完了")