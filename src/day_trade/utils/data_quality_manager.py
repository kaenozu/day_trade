logger.info(
            "データ品質管理システム初期化完了"
        )
        logger.info(f"  - 自動修正: {'有効' if auto_fix_enabled else '無効'}")
        logger.info(f"  - 品質閾値: {quality_threshold}")
        logger.info(f"  - キャッシュ: {'有効' if self.cache_enabled else '無効'}")