"""
ウォッチリスト機能のデモンストレーション
使用方法とサンプル出力を表示する
"""

from src.day_trade.core.watchlist import AlertCondition, AlertType, WatchlistManager


def main():
    print("=== ウォッチリスト機能デモ ===\n")

    try:
        # ステップ1: ウォッチリストマネージャーの初期化
        print("1. ウォッチリストマネージャーを初期化中...")
        manager = WatchlistManager()
        print("[OK] ウォッチリストマネージャーを初期化しました\n")

        # ステップ2: 銘柄の追加
        print("2. 銘柄をウォッチリストに追加中...")
        stocks_to_add = [
            ("7203", "自動車", "トヨタ自動車 - 主力株"),
            ("8306", "銀行", "三菱UFJ銀行 - 大手銀行"),
            ("9984", "テクノロジー", "ソフトバンクグループ - 投資会社"),
            ("6758", "テクノロジー", "ソニーグループ - エンタメ"),
            ("4689", "テクノロジー", "Zホールディングス - IT"),
        ]

        added_count = 0
        for code, group, memo in stocks_to_add:
            try:
                success = manager.add_stock(code, group, memo)
                if success:
                    print(f"   追加: {code} ({group}) - {memo}")
                    added_count += 1
                else:
                    print(f"   [重複] {code} は既に存在します")
            except Exception as e:
                print(f"   [エラー] {code}: {e}")

        print(f"[OK] {added_count}銘柄を追加しました\n")

        # ステップ3: ウォッチリスト一覧表示
        print("3. ウォッチリスト一覧を表示...")
        watchlist = manager.get_watchlist()

        if watchlist:
            print(f"[OK] {len(watchlist)}銘柄が登録されています")
            for item in watchlist:
                print(
                    f"   {item['stock_code']}: {item.get('stock_name', 'N/A')} "
                    f"({item['group_name']}) - {item.get('memo', '')}"
                )
        else:
            print("[INFO] ウォッチリストは空です")
        print()

        # ステップ4: グループ別表示
        print("4. グループ別ウォッチリスト...")
        groups = manager.get_groups()

        for group in groups:
            group_items = manager.get_watchlist(group)
            print(f"【{group}グループ】({len(group_items)}銘柄)")
            for item in group_items:
                print(f"   - {item['stock_code']}: {item.get('stock_name', 'N/A')}")
        print()

        # ステップ5: アラート設定
        print("5. アラート条件を設定中...")
        alert_conditions = [
            AlertCondition("7203", AlertType.PRICE_ABOVE, 3000.0, "価格上昇アラート"),
            AlertCondition("8306", AlertType.PRICE_BELOW, 700.0, "価格下落アラート"),
            AlertCondition(
                "9984", AlertType.CHANGE_PERCENT_UP, 5.0, "変化率上昇アラート"
            ),
            AlertCondition(
                "6758", AlertType.CHANGE_PERCENT_DOWN, -3.0, "変化率下落アラート"
            ),
            AlertCondition(
                "4689", AlertType.VOLUME_SPIKE, 5000000, "出来高急増アラート"
            ),
        ]

        alert_added = 0
        for condition in alert_conditions:
            try:
                success = manager.add_alert(condition)
                if success:
                    print(
                        f"   追加: {condition.stock_code} {condition.alert_type.value} "
                        f"{condition.threshold} - {condition.memo}"
                    )
                    alert_added += 1
                else:
                    print(f"   [重複] {condition.stock_code} のアラートは既に存在")
            except Exception as e:
                print(f"   [エラー] {condition.stock_code}: {e}")

        print(f"[OK] {alert_added}個のアラートを設定しました\n")

        # ステップ6: アラート一覧表示
        print("6. アラート一覧を表示...")
        alerts = manager.get_alerts()

        if alerts:
            print(f"[OK] {len(alerts)}個のアラートが設定されています")
            for alert in alerts:
                active_status = "ON" if alert["is_active"] else "OFF"
                last_triggered = (
                    alert["last_triggered"].strftime("%m-%d %H:%M")
                    if alert["last_triggered"]
                    else "未発動"
                )
                print(
                    f"   [{active_status}] {alert['stock_code']}: {alert['alert_type']} "
                    f"{alert['threshold']} (最終: {last_triggered})"
                )
        else:
            print("[INFO] アラートは設定されていません")
        print()

        # ステップ7: 価格情報付きウォッチリスト（実際のAPI呼び出しは制限）
        print("7. 価格情報付きウォッチリストを取得...")
        try:
            # 実際の価格取得はデモでは制限（API制限のため）
            print("[INFO] デモモードのため、実際の価格取得はスキップします")
            print("   実際の使用では以下のように価格情報を表示:")
            print("   7203: トヨタ自動車 2,650円 (+50円, +1.9%) 出来高: 1,234,567")
            print("   8306: 三菱UFJ銀行 780円 (-10円, -1.3%) 出来高: 9,876,543")
        except Exception as e:
            print(f"[INFO] 価格取得をスキップ: {e}")
        print()

        # ステップ8: アラートチェック（実際のチェックはスキップ）
        print("8. アラート条件チェック...")
        try:
            print("[INFO] デモモードのため、実際のアラートチェックはスキップします")
            print("   実際の使用では条件に合致した場合にアラート通知が生成されます")
        except Exception as e:
            print(f"[INFO] アラートチェックをスキップ: {e}")
        print()

        # ステップ9: サマリー情報
        print("9. ウォッチリストサマリー...")
        try:
            summary = manager.get_watchlist_summary()
            print("[OK] サマリー情報:")
            print(f"   総グループ数: {summary['total_groups']}グループ")
            print(f"   総銘柄数: {summary['total_stocks']}銘柄")
            print(f"   総アラート数: {summary['total_alerts']}個")
            print(f"   グループ: {', '.join(summary['groups'])}")
        except Exception as e:
            print(f"[エラー] サマリー取得失敗: {e}")
        print()

        # ステップ10: メモ更新のデモ
        print("10. メモ更新のデモ...")
        try:
            success = manager.update_memo(
                "7203", "自動車", "トヨタ自動車 - 更新されたメモ"
            )
            if success:
                print("[OK] トヨタ自動車のメモを更新しました")
            else:
                print("[WARNING] メモ更新に失敗しました")
        except Exception as e:
            print(f"[エラー] メモ更新エラー: {e}")
        print()

        # ステップ11: グループ移動のデモ
        print("11. グループ移動のデモ...")
        try:
            success = manager.move_to_group("6758", "テクノロジー", "お気に入り")
            if success:
                print("[OK] ソニーグループを「お気に入り」グループに移動しました")
            else:
                print("[WARNING] グループ移動に失敗しました")
        except Exception as e:
            print(f"[エラー] グループ移動エラー: {e}")
        print()

        # ステップ12: CSVエクスポート（デモ）
        print("12. CSVエクスポートのデモ...")
        try:
            # 実際のエクスポートはスキップ（パンダス依存）
            print("[INFO] 実際の使用では以下のようにCSVエクスポートできます:")
            print("   manager.export_watchlist_to_csv('my_watchlist.csv')")
            print("   ファイルには銘柄コード、名前、グループ、価格情報などが含まれます")
        except Exception as e:
            print(f"[INFO] CSVエクスポートをスキップ: {e}")
        print()

        print("=== ウォッチリスト機能の基本操作 ===")
        print(
            """
主な機能:
✓ 銘柄の追加・削除
✓ グループ管理
✓ アラート設定（価格、変化率、出来高）
✓ 価格情報付きリスト表示
✓ アラート条件チェック
✓ CSVエクスポート
✓ メモ管理
✓ グループ間移動

使用例:
```python
from src.day_trade.core.watchlist import WatchlistManager, AlertType, AlertCondition

manager = WatchlistManager()

# 銘柄追加
manager.add_stock("7203", "自動車", "トヨタ自動車のメモ")

# アラート設定
condition = AlertCondition("7203", AlertType.PRICE_ABOVE, 3000.0, "価格上昇")
manager.add_alert(condition)

# アラートチェック
notifications = manager.check_alerts()
for notification in notifications:
    print(f"アラート: {notification.stock_name} {notification.alert_type}")
```
        """
        )

    except Exception as e:
        print(f"[エラー] デモ実行中にエラーが発生しました: {e}")

    print("\n=== ウォッチリスト機能デモ完了 ===")


if __name__ == "__main__":
    main()
