# 構造化ロギング統合レポート

**生成日時**: 2025年08月03日 05:31:27

## 📊 概要

- **分析ファイル数**: 60
- **print文を含むファイル数**: 24
- **総print文数**: 305

## 📁 ファイル別詳細

### src\day_trade\analysis\backtest.py

**print文数**: 9

- **行 1004** (推奨レベル: info): `print("=== バックテスト結果 ===")`
- **行 1005** (推奨レベル: info): `print(f"期間: {result.start_date.date()} - {result.end_date.date()}")`
- **行 1006** (推奨レベル: info): `print(f"総リターン: {result.total_return:.2%}")`
- **行 1007** (推奨レベル: info): `print(f"年率リターン: {result.annualized_return:.2%}")`
- **行 1008** (推奨レベル: info): `print(f"ボラティリティ: {result.volatility:.2%}")`
- **行 1009** (推奨レベル: info): `print(f"シャープレシオ: {result.sharpe_ratio:.2f}")`
- **行 1010** (推奨レベル: info): `print(f"最大ドローダウン: {result.max_drawdown:.2%}")`
- **行 1011** (推奨レベル: info): `print(f"勝率: {result.win_rate:.1%}")`
- **行 1012** (推奨レベル: info): `print(f"総取引数: {result.total_trades}")`

### src\day_trade\analysis\comprehensive_ensemble_test.py

**print文数**: 1

- **行 1022** (推奨レベル: info): `print(detailed_report)`

### src\day_trade\analysis\ensemble.py

**print文数**: 13

- **行 724** (推奨レベル: info): `print(f"アンサンブルシグナル: {signal.signal_type.value.upper()}")`
- **行 725** (推奨レベル: info): `print(f"強度: {signal.strength.value}")`
- **行 726** (推奨レベル: info): `print(f"信頼度: {signal.confidence:.1f}%")`
- **行 727** (推奨レベル: info): `print(f"価格: {signal.price:.2f}")`
- **行 729** (推奨レベル: info): `print("\n戦略別貢献度:")`
- **行 731** (推奨レベル: info): `print(f"  {strategy_name}: {score:.2f}")`
- **行 733** (推奨レベル: info): `print("\n戦略重み:")`
- **行 735** (推奨レベル: info): `print(f"  {strategy_name}: {weight:.2f}")`
- **行 737** (推奨レベル: info): `print("\nメタ特徴量:")`
- **行 739** (推奨レベル: info): `print(f"  {feature}: {value}")`
- **行 741** (推奨レベル: info): `print("アンサンブルシグナルなし")`
- **行 744** (推奨レベル: info): `print("\n戦略サマリー:")`
- **行 747** (推奨レベル: info): `print(f"  {key}: {value}")`

### src\day_trade\analysis\ensemble_voting_validation.py

**print文数**: 1

- **行 872** (推奨レベル: info): `print(detailed_report)`

### src\day_trade\analysis\indicators.py

**print文数**: 9

- **行 633** (推奨レベル: info): `print("=== SMA（20日） ===")`
- **行 635** (推奨レベル: info): `print(sma20.tail())`
- **行 637** (推奨レベル: info): `print("\n=== RSI（14日） ===")`
- **行 639** (推奨レベル: info): `print(rsi.tail())`
- **行 641** (推奨レベル: info): `print("\n=== MACD ===")`
- **行 643** (推奨レベル: info): `print(macd.tail())`
- **行 645** (推奨レベル: info): `print("\n=== 全指標計算 ===")`
- **行 647** (推奨レベル: info): `print(all_indicators.columns.tolist())`
- **行 648** (推奨レベル: info): `print(f"計算完了: {len(all_indicators.columns)}列")`

### src\day_trade\analysis\ml_performance_benchmark.py

**print文数**: 1

- **行 736** (推奨レベル: info): `print(report)`

### src\day_trade\analysis\patterns.py

**print文数**: 11

- **行 443** (推奨レベル: info): `print("=== ゴールデン・デッドクロス ===")`
- **行 448** (推奨レベル: info): `print(f"ゴールデンクロス: {len(golden_dates)}回")`
- **行 451** (推奨レベル: info): `print(f"  {date.date()}: 信頼度 {confidence:.1f}%")`
- **行 453** (推奨レベル: info): `print(f"\nデッドクロス: {len(dead_dates)}回")`
- **行 456** (推奨レベル: info): `print(f"  {date.date()}: 信頼度 {confidence:.1f}%")`
- **行 458** (推奨レベル: info): `print("\n=== サポート・レジスタンス ===")`
- **行 460** (推奨レベル: info): `print(f"レジスタンス: {[f'{level:.2f}' for level in levels['resistance']]}")`
- **行 461** (推奨レベル: info): `print(f"サポート: {[f'{level:.2f}' for level in levels['support']]}")`
- **行 463** (推奨レベル: info): `print("\n=== 全パターン検出 ===")`
- **行 465** (推奨レベル: info): `print(f"総合信頼度: {all_patterns['overall_confidence']:.1f}%")`
- **行 469** (推奨レベル: info): `print(f"最新シグナル: {signal['type']} (信頼度: {signal['confidence']:.1f}%)")`

### src\day_trade\analysis\performance_integration_test.py

**print文数**: 1

- **行 746** (推奨レベル: info): `print(detailed_report)`

### src\day_trade\analysis\screener.py

**print文数**: 2

- **行 587** (推奨レベル: info): `print(create_screening_report(results))`
- **行 589** (推奨レベル: info): `print("条件を満たす銘柄が見つかりませんでした。")`

### src\day_trade\analysis\signals.py

**print文数**: 13

- **行 965** (推奨レベル: info): `print(f"シグナル: {signal.signal_type.value.upper()}")`
- **行 966** (推奨レベル: info): `print(f"強度: {signal.strength.value}")`
- **行 967** (推奨レベル: info): `print(f"信頼度: {signal.confidence:.1f}%")`
- **行 968** (推奨レベル: info): `print(f"価格: {float(signal.price):.2f}")`
- **行 969** (推奨レベル: info): `print(f"タイムスタンプ: {signal.timestamp}")`
- **行 970** (推奨レベル: info): `print("\n理由:")`
- **行 972** (推奨レベル: info): `print(f"  - {reason}")`
- **行 973** (推奨レベル: info): `print("\n条件の状態:")`
- **行 976** (推奨レベル: info): `print(f"  {status} {condition}")`
- **行 977** (推奨レベル: info): `print(f"\n有効性スコア: {validity:.1f}%")`
- **行 980** (推奨レベル: info): `print("シグナルなし")`
- **行 1003** (推奨レベル: info): `print("\n=== 時系列シグナル ===")`
- **行 1004** (推奨レベル: info): `print(active_signals.tail(10))`

### src\day_trade\analysis\test_ml_integration.py

**print文数**: 1

- **行 563** (推奨レベル: info): `print(report)`

### src\day_trade\cli\enhanced_interactive.py

**print文数**: 36

- **行 81** (推奨レベル: info): `console.print("[bold green]画面をクリアしました[/bold green]")`
- **行 204** (推奨レベル: info): `console.print(Panel(help_text, title="ヘルプ", border_style="cyan"))`
- **行 258** (推奨レベル: info): `console.print(`
- **行 264** (推奨レベル: error): `console.print(create_error_panel(f"コマンド実行エラー: {e}"))`
- **行 272** (推奨レベル: error): `console.print(create_error_panel(f"無効な銘柄コード: {code}"))`
- **行 278** (推奨レベル: error): `console.print(create_error_panel(f"銘柄コード正規化に失敗: {code}"))`
- **行 282** (推奨レベル: info): `console.print(f"[cyan]銘柄 {code} の情報を取得中...[/cyan]")`
- **行 287** (推奨レベル: info): `console.print(`
- **行 295** (推奨レベル: info): `console.print(`
- **行 299** (推奨レベル: error): `console.print(create_error_panel(f"銘柄 {code} の情報取得に失敗"))`
- **行 301** (推奨レベル: error): `console.print(create_error_panel(f"データ取得エラー: {e}"))`
- **行 305** (推奨レベル: info): `console.print(f"[cyan]銘柄 {code} のヒストリカルデータを取得中...[/cyan]")`
- **行 307** (推奨レベル: info): `console.print(create_info_panel(f"銘柄 {code} の過去データ（実装予定）"))`
- **行 311** (推奨レベル: info): `console.print(f"[cyan]{len(codes)} 銘柄を監視中...[/cyan]")`
- **行 313** (推奨レベル: info): `console.print(f"監視中: {code}")`
- **行 320** (推奨レベル: info): `console.print(`
- **行 328** (推奨レベル: info): `console.print(create_info_panel("ウォッチリスト一覧（実装予定）"))`
- **行 330** (推奨レベル: info): `console.print(`
- **行 334** (推奨レベル: info): `console.print(`
- **行 338** (推奨レベル: warning): `console.print(create_warning_panel(f"不明なサブコマンド: {subcommand}"))`
- **行 345** (推奨レベル: info): `console.print(create_info_panel("設定表示（実装予定）"))`
- **行 348** (推奨レベル: info): `console.print(create_success_panel(f"設定更新: {key} = {value}"))`
- **行 351** (推奨レベル: info): `console.print(create_success_panel("設定をリセットしました"))`
- **行 353** (推奨レベル: warning): `console.print(create_warning_panel("使用法: config [show|set|reset]"))`
- **行 360** (推奨レベル: info): `console.print(f"銘柄コード {code}: {status}")`
- **行 364** (推奨レベル: info): `console.print(create_info_panel("インタラクティブバックテスト（実装予定）"))`
- **行 376** (推奨レベル: info): `console.print(`
- **行 389** (推奨レベル: info): `console.print(`
- **行 400** (推奨レベル: info): `console.print(`
- **行 404** (推奨レベル: info): `console.print(`
- **行 409** (推奨レベル: error): `console.print(create_error_panel("スクリーニング機能が利用できません"))`
- **行 411** (推奨レベル: error): `console.print(create_error_panel(f"スクリーニングエラー: {e}"))`
- **行 415** (推奨レベル: info): `console.print(`
- **行 464** (推奨レベル: info): `console.print("[yellow]継続します...[/yellow]")`
- **行 469** (推奨レベル: error): `console.print(create_error_panel(f"予期しないエラー: {e}"))`
- **行 472** (推奨レベル: info): `console.print(`

### src\day_trade\cli\enhanced_main.py

**print文数**: 34

- **行 52** (推奨レベル: info): `console.print(create_success_panel("データベースを初期化しました。"))`
- **行 54** (推奨レベル: info): `console.print(`
- **行 71** (推奨レベル: info): `console.print(`
- **行 78** (推奨レベル: info): `console.print(`
- **行 89** (推奨レベル: info): `console.print(`
- **行 104** (推奨レベル: info): `console.print(`
- **行 114** (推奨レベル: info): `console.print(table)`
- **行 123** (推奨レベル: info): `console.print("\n")`
- **行 124** (推奨レベル: info): `console.print(detail_table)`
- **行 126** (推奨レベル: info): `console.print("\n")`
- **行 127** (推奨レベル: info): `console.print(`
- **行 144** (推奨レベル: info): `console.print(`
- **行 153** (推奨レベル: info): `console.print(`
- **行 162** (推奨レベル: info): `console.print(`
- **行 173** (推奨レベル: info): `console.print(`
- **行 187** (推奨レベル: info): `console.print(`
- **行 197** (推奨レベル: info): `console.print(table)`
- **行 200** (推奨レベル: info): `console.print("\n[bold]サマリー:[/bold]")`
- **行 201** (推奨レベル: info): `console.print(f"期間高値: ¥{df['High'].max():,.0f}")`
- **行 202** (推奨レベル: info): `console.print(f"期間安値: ¥{df['Low'].min():,.0f}")`
- **行 203** (推奨レベル: info): `console.print(f"平均出来高: {int(df['Volume'].mean()):,}")`
- **行 213** (推奨レベル: info): `console.print(`
- **行 227** (推奨レベル: info): `console.print(`
- **行 236** (推奨レベル: info): `console.print(table)`
- **行 263** (推奨レベル: info): `console.print(table)`
- **行 283** (推奨レベル: info): `console.print(`
- **行 287** (推奨レベル: info): `console.print(`
- **行 301** (推奨レベル: info): `console.print(create_success_panel("設定をデフォルトにリセットしました。"))`
- **行 303** (推奨レベル: info): `console.print(`
- **行 317** (推奨レベル: info): `console.print(create_success_panel(f"設定をエクスポートしました: {file_path}"))`
- **行 319** (推奨レベル: info): `console.print(`
- **行 333** (推奨レベル: info): `console.print(create_success_panel(f"設定をインポートしました: {file_path}"))`
- **行 335** (推奨レベル: info): `console.print(`
- **行 364** (推奨レベル: info): `console.print(table)`

### src\day_trade\cli\interactive.py

**print文数**: 76

- **行 101** (推奨レベル: info): `console.print(`
- **行 110** (推奨レベル: info): `console.print(table)`
- **行 118** (推奨レベル: info): `console.print("\n")`
- **行 119** (推奨レベル: info): `console.print(detail_table)`
- **行 121** (推奨レベル: info): `console.print("\n")`
- **行 122** (推奨レベル: info): `console.print(`
- **行 135** (推奨レベル: info): `console.print(`
- **行 143** (推奨レベル: info): `console.print(table)`
- **行 144** (推奨レベル: info): `console.print("\n[bold]サマリー:[/bold]")`
- **行 145** (推奨レベル: info): `console.print(f"期間高値: ¥{df['High'].max():,.0f}")`
- **行 146** (推奨レベル: info): `console.print(f"期間安値: ¥{df['Low'].min():,.0f}")`
- **行 147** (推奨レベル: info): `console.print(f"平均出来高: {int(df['Volume'].mean()):,}")`
- **行 152** (推奨レベル: info): `console.print(`
- **行 155** (推奨レベル: info): `console.print(`
- **行 158** (推奨レベル: info): `console.print("[dim]Ctrl+C で終了[/dim]\n")`
- **行 210** (推奨レベル: info): `console.print("\n[green]インタラクティブデモが完了しました！[/green]")`
- **行 212** (推奨レベル: info): `console.print("\n[yellow]デモを中断しました。[/yellow]")`
- **行 225** (推奨レベル: info): `console.print(create_success_panel("データベースを初期化しました。"))`
- **行 227** (推奨レベル: info): `console.print(`
- **行 244** (推奨レベル: info): `console.print(`
- **行 251** (推奨レベル: info): `console.print(`
- **行 262** (推奨レベル: info): `console.print(`
- **行 288** (推奨レベル: error): `console.print(create_error_panel(f"無効な銘柄コード: {code}"))`
- **行 294** (推奨レベル: info): `console.print(`
- **行 309** (推奨レベル: info): `console.print(`
- **行 316** (推奨レベル: info): `console.print(`
- **行 328** (推奨レベル: info): `console.print(`
- **行 341** (推奨レベル: info): `console.print(`
- **行 350** (推奨レベル: info): `console.print(table)`
- **行 368** (推奨レベル: info): `console.print(`
- **行 381** (推奨レベル: info): `console.print(`
- **行 386** (推奨レベル: warning): `console.print(create_warning_panel(f"{code} は既に追加されています。"))`
- **行 388** (推奨レベル: info): `console.print(`
- **行 395** (推奨レベル: info): `console.print(`
- **行 403** (推奨レベル: info): `console.print(create_success_panel(f"{added_count} 件の銘柄を追加しました。"))`
- **行 413** (推奨レベル: info): `console.print(`
- **行 426** (推奨レベル: info): `console.print(`
- **行 431** (推奨レベル: info): `console.print(`
- **行 435** (推奨レベル: info): `console.print(`
- **行 443** (推奨レベル: info): `console.print(create_success_panel(f"{removed_count} 件の銘柄を削除しました。"))`
- **行 454** (推奨レベル: info): `console.print(`
- **行 485** (推奨レベル: info): `console.print(table)`
- **行 496** (推奨レベル: info): `console.print(`
- **行 519** (推奨レベル: info): `console.print(create_success_panel(f"{code} のメモを更新しました。"))`
- **行 521** (推奨レベル: error): `console.print(create_error_panel(f"{code} はウォッチリストにありません。"))`
- **行 523** (推奨レベル: info): `console.print(`
- **行 539** (推奨レベル: info): `console.print(`
- **行 551** (推奨レベル: info): `console.print(`
- **行 555** (推奨レベル: error): `console.print(create_error_panel(f"{code} はウォッチリストにありません。"))`
- **行 557** (推奨レベル: info): `console.print(`
- **行 572** (推奨レベル: info): `console.print(create_success_panel("ウォッチリストを全てクリアしました。"))`
- **行 574** (推奨レベル: info): `console.print(`
- **行 606** (推奨レベル: info): `console.print(table)`
- **行 626** (推奨レベル: info): `console.print(`
- **行 630** (推奨レベル: info): `console.print(`
- **行 644** (推奨レベル: info): `console.print(create_success_panel("設定をデフォルトにリセットしました。"))`
- **行 646** (推奨レベル: info): `console.print(`
- **行 675** (推奨レベル: info): `console.print(table)`
- **行 689** (推奨レベル: info): `console.print(`
- **行 700** (推奨レベル: info): `console.print(`
- **行 713** (推奨レベル: info): `console.print(`
- **行 729** (推奨レベル: info): `console.print(`
- **行 735** (推奨レベル: info): `console.print("[dim]対話的なコマンド実行機能は開発中です...[/dim]")`
- **行 772** (推奨レベル: info): `console.print(f"[cyan]対象銘柄: {len(symbol_list)}銘柄[/cyan]")`
- **行 786** (推奨レベル: info): `console.print(`
- **行 834** (推奨レベル: info): `console.print(table)`
- **行 837** (推奨レベル: info): `console.print(`
- **行 843** (推奨レベル: info): `console.print("\n[bold]🏆 トップ3銘柄の詳細:[/bold]")`
- **行 846** (推奨レベル: info): `console.print(`
- **行 850** (推奨レベル: info): `console.print(`
- **行 854** (推奨レベル: info): `console.print(f"   20日平均出来高: {tech_data['volume_avg_20d']:,}")`
- **行 857** (推奨レベル: info): `console.print(`
- **行 864** (推奨レベル: error): `console.print(create_error_panel(f"スクリーニング実行エラー: {e}"))`
- **行 879** (推奨レベル: info): `self.console.print("対話型モードを開始します。")`
- **行 896** (推奨レベル: info): `self.console.print(f"コマンド '{command}' を実行しました。")`
- **行 901** (推奨レベル: info): `self.console.print("対話型モードを終了します。")`

### src\day_trade\cli\main.py

**print文数**: 30

- **行 53** (推奨レベル: info): `console.print(create_success_panel("データベースを初期化しました。"))`
- **行 55** (推奨レベル: info): `console.print(`
- **行 72** (推奨レベル: info): `console.print(`
- **行 79** (推奨レベル: info): `console.print(`
- **行 90** (推奨レベル: info): `console.print(`
- **行 105** (推奨レベル: info): `console.print(`
- **行 115** (推奨レベル: info): `console.print(table)`
- **行 124** (推奨レベル: info): `console.print("\n")`
- **行 125** (推奨レベル: info): `console.print(detail_table)`
- **行 127** (推奨レベル: info): `console.print("\n")`
- **行 128** (推奨レベル: info): `console.print(`
- **行 145** (推奨レベル: info): `console.print(`
- **行 154** (推奨レベル: info): `console.print(`
- **行 163** (推奨レベル: info): `console.print(`
- **行 174** (推奨レベル: info): `console.print(`
- **行 188** (推奨レベル: info): `console.print(`
- **行 198** (推奨レベル: info): `console.print(table)`
- **行 201** (推奨レベル: info): `console.print("\n[bold]サマリー:[/bold]")`
- **行 202** (推奨レベル: info): `console.print(f"期間高値: ¥{df['High'].max():,.0f}")`
- **行 203** (推奨レベル: info): `console.print(f"期間安値: ¥{df['Low'].min():,.0f}")`
- **行 204** (推奨レベル: info): `console.print(f"平均出来高: {int(df['Volume'].mean()):,}")`
- **行 214** (推奨レベル: info): `console.print(`
- **行 228** (推奨レベル: info): `console.print(`
- **行 237** (推奨レベル: info): `console.print(table)`
- **行 264** (推奨レベル: info): `console.print(table)`
- **行 284** (推奨レベル: info): `console.print(`
- **行 288** (推奨レベル: info): `console.print(`
- **行 302** (推奨レベル: info): `console.print(create_success_panel("設定をデフォルトにリセットしました。"))`
- **行 304** (推奨レベル: info): `console.print(`
- **行 333** (推奨レベル: info): `console.print(table)`

### src\day_trade\cli\watchlist_commands.py

**print文数**: 20

- **行 41** (推奨レベル: info): `console.print(`
- **行 60** (推奨レベル: info): `console.print(`
- **行 67** (推奨レベル: info): `console.print(`
- **行 85** (推奨レベル: info): `console.print(`
- **行 104** (推奨レベル: info): `console.print(`
- **行 111** (推奨レベル: info): `console.print(`
- **行 132** (推奨レベル: info): `console.print(`
- **行 177** (推奨レベル: info): `console.print(`
- **行 208** (推奨レベル: info): `console.print(table)`
- **行 218** (推奨レベル: info): `console.print(`
- **行 234** (推奨レベル: info): `console.print(table)`
- **行 246** (推奨レベル: info): `console.print(`
- **行 263** (推奨レベル: info): `console.print(create_success_panel(f"銘柄 {code} のメモを更新しました。"))`
- **行 265** (推奨レベル: info): `console.print(`
- **行 282** (推奨レベル: info): `console.print(`
- **行 295** (推奨レベル: info): `console.print(`
- **行 301** (推奨レベル: info): `console.print(`
- **行 318** (推奨レベル: info): `console.print("キャンセルしました。")`
- **行 323** (推奨レベル: info): `console.print("キャンセルしました。")`
- **行 333** (推奨レベル: info): `console.print(create_success_panel(f"{success_count}件の銘柄を削除しました。"))`

### src\day_trade\config\config_manager.py

**print文数**: 10

- **行 470** (推奨レベル: info): `print("=== 設定情報 ===")`
- **行 471** (推奨レベル: info): `print(f"監視銘柄数: {len(config_manager.get_symbol_codes())}")`
- **行 472** (推奨レベル: info): `print(f"銘柄コード: {config_manager.get_symbol_codes()}")`
- **行 473** (推奨レベル: info): `print(f"高優先度銘柄: {config_manager.get_high_priority_symbols()}")`
- **行 476** (推奨レベル: info): `print(f"現在市場オープン中: {config_manager.is_market_open()}")`
- **行 480** (推奨レベル: info): `print(f"テクニカル指標有効: {tech_settings.enabled}")`
- **行 483** (推奨レベル: info): `print(f"アラート有効: {alert_settings.enabled}")`
- **行 486** (推奨レベル: info): `print(f"レポート出力形式: {report_settings.formats}")`
- **行 488** (推奨レベル: info): `print("設定管理システムのテストが完了しました")`
- **行 491** (推奨レベル: error): `print(f"エラー: {e}")`

### src\day_trade\core\config.py

**print文数**: 1

- **行 72** (推奨レベル: error): `print(f"設定ファイル読み込みエラー: {e}")`

### src\day_trade\core\portfolio.py

**print文数**: 12

- **行 523** (推奨レベル: info): `print("=== ポートフォリオメトリクス ===")`
- **行 524** (推奨レベル: info): `print(f"総資産: {metrics.total_value:,}円")`
- **行 525** (推奨レベル: info): `print(f"総損益: {metrics.total_pnl:,}円 ({metrics.total_pnl_percent:.2f}%)")`
- **行 526** (推奨レベル: info): `print(`
- **行 531** (推奨レベル: info): `print(`
- **行 538** (推奨レベル: info): `print("\n=== セクター配分 ===")`
- **行 541** (推奨レベル: info): `print(f"{alloc.sector}: {alloc.percentage:.1f}% ({alloc.value:,}円)")`
- **行 544** (推奨レベル: info): `print("\n=== パフォーマンスランキング ===")`
- **行 546** (推奨レベル: info): `print("上位:")`
- **行 548** (推奨レベル: info): `print(f"  {symbol}: {pnl_pct:.2f}%")`
- **行 549** (推奨レベル: info): `print("下位:")`
- **行 551** (推奨レベル: info): `print(f"  {symbol}: {pnl_pct:.2f}%")`

### src\day_trade\core\trade_manager.py

**print文数**: 19

- **行 1553** (推奨レベル: info): `print("=== ポジション情報 ===")`
- **行 1554** (推奨レベル: info): `print(f"銘柄: {position.symbol}")`
- **行 1555** (推奨レベル: info): `print(f"数量: {position.quantity}株")`
- **行 1556** (推奨レベル: info): `print(f"平均単価: {position.average_price}円")`
- **行 1557** (推奨レベル: info): `print(f"総コスト: {position.total_cost}円")`
- **行 1558** (推奨レベル: info): `print(f"現在価格: {position.current_price}円")`
- **行 1559** (推奨レベル: info): `print(f"時価総額: {position.market_value}円")`
- **行 1560** (推奨レベル: info): `print(`
- **行 1572** (推奨レベル: info): `print("\n=== 実現損益 ===")`
- **行 1574** (推奨レベル: info): `print(f"銘柄: {pnl.symbol}")`
- **行 1575** (推奨レベル: info): `print(f"数量: {pnl.quantity}株")`
- **行 1576** (推奨レベル: info): `print(f"買値: {pnl.buy_price}円")`
- **行 1577** (推奨レベル: info): `print(f"売値: {pnl.sell_price}円")`
- **行 1578** (推奨レベル: info): `print(f"損益: {pnl.pnl}円 ({pnl.pnl_percent}%)")`
- **行 1582** (推奨レベル: info): `print("\n=== ポートフォリオサマリー ===")`
- **行 1584** (推奨レベル: info): `print(f"{key}: {value}")`
- **行 1587** (推奨レベル: info): `print("\n=== CSV出力例 ===")`
- **行 1592** (推奨レベル: info): `print("CSV出力完了")`
- **行 1594** (推奨レベル: error): `print(f"CSV出力エラー: {e}")`

### src\day_trade\models\test_database_optimization.py

**print文数**: 1

- **行 665** (推奨レベル: info): `print(detailed_report)`

### src\day_trade\utils\enhanced_error_handler.py

**print文数**: 1

- **行 217** (推奨レベル: error): `console.print(error_panel)`

### src\day_trade\utils\friendly_error_handler.py

**print文数**: 1

- **行 333** (推奨レベル: error): `console.print(error_panel)`

### src\day_trade\utils\progress.py

**print文数**: 2

- **行 288** (推奨レベル: info): `console.print(Panel("処理結果がありません", title=title))`
- **行 308** (推奨レベル: info): `console.print(table)`


## 🛠️ 推奨アクション

### 1. 自動変換の実行

```bash
# ドライラン（変更は行わず、プレビューのみ）
python scripts/logging_integration.py convert --dry-run

# 実際の変換実行
python scripts/logging_integration.py convert
```

### 2. 手動での対応が必要なケース

- 複雑なprint文
- デバッグ用の一時的なprint文
- ライブラリ固有の出力形式

### 3. ロギング設定の確認

```python
from day_trade.utils.logging_config import setup_logging
setup_logging()
```

### 4. 環境変数の設定

```bash
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export ENVIRONMENT=production
```

## 📝 ベストプラクティス

1. **適切なログレベルの使用**
   - `debug`: 開発時のデバッグ情報
   - `info`: 一般的な情報
   - `warning`: 警告レベルの問題
   - `error`: エラー情報

2. **構造化データの活用**
   ```python
   logger.info("取引実行", symbol="AAPL", quantity=100, price=150.25)
   ```

3. **コンテキスト情報の追加**
   ```python
   logger = get_context_logger(user_id="user123", session_id="sess456")
   logger.info("ユーザーアクション", action="buy_stock")
   ```

---
*このレポートは logging_integration.py により自動生成されました。*
