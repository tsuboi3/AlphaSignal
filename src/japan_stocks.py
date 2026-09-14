"""
日本株キャピタルゲイン注目100銘柄の管理・検索・選択モジュール
Googleスプレッドシート（japan_stocks_capital_gain_100）から抽出された100銘柄の
セクター別分類、エイリアス解決、対話式選択メニューを提供します。
"""

import os
import re
import json
from typing import List, Dict, Any, Optional

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "japan_stocks_100.json")


def load_japan_stocks(data_path: str = DATA_PATH) -> List[Dict[str, Any]]:
    """100銘柄のマスターデータをロード"""
    if not os.path.exists(data_path):
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sectors(stocks: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """一意なセクター一覧を順序を維持して取得"""
    if stocks is None:
        stocks = load_japan_stocks()
    sectors = []
    seen = set()
    for s in stocks:
        sec = s.get("sector")
        if sec and sec not in seen:
            seen.add(sec)
            sectors.append(sec)
    return sectors


def get_stocks_by_sector(sector: str, stocks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """指定セクターの銘柄リストを取得"""
    if stocks is None:
        stocks = load_japan_stocks()
    return [s for s in stocks if s.get("sector") == sector]


def generate_japan_stock_aliases(stocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, str]:
    """
    100銘柄データから表記揺れを吸収するエイリアス辞書を自動生成
    例: '8035' -> '8035.T', '東京エレクトロン' -> '8035.T'
        'コマツ' -> '6301.T', '小松製作所' -> '6301.T'
    """
    if stocks is None:
        stocks = load_japan_stocks()

    aliases: Dict[str, str] = {}

    for item in stocks:
        code = item.get("code", "").strip()
        ticker = item.get("ticker", f"{code}.T").strip()
        name = item.get("name", "").strip()

        if not code or not ticker:
            continue

        # 1. 証券コード単体 (例: "8035" -> "8035.T")
        aliases[code.lower()] = ticker

        # 2. 正式名称 (例: "東京エレクトロン" -> "8035.T")
        if name:
            aliases[name.lower()] = ticker

            # 3. 括弧付きの名称を分解 (例: "小松製作所（コマツ）" -> "小松製作所", "コマツ")
            paren_match = re.match(r"^(.+?)[（\(](.+?)[）\)](.*)$", name)
            if paren_match:
                part1 = (paren_match.group(1) + paren_match.group(3)).strip().lower()
                part2 = paren_match.group(2).strip().lower()
                if part1:
                    aliases[part1] = ticker
                if part2:
                    aliases[part2] = ticker

            # 4. 「ホールディングス」「HD」「グループ」の省略形
            stripped_name = re.sub(r"(ホールディングス|HD|グループ)$", "", name, flags=re.IGNORECASE).strip().lower()
            if stripped_name and stripped_name != name.lower():
                aliases[stripped_name] = ticker

    return aliases


def search_stocks(query: str, stocks: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """銘柄名、証券コード、セクター、または注目理由の部分一致検索"""
    if stocks is None:
        stocks = load_japan_stocks()
    q = query.strip().lower()
    if not q:
        return stocks

    results = []
    for s in stocks:
        code = s.get("code", "").lower()
        name = s.get("name", "").lower()
        sector = s.get("sector", "").lower()
        focus = s.get("focus", "").lower()
        if q in code or q in name or q in sector or q in focus:
            results.append(s)
    return results


def interactive_select_japan_stock() -> Optional[str]:
    """
    日本株注目100銘柄からセクター別または検索で選択する対話式メニュー
    :return: 選択されたティッカーシンボル（例: '8035.T'）、またはキャンセル時 None
    """
    stocks = load_japan_stocks()
    if not stocks:
        print("  ⚠ 銘柄データが見つかりません。")
        return None

    sectors = get_sectors(stocks)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     日本株 キャピタルゲイン注目100銘柄 セレクター        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print("  セクターを選択するか、キーワード（銘柄名・コード）で検索できます。")
    print()

    while True:
        print("  【セクター一覧】")
        for i, sec in enumerate(sectors, 1):
            count = len(get_stocks_by_sector(sec, stocks))
            print(f"   [{i:2}] {sec:<24} ({count:2}銘柄)")
        print("   [ S] キーワード検索")
        print("   [ A] 全100銘柄一覧表示")
        print("   [ 0] 戻る / キャンセル")
        print()

        choice = input("  選択番号またはコマンドを入力: ").strip().lower()

        if choice in ("0", "q", "exit", "戻る"):
            return None

        target_list: List[Dict[str, Any]] = []

        if choice in ("s", "search", "検索"):
            q = input("  検索キーワード (コード/企業名/技術等): ").strip()
            if not q:
                continue
            target_list = search_stocks(q, stocks)
            print(f"\n  検索結果: {len(target_list)} 件")
        elif choice in ("a", "all", "全"):
            target_list = stocks
        elif choice.isdigit() and 1 <= int(choice) <= len(sectors):
            selected_sec = sectors[int(choice) - 1]
            target_list = get_stocks_by_sector(selected_sec, stocks)
            print(f"\n  ── 【{selected_sec}】 ({len(target_list)}銘柄) ──")
        else:
            print("  ⚠ 正しい選択肢を入力してください。\n")
            continue

        if not target_list:
            print("  該当する銘柄が見つかりませんでした。\n")
            continue

        # 銘柄一覧の表示
        print()
        print("  ┌────┬────────┬──────────────────────────┬──────────────────────────────────────────┐")
        print("  │ 番号 │ コード │ 銘柄名                   │ 注目・成長の視点                         │")
        print("  ├────┼────────┼──────────────────────────┼──────────────────────────────────────────┘")
        for idx, s in enumerate(target_list, 1):
            c = s.get("code", "")
            n = s.get("name", "")
            f = s.get("focus", "")
            if len(f) > 38:
                f = f[:37] + "…"
            print(f"  │ {idx:2} │ {c:<6} │ {n:<24} │ {f}")
        print("  └────┴────────┴──────────────────────────┴──────────────────────────────────────────┘")
        print("   [0] 一覧に戻る")
        print()

        sub_choice = input("  対象銘柄の番号を入力 [1-{len(target_list)}]: ").strip()
        if sub_choice.isdigit():
            sub_idx = int(sub_choice)
            if 1 <= sub_idx <= len(target_list):
                selected = target_list[sub_idx - 1]
                ticker = selected["ticker"]
                print(f"\n  ✓ 選択された銘柄: {selected['name']} ({ticker})")
                print(f"    注目理由: {selected.get('focus', '')}\n")
                return ticker
            elif sub_idx == 0:
                continue

        print("  一覧に戻ります。\n")
