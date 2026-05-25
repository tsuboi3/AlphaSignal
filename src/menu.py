"""
AlphaSignal - 対話式メニューモジュール
企業名・ティッカー・期間・詳細設定をインタラクティブに入力する
"""

import os
import sys
import re
from datetime import datetime, date
from typing import Optional, Dict, Any


# ──────────────────────────────────────────────
# よく使う企業のティッカー辞書（日本語・英語対応）
# ──────────────────────────────────────────────
COMPANY_ALIASES: Dict[str, str] = {
    # 米国
    "apple": "AAPL", "アップル": "AAPL",
    "microsoft": "MSFT", "マイクロソフト": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "グーグル": "GOOGL",
    "amazon": "AMZN", "アマゾン": "AMZN",
    "meta": "META", "facebook": "META", "メタ": "META",
    "tesla": "TSLA", "テスラ": "TSLA",
    "nvidia": "NVDA", "エヌビディア": "NVDA",
    "netflix": "NFLX", "ネットフリックス": "NFLX",
    "disney": "DIS", "ディズニー": "DIS",
    "coca-cola": "KO", "coca cola": "KO", "コカコーラ": "KO",
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpモルガン": "JPM",
    "berkshire": "BRK-B", "バークシャー": "BRK-B",
    "sp500": "^GSPC", "s&p500": "^GSPC", "s&p 500": "^GSPC",
    "nasdaq": "^IXIC", "ナスダック": "^IXIC",
    "dow": "^DJI", "ダウ": "^DJI",
    # 日本
    "トヨタ": "7203.T", "toyota": "7203.T",
    "ソニー": "6758.T", "sony": "6758.T",
    "ソフトバンク": "9984.T", "softbank": "9984.T",
    "任天堂": "7974.T", "nintendo": "7974.T",
    "キーエンス": "6861.T", "keyence": "6861.T",
    "三菱ufj": "8306.T", "mufg": "8306.T",
    "日立": "6501.T", "hitachi": "6501.T",
    "パナソニック": "6752.T", "panasonic": "6752.T",
    "ntt": "9432.T", "エヌティーティー": "9432.T",
    "楽天": "4755.T", "rakuten": "4755.T",
}

# プリセット期間
PERIOD_PRESETS = [
    ("1年", "-1y"),
    ("3年", "-3y"),
    ("5年", "-5y"),
    ("10年", "-10y"),
    ("カスタム期間を入力", "custom"),
]

SEPARATOR = "─" * 62


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          AlphaSignal  株価予測システム  v1.0             ║")
    print("║   LightGBM + Transformer + ニュースセンチメント          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_separator(title: str = ""):
    if title:
        pad = (62 - len(title) - 2) // 2
        print(f"{'─' * pad} {title} {'─' * (62 - pad - len(title) - 2)}")
    else:
        print(SEPARATOR)


def input_with_prompt(prompt: str, default: str = "") -> str:
    """デフォルト値付きinput"""
    if default:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    else:
        val = input(f"  {prompt}: ").strip()
        return val


def resolve_ticker(user_input: str) -> Optional[str]:
    """
    企業名・ティッカーシンボルを正規化して返す。
    例: 'apple' → 'AAPL', 'トヨタ' → '7203.T', 'AAPL' → 'AAPL'
    """
    normalized = user_input.strip().lower()
    # 辞書引き
    if normalized in COMPANY_ALIASES:
        return COMPANY_ALIASES[normalized]
    # そのままティッカーとして使う（大文字化）
    ticker = user_input.strip().upper()
    return ticker


def validate_date(date_str: str) -> bool:
    """YYYY-MM-DD 形式チェック"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def calc_date_from_preset(preset: str):
    """'-3y' などのプリセットから (start, end) を返す"""
    today = date.today()
    end_str = today.strftime("%Y-%m-%d")
    m = re.match(r"-(\d+)y", preset)
    if m:
        years = int(m.group(1))
        start = date(today.year - years, today.month, today.day)
        return start.strftime("%Y-%m-%d"), end_str
    return None, None


def menu_select_company() -> str:
    """企業・ティッカー選択画面"""
    print_separator("STEP 1 / 4  ─  銘柄を選択")
    print()
    print("  企業名（日本語可）またはティッカーシンボルを入力してください。")
    print()
    print("  例: apple  /  アップル  /  AAPL")
    print("      トヨタ  /  toyota  /  7203.T")
    print("      nvidia  /  エヌビディア  /  NVDA")
    print()
    print("  ヒント: 企業名で入力するとティッカーに自動変換されます。")
    print()

    while True:
        raw = input_with_prompt("企業名 または ティッカーシンボル")
        if not raw:
            print("  ⚠  入力が空です。もう一度入力してください。\n")
            continue

        ticker = resolve_ticker(raw)
        alias_msg = ""
        if raw.lower() in COMPANY_ALIASES:
            alias_msg = f"  ✓ '{raw}' → ティッカー: {ticker}\n"
        else:
            alias_msg = f"  ✓ ティッカー: {ticker}\n"

        print()
        print(alias_msg)
        confirm = input("  この銘柄で進みますか？ [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes", "はい"):
            return ticker
        print()


def menu_select_period() -> tuple:
    """期間選択画面"""
    print_separator("STEP 2 / 4  ─  分析期間を選択")
    print()
    print("  ┌────┬──────────────────────────┐")
    for i, (label, _) in enumerate(PERIOD_PRESETS, 1):
        print(f"  │ {i:2} │  {label:<24}  │")
    print("  └────┴──────────────────────────┘")
    print()

    while True:
        raw = input_with_prompt("番号を選択", "3")
        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(PERIOD_PRESETS)):
                raise ValueError
        except ValueError:
            print(f"  ⚠  1〜{len(PERIOD_PRESETS)} の数字を入力してください。\n")
            continue

        label, preset = PERIOD_PRESETS[idx]

        if preset == "custom":
            print()
            print("  カスタム期間を入力してください（形式: YYYY-MM-DD）")
            while True:
                start = input_with_prompt("開始日", "2020-01-01")
                if validate_date(start):
                    break
                print("  ⚠  日付形式が正しくありません（例: 2020-01-01）\n")
            while True:
                end = input_with_prompt("終了日", date.today().strftime("%Y-%m-%d"))
                if validate_date(end):
                    if end > start:
                        break
                    print("  ⚠  終了日は開始日より後にしてください。\n")
                else:
                    print("  ⚠  日付形式が正しくありません。\n")
        else:
            start, end = calc_date_from_preset(preset)

        print()
        print(f"  ✓ 期間: {start}  ～  {end}  （{label}）")
        print()
        confirm = input("  この期間で進みますか？ [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes", "はい"):
            return start, end
        print()


def menu_advanced_settings(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """詳細設定画面（オプション）"""
    print_separator("STEP 3 / 4  ─  詳細設定（任意）")
    print()
    print("  Enterキーを押すとデフォルト値が使用されます。")
    print()

    settings = dict(defaults)

    # シーケンス長
    while True:
        raw = input_with_prompt("Transformer シーケンス長（推奨: 20〜60）",
                                str(settings['seq_len']))
        try:
            v = int(raw)
            if 5 <= v <= 120:
                settings['seq_len'] = v
                break
            print("  ⚠  5〜120 の範囲で入力してください。\n")
        except ValueError:
            print("  ⚠  整数を入力してください。\n")

    # エポック数
    while True:
        raw = input_with_prompt("Transformer 学習エポック数（推奨: 30〜100）",
                                str(settings['epochs']))
        try:
            v = int(raw)
            if 5 <= v <= 500:
                settings['epochs'] = v
                break
            print("  ⚠  5〜500 の範囲で入力してください。\n")
        except ValueError:
            print("  ⚠  整数を入力してください。\n")

    # 結果保存先
    raw = input_with_prompt("結果保存ディレクトリ", settings['save_dir'])
    settings['save_dir'] = raw if raw else settings['save_dir']

    print()
    return settings


def menu_confirm(ticker: str, start: str, end: str,
                 settings: Dict[str, Any]) -> bool:
    """実行前確認画面"""
    print_separator("STEP 4 / 4  ─  実行確認")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  銘柄         : {ticker:<38}│")
    print(f"  │  期間         : {start}  ～  {end}          │")
    print(f"  │  シーケンス長 : {settings['seq_len']:<38}│")
    print(f"  │  エポック数   : {settings['epochs']:<38}│")
    print(f"  │  保存先       : {settings['save_dir']:<38}│")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    ans = input("  上記の設定で予測を開始しますか？ [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes", "はい")


def menu_post_run(ticker: str, save_dir: str) -> str:
    """実行後メニュー"""
    print()
    print_separator("実行後メニュー")
    print()
    print("  ┌────┬──────────────────────────────┐")
    print("  │  1 │  別の銘柄・期間で再実行      │")
    print("  │  2 │  同じ設定で再実行            │")
    print("  │  3 │  終了                        │")
    print("  └────┴──────────────────────────────┘")
    print()
    while True:
        raw = input("  番号を選択 [1/2/3]: ").strip()
        if raw in ("1", "2", "3"):
            return raw
        print("  ⚠  1〜3 を入力してください。\n")


def run_interactive_menu() -> Optional[Dict[str, Any]]:
    """
    フルインタラクティブメニューを実行し、設定辞書を返す。
    ユーザーがキャンセルした場合は None を返す。
    """
    clear_screen()
    print_header()

    defaults = {
        'seq_len': 30,
        'epochs': 50,
        'save_dir': 'results',
    }

    while True:
        try:
            # STEP 1: 銘柄
            print()
            ticker = menu_select_company()
            print()

            # STEP 2: 期間
            start, end = menu_select_period()
            print()

            # STEP 3: 詳細設定
            use_adv = input(
                "  詳細設定を変更しますか？（デフォルトで問題なければ N） [y/N]: "
            ).strip().lower()
            print()
            if use_adv in ("y", "yes", "はい"):
                settings = menu_advanced_settings(defaults)
            else:
                settings = dict(defaults)
                print(f"  ✓ デフォルト設定を使用します。")
                print(f"    シーケンス長={settings['seq_len']}  "
                      f"エポック={settings['epochs']}  "
                      f"保存先={settings['save_dir']}")
            print()

            # STEP 4: 確認
            if not menu_confirm(ticker, start, end, settings):
                print()
                retry = input("  最初からやり直しますか？ [Y/n]: ").strip().lower()
                if retry not in ("", "y", "yes", "はい"):
                    print("\n  AlphaSignal を終了します。\n")
                    return None
                clear_screen()
                print_header()
                continue

            # 設定を返す
            return {
                'ticker':   ticker,
                'start':    start,
                'end':      end,
                'seq_len':  settings['seq_len'],
                'epochs':   settings['epochs'],
                'save_dir': settings['save_dir'],
            }

        except KeyboardInterrupt:
            print("\n\n  AlphaSignal を終了します。\n")
            return None
