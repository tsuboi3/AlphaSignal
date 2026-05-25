# AlphaSignal — プロジェクト管理ドキュメント

> **⚠️ AI引き継ぎ規則**
> - **再開時**: このファイルを最初に読み込み、「現在のバージョン」と「変更履歴」を確認してから作業を開始すること。
> - **終了時**: 変更した内容を必ず「変更履歴」セクションに追記してからセッションを終了すること。
> - このドキュメントはプロジェクトの唯一の信頼できる情報源（Single Source of Truth）である。

---

## 1. プロジェクト概要

| 項目 | 内容 |
|------|------|
| **プロジェクト名** | AlphaSignal |
| **バージョン** | v1.0.0 |
| **最終更新** | 2026-05-25 |
| **目的** | LightGBM + Transformer + ニュースセンチメント分析を組み合わせた株価予測システム |
| **主要言語** | Python 3.12+ |
| **メインエントリ** | `alphasignal.py` |

### プロジェクトの目的

金融市場における株価動向を機械学習で予測するシステムを構築する。
従来のテクニカル指標・価格データに加えて**ニュースセンチメント指数（NIS）**を組み込むことで、
投資家心理・市場の非効率性を特徴量として活用し、予測精度の向上を目指す。

---

## 2. ディレクトリ構成

```
stock_prediction_project/
│
├── alphasignal.py              # 【メインエントリ】対話式メニュー + 予測パイプライン
├── main.py                     # レガシー CLI（互換性維持のため残存）
├── AlphaSignal.md              # 【本ファイル】プロジェクト管理・AI引き継ぎ文書
├── README.md                   # ユーザー向けセットアップ・使い方ガイド
├── requirements.txt            # pip 依存ライブラリ一覧
│
├── src/
│   ├── __init__.py
│   ├── menu.py                 # 対話式メニューUI（企業名解決・期間選択）
│   ├── features.py             # 特徴量エンジニアリング（テクニカル指標・前処理）
│   ├── news_sentiment.py       # ニュースセンチメント取得・NIS計算
│   ├── lgbm_model.py           # LightGBM モデル定義・学習
│   ├── transformer_model.py    # Transformer モデル定義・学習（Keras）
│   └── ensemble.py             # スタッキングアンサンブル・評価・バックテスト
│
├── data/
│   └── {TICKER}_dummy.csv      # オフライン用ダミー株価データ（テスト用）
│
└── results/
    ├── {TICKER}_results.json   # 評価指標サマリー（JSON）
    └── {TICKER}_predictions.csv # 日次予測値（CSV）
```

---

## 3. アーキテクチャ

### 3.1 予測パイプライン全体像

```
[入力]
  企業名 / ティッカー / 期間
       ↓
[データ取得]
  株価データ (Yahoo Finance / CSVフォールバック)
  ニュースデータ (API / モックアップ)
       ↓
[特徴量エンジニアリング]
  価格指標・テクニカル指標・時間特徴量
  ニュースインパクト指数 (NIS)
       ↓
[モデル学習 (80/20 時系列分割)]
  ┌─────────────┐   ┌──────────────────┐
  │  LightGBM   │   │   Transformer    │
  │（2D 入力）  │   │（3D シーケンス） │
  └──────┬──────┘   └────────┬─────────┘
         └─────────┬──────────┘
                   ↓
        [スタッキングアンサンブル]
         線形回帰メタモデル
         入力: [LightGBM予測, Transformer予測, NIS]
                   ↓
[評価・出力]
  RMSE / MAE / MAPE / 方向性精度 / 相関係数
  バックテスト（シャープレシオ・最大ドローダウン）
  results/{TICKER}_results.json
  results/{TICKER}_predictions.csv
```

### 3.2 特徴量一覧（`src/features.py` `FEATURE_COLS`）

| カテゴリ | 特徴量 |
|----------|--------|
| 価格・出来高 | Open, High, Low, Close, Volume |
| 価格比率 | Open_Close, High_Low, High_Close, Low_Close, Open_High |
| リターン | Daily_Return, Log_Return |
| 移動平均 | SMA_5/10/20, EMA_5/10/20 |
| テクニカル指標 | RSI, MACD, Signal_Line, MACD_Hist |
| ボリンジャーバンド | 20_SMA, 20_STD, Upper_Band, Lower_Band, BB_Width |
| ボリューム | Volume_SMA_5/10, Volume_Ratio |
| 時間特徴量 | DayOfWeek, DayOfMonth, Month, Year |
| センチメント | news_impact_score, news_impact_ma, news_volume, news_volume_ma |

### 3.3 ニュースインパクト指数（NIS）の計算式

```
NIS_t = Σ (w_i,t × S_i,t × D(days_ago_i,t))

  w_i,t       : ボリューム加重 × 関連性加重
  S_i,t       : FinBERTスコア (positive確率 - negative確率)
  D(days_ago) : exp(-decay_rate × days_ago)  指数関数的時間減衰
```

---

## 4. 開発ルール

### 4.1 コーディング規約

1. **言語**: Python 3.12+。型ヒント（Type Hints）を原則すべての関数に付与する。
2. **文字コード**: UTF-8。日本語コメント・文字列を積極的に使用してよい。
3. **モジュール分割**: 機能ごとに `src/` 配下のファイルに分離する。`alphasignal.py` はエントリポイントのみとし、ロジックを直書きしない。
4. **docstring**: 全関数に Google スタイルの docstring を記載する。
5. **定数**: マジックナンバーは定数（大文字スネークケース）として定義する。
6. **エラー処理**: ユーザー入力・外部API呼び出しには必ず try/except を付ける。エラーメッセージは日本語で `⚠` プレフィックスを付ける。
7. **ログ出力**: `print()` を使用。本番移行時は `logging` モジュールへ切り替える。

### 4.2 データ・モデル規約

1. **データ分割**: 時系列データのため、**シャッフルなし**の時系列分割（訓練80% / テスト20%）を徹底する。
2. **スケーリング**: `MinMaxScaler` を使用。fit は訓練データのみ、transform はテストデータに適用。
3. **ターゲット変数**: 翌営業日の終値（`Close.shift(-1)`）を予測対象とする。
4. **ダミーデータ**: `data/{TICKER}_dummy.csv` にフォールバック機能を維持すること（オフライン・テスト用途）。

### 4.3 UI/メニュー規約

1. **対話式メニュー**は `src/menu.py` に集約する。`alphasignal.py` から呼び出す構造を維持する。
2. 企業名解決辞書 `COMPANY_ALIASES` は `src/menu.py` 内で管理する。新規追加時はここに追記する。
3. メニュー画面の罫線・スタイルは既存デザイン（`╔╗╚╝║`, `┌┐└┘│`）を踏襲する。
4. すべての入力にデフォルト値（`[デフォルト値]`形式）を提示する。

### 4.4 結果出力規約

1. JSON 結果: `results/{TICKER}_results.json`（`NumpyEncoder` で float32 を変換）
2. CSV 予測値: `results/{TICKER}_predictions.csv`
3. ティッカー名に `^` や `/` が含まれる場合はファイル名をサニタイズする（`safe_ticker`）。

### 4.5 禁止事項

- `main.py` への新機能追加（レガシー互換のみ維持）
- 訓練データにテストデータを混入させるリーク処理
- API キーをコード内にハードコーディング
- `results/` ディレクトリ以外への出力ファイル生成

---

## 5. 実行方法

```bash
# セットアップ
pip install -r requirements.txt

# 【推奨】対話式メニュー
python alphasignal.py

# CLIモード（スクリプト・自動化用）
python alphasignal.py --cli --ticker AAPL --start 2020-01-01 --end 2023-12-31
python alphasignal.py --cli --ticker トヨタ --start 2019-01-01 --end 2024-01-01
python alphasignal.py --cli --ticker NVDA --epochs 100 --seq_len 60

# FinBERT 実運用時（オプション）
pip install transformers torch
# src/news_sentiment.py の get_finbert_sentiment() コメントアウトを解除
```

---

## 6. 今後の拡張予定（Roadmap）

| 優先度 | 機能 | 対象ファイル |
|--------|------|-------------|
| 高 | FinBERT 実装を有効化（transformers 導入） | `src/news_sentiment.py` |
| 高 | 実ニュース API 連携（Alpha Vantage / News API） | `src/news_sentiment.py` |
| 中 | 予測結果のグラフ可視化（matplotlib） | `src/visualize.py`（新規） |
| 中 | 複数銘柄の一括バッチ実行 | `alphasignal.py` |
| 中 | モデルの保存・ロード機能（joblib / keras save） | `src/lgbm_model.py`, `src/transformer_model.py` |
| 低 | ソーシャルメディア（Twitter/X, Reddit）センチメント | `src/news_sentiment.py` |
| 低 | リアルタイム予測モード | `alphasignal.py` |
| 低 | Web UI 化（Streamlit / Gradio） | `app.py`（新規） |

---

## 7. 変更履歴

> **⚠️ AI作業者へ**: セッション終了前に必ずここに追記すること。
> フォーマット: `| YYYY-MM-DD | バージョン | 変更者 | 変更内容 |`

| 日付 | バージョン | 変更者 | 変更内容 |
|------|-----------|--------|---------|
| 2026-05-25 | v1.0.0 | Claude Sonnet 4.6 | 初期リリース。LightGBM + Transformer + ニュースセンチメントの基本パイプライン実装。`main.py` 作成。 |
| 2026-05-25 | v1.1.0 | Claude Sonnet 4.6 | プロジェクト名を AlphaSignal に変更。`alphasignal.py`（メインエントリ）と `src/menu.py`（対話式メニュー）を新規作成。企業名→ティッカー自動変換辞書（日英約40社）追加。実行後メニュー（再実行・終了）追加。`AlphaSignal.md`（本ファイル）作成。 |
| 2026-05-25 | v1.1.1 | Gemini CLI | AVX非対応環境でのIllegal instructionエラー回避のため、TensorFlowからPyTorchへモデル実装を移行。`main.py`および`alphasignal.py`の動作を確認。 |
| 2026-05-25 | v1.2.0 | Gemini CLI | SQLite データベース (`alphasignal.db`) を統合。データの永続化、欠損値チェック、重複防止機能を実装。`src/database.py` を新規作成し、各モジュールと連携。 |

---

## 8. 既知の問題・制限事項

| # | 内容 | 回避策 |
|---|------|--------|
| 1 | Yahoo Finance がネットワーク制限環境で 403 エラー | `data/{TICKER}_dummy.csv` にフォールバック |
| 2 | ニュースセンチメントがダミーデータ（ランダム値） | FinBERT 実装有効化 or 実 API 連携が必要（Roadmap 参照） |
| 3 | Transformer の学習時間が長い（CPU環境で数分） | `--epochs` を下げる。GPU環境では大幅短縮 |
| 4 | `^GSPC` など特殊文字を含むティッカーのファイル名 | `safe_ticker` でサニタイズ済み |

---

*このドキュメントは AlphaSignal プロジェクトの Single Source of Truth です。*
*再開時は必ず読み込み、終了時は変更履歴を更新してください。*
