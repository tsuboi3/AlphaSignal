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
| **バージョン** | v1.2.0 |
| **最終更新** | 2026-05-25 |
| **目的** | LightGBM + Transformer + ニュースセンチメント分析を組み合わせた株価予測システム |
| **主要言語** | Python 3.12+ (PyTorch版) |
| **メインエントリ** | `alphasignal.py` |

### プロジェクトの目的

金融市場における株価動向を機械学習で予測するシステムを構築する。
従来のテクニカル指標・価格データに加えて**ニュースセンチメント指数（NIS）**を組み込むことで、
投資家心理・市場の非効率性を特徴量として活用し、予測精度の向上を目指す。
※ AVX非対応環境への対応のため、TransformerモデルはPyTorchで実装されている。
※ データの永続化と整合性維持のため、SQLite データベースが統合されている。

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
├── alphasignal.db              # SQLite データベース（永続化データ）
├── .gitignore                  # Git 除外設定
│
├── src/
│   ├── __init__.py
│   ├── database.py             # SQLite データベース管理（永続化・重複排除）
│   ├── menu.py                 # 対話式メニューUI（企業名解決・期間選択）
│   ├── features.py             # 特徴量エンジニアリング（DB連携・前処理）
│   ├── news_sentiment.py       # ニュースセンチメント取得・NIS計算（DB連携）
│   ├── lgbm_model.py           # LightGBM モデル定義・学習
│   ├── transformer_model.py    # Transformer モデル定義・学習（PyTorch）
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
[データ取得・永続化]
  SQLite (alphasignal.db) から読み込み
  不足分を Yahoo Finance / ニュースAPI から取得し DB へ保存
  （重複排除・欠損値チェックを自動実行）
       ↓
[特徴量エンジニアリング]
  価格指標・テクニカル指標・時間特徴量
  ニュースインパクト指数 (NIS)
       ↓
[モデル学習 (80/20 時系列分割)]
  ┌─────────────┐   ┌──────────────────┐
  │  LightGBM   │   │   Transformer    │
  │（2D 入力）  │   │（PyTorch実装）   │
  └──────┬──────┘   └────────┬─────────┘
         └─────────┬──────────┘
                   ↓
        [スタッキングアンサンブル]
         線形回帰メタモデル
         入力: [LightGBM予測, Transformer予測, NIS]
```
※ Transformerは以前のTensorFlow/Keras実装からPyTorch実装に移行。インターフェースは互換性を維持。
※ データベース構造の変更が必要な場合は、既存カラムを削除せず新しいカラムを追加することで後方互換性を維持する。

---

## 4. 開発ルール

（省略: 以前の内容を維持）

---

## 7. 変更履歴

| 日付 | バージョン | 変更者 | 変更内容 |
|------|-----------|--------|---------|
| 2026-05-25 | v1.0.0 | Claude Sonnet 4.6 | 初期リリース。LightGBM + Transformer + ニュースセンチメントの基本パイプライン実装。`main.py` 作成。 |
| 2026-05-25 | v1.1.0 | Claude Sonnet 4.6 | プロジェクト名を AlphaSignal に変更。`alphasignal.py`（メインエントリ）と `src/menu.py`（対話式メニュー）を新規作成。企業名→ティッカー自動変換辞書（日英約40社）追加。実行後メニュー（再実行・終了）追加。`AlphaSignal.md`（本ファイル）作成。 |
| 2026-05-25 | v1.1.1 | Gemini CLI | AVX非対応環境でのIllegal instructionエラー回避のため、TensorFlowからPyTorchへモデル実装を移行。`main.py`および`alphasignal.py`の動作を確認。 |
| 2026-05-25 | v1.2.0 | Gemini CLI | SQLite データベース (`alphasignal.db`) を統合。データの永続化、欠損値チェック、重複防止機能を実装。`src/database.py` を新規作成し、各モジュールと連携。 |
| 2026-05-25 | v1.2.0 | Gemini CLI | Git リポジトリの初期化、`.gitignore` の作成、および初期コミットを実施。GitHub 連携の準備完了。 |
| 2026-05-26 | v1.2.1 | Gemini CLI | 環境変数（`.env`）による設定管理を導入。`python-dotenv` を追加し、DB名やAPIキーの外部管理を可能に。 |

---

*このドキュメントは AlphaSignal プロジェクトの Single Source of Truth です。*
*再開時は必ず読み込み、終了時は変更履歴を更新してください。*
