# 株価予測モデル: LightGBM + Transformer + ニュースセンチメント

## 概要

本プロジェクトは「株価予測モデル設計書」に基づき、LightGBM・Transformer・
ニュースセンチメント指数をアンサンブルした株価予測システムです。

## プロジェクト構成

```
stock_prediction_project/
├── main.py                    # メイン実行スクリプト
├── requirements.txt           # 依存ライブラリ
├── src/
│   ├── features.py            # 特徴量エンジニアリング（テクニカル指標など）
│   ├── news_sentiment.py      # ニュースセンチメント分析・NIS計算
│   ├── lgbm_model.py          # LightGBMモデル
│   ├── transformer_model.py   # Transformerモデル（Keras）
│   └── ensemble.py            # スタッキングアンサンブル・評価
└── results/                   # 出力（JSONサマリー・CSVファイル）
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 実行方法

```bash
# デフォルト（AAPL, 2018-2023）
python main.py

# カスタム設定
python main.py --ticker 7203.T --start 2019-01-01 --end 2024-01-01 --epochs 100

# オプション
--ticker   ティッカーシンボル（Yahoo Finance形式）
--start    開始日 YYYY-MM-DD
--end      終了日 YYYY-MM-DD
--seq_len  Transformerシーケンス長（デフォルト: 30）
--epochs   Transformer学習エポック数（デフォルト: 50）
--save_dir 結果保存ディレクトリ（デフォルト: results/）
```

## アーキテクチャ

### 特徴量 (src/features.py)
- 価格指標: OHLCV、比率、差分、対数リターン
- テクニカル指標: SMA/EMA (5/10/20日)、RSI、MACD、ボリンジャーバンド
- ボリューム指標: 移動平均、ボリューム比率
- 時間特徴量: 曜日、日、月、年
- **ニュースセンチメント**: NIS日次/移動平均、ニュース量

### ニュースインパクト指数 (src/news_sentiment.py)
```
NIS_t = Σ w_i,t × S_i,t × D(days_ago_i,t)
```
- センチメントスコア: FinBERT (positive確率 - negative確率)
- 時間減衰: 指数関数的減衰 exp(-λ × days_ago)
- ボリューム加重: 記事数・関連性スコアで加重

### モデル (src/lgbm_model.py, src/transformer_model.py)
- **LightGBM**: 勾配ブースティング、Early Stoppingあり
- **Transformer**: Multi-Head Self-Attention × 2ブロック + MLP出力

### アンサンブル (src/ensemble.py)
- メタ特徴量: [LightGBM予測, Transformer予測, NIS] → 線形回帰

## FinBERTの実装（実運用）

`src/news_sentiment.py` の `get_finbert_sentiment()` にコメントアウトされた
実装例があります。本番利用時は以下をインストールして有効化してください:

```bash
pip install transformers torch
```

## 評価指標

- **RMSE**: 二乗平均平方根誤差
- **MAE**: 平均絶対誤差
- **MAPE**: 平均絶対パーセント誤差
- **方向性精度**: 騰落方向の一致率
- **バックテスト**: シャープレシオ、最大ドローダウン、総リターン
