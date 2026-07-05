"""
株価予測モデル メイン実行スクリプト
LightGBM + Transformer + ニュースセンチメント アンサンブル

使用方法:
    python main.py
    python main.py --ticker 7203.T --start 2018-01-01 --end 2023-12-31
"""

import argparse
import os
import sys
import json
import time
import warnings
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

# プロジェクト内モジュール
sys.path.insert(0, os.path.dirname(__file__))
from src.features import (
    fetch_stock_data, create_features_with_sentiment,
    preprocess_data, FEATURE_COLS
)
from src.news_sentiment import fetch_news_data, calculate_news_impact_score, calculate_news_impact_with_db
from src.lgbm_model import build_and_train_lightgbm
from src.transformer_model import build_transformer_model, train_transformer
from src.ensemble import (
    ensemble_predictions, evaluate_model, backtest_strategy
)


def parse_args():
    parser = argparse.ArgumentParser(description='株価予測モデル（LightGBM+Transformer+Sentiment）')
    parser.add_argument('--ticker',    type=str, default='AAPL',       help='ティッカーシンボル')
    parser.add_argument('--start',     type=str, default='2018-01-01', help='開始日 (YYYY-MM-DD)')
    parser.add_argument('--end',       type=str, default='2023-12-31', help='終了日 (YYYY-MM-DD)')
    parser.add_argument('--seq_len',   type=int, default=30,           help='Transformer入力シーケンス長')
    parser.add_argument('--epochs',    type=int, default=50,           help='Transformer学習エポック数')
    parser.add_argument('--save_dir',  type=str, default='results',    help='結果保存ディレクトリ')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("  株価予測モデル: LightGBM + Transformer + ニュースセンチメント")
    print("=" * 60)
    print(f"  ティッカー  : {args.ticker}")
    print(f"  期間        : {args.start} ～ {args.end}")
    print(f"  シーケンス長: {args.seq_len}")
    print("=" * 60)

    # ── 1. データ取得 ────────────────────────────────────────
    print("\n[1/6] 株価データ取得")
    df = fetch_stock_data(args.ticker, args.start, args.end)
    if len(df) < 200:
        print("ERROR: データが少なすぎます（200件以上必要）")
        sys.exit(1)

    print("\n[2/6] ニュースデータ取得・センチメント計算")
    news_df = fetch_news_data(args.ticker, args.start, args.end)
    print(f"  → ニュース記事数: {len(news_df)} 件")
    news_impact_df = calculate_news_impact_with_db(args.ticker, news_df, decay_rate=0.1, window_size=5)
    print(f"  → ニュースインパクト指数計算完了 ({len(news_impact_df)} 日分)")

    # ── 2. 特徴量生成 ────────────────────────────────────────
    print("\n[3/6] 特徴量エンジニアリング")
    df_feat = create_features_with_sentiment(df, news_impact_df)
    print(f"  → 特徴量数: {len(FEATURE_COLS)} | サンプル数: {len(df_feat)}")

    X_lgbm, y_lgbm, X_trans, y_trans, feat_scaler, tgt_scaler = preprocess_data(
        df_feat, FEATURE_COLS, 'Target', args.seq_len
    )
    print(f"  → LightGBM形状: {X_lgbm.shape} | Transformer形状: {X_trans.shape}")

    # ── 3. 訓練/テスト分割（時系列分割: 80/20）──────────────
    split_lgbm = int(len(X_lgbm) * 0.8)
    split_trans = int(len(X_trans) * 0.8)

    X_lgbm_train, X_lgbm_test = X_lgbm[:split_lgbm], X_lgbm[split_lgbm:]
    y_lgbm_train, y_lgbm_test = y_lgbm[:split_lgbm], y_lgbm[split_lgbm:]
    X_trans_train, X_trans_test = X_trans[:split_trans], X_trans[split_trans:]
    y_trans_train, y_trans_test = y_trans[:split_trans], y_trans[split_trans:]

    val_split = int(len(X_lgbm_train) * 0.9)
    X_lgbm_val, y_lgbm_val = X_lgbm_train[val_split:], y_lgbm_train[val_split:]
    X_lgbm_train2, y_lgbm_train2 = X_lgbm_train[:val_split], y_lgbm_train[:val_split]

    # ── 4. LightGBM学習 ─────────────────────────────────────
    print("\n[4/6] LightGBMモデル学習")
    t0 = time.time()
    lgbm_model = build_and_train_lightgbm(
        X_lgbm_train2, y_lgbm_train2, X_lgbm_val, y_lgbm_val
    )
    print(f"  → 学習時間: {time.time() - t0:.1f}秒")
    lgbm_preds_scaled = lgbm_model.predict(X_lgbm_test)

    # ── 5. Transformer学習 ──────────────────────────────────
    print("\n[5/6] Transformerモデル学習")
    t0 = time.time()
    n_features = X_trans.shape[2]
    transformer = build_transformer_model(
        input_shape=(args.seq_len, n_features),
        head_size=64,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2,
        mlp_units=[128, 64],
        dropout=0.1,
    )
    transformer = train_transformer(
        transformer, X_trans_train, y_trans_train,
        epochs=args.epochs, batch_size=32, validation_split=0.1,
    )
    print(f"  → 学習時間: {time.time() - t0:.1f}秒")
    trans_preds_scaled = transformer.predict(X_trans_test, verbose=0).flatten()

    # ── 6. アンサンブル・評価 ───────────────────────────────
    print("\n[6/6] アンサンブル・評価")

    # Transformer側に合わせてLightGBMの予測をトリミング
    offset = len(lgbm_preds_scaled) - len(trans_preds_scaled)
    lgbm_preds_aligned = lgbm_preds_scaled[offset:]
    y_true_aligned = y_lgbm_test[offset:]

    # 逆スケーリング
    def inv(arr):
        return tgt_scaler.inverse_transform(arr.reshape(-1, 1)).ravel()

    lgbm_preds   = inv(lgbm_preds_aligned)
    trans_preds  = inv(trans_preds_scaled)
    y_true       = inv(y_true_aligned)

    # ニュースインパクト指数（テスト期間のみ）
    nis_col = 'news_impact_score'
    nis_vals = None
    if nis_col in df_feat.columns:
        nis_array = df_feat[nis_col].values
        # Transformer test起点に揃える
        nis_test = nis_array[split_lgbm + offset:][:len(trans_preds)]
        if len(nis_test) == len(trans_preds):
            nis_vals = nis_test

    # アンサンブル
    ensemble_preds, meta_model = ensemble_predictions(
        lgbm_preds, trans_preds, y_true, nis_vals
    )

    # 評価
    print("\n  ── 予測性能評価 ──")
    metrics_lgbm   = evaluate_model(y_true, lgbm_preds,    "LightGBM")
    metrics_trans  = evaluate_model(y_true, trans_preds,   "Transformer")
    metrics_ens    = evaluate_model(y_true, ensemble_preds,"Ensemble (Stack)")

    # バックテスト
    print("\n  ── バックテスト結果 ──")
    prices = df_feat['Close'].values[split_lgbm + offset:][:len(trans_preds)]
    if len(prices) == len(ensemble_preds):
        print("\n  【Ensemble戦略】")
        bt_results = backtest_strategy(y_true, ensemble_preds, prices)

    # 最新5日の予測表示
    print("\n  ── 直近5日の予測 ──")
    print(f"  {'日付':<12} {'実績':>10} {'LightGBM':>10} {'Transformer':>12} {'Ensemble':>10}")
    print("  " + "-" * 58)
    test_dates = df_feat.index[split_lgbm + offset:][:len(trans_preds)]
    for i in range(-5, 0):
        dt = test_dates[i].strftime('%Y-%m-%d') if hasattr(test_dates[i], 'strftime') else str(test_dates[i])
        print(f"  {dt:<12} {y_true[i]:>10.2f} {lgbm_preds[i]:>10.2f} "
              f"{trans_preds[i]:>12.2f} {ensemble_preds[i]:>10.2f}")

    # ── 結果保存 ────────────────────────────────────────────
    result_summary = {
        'ticker':     args.ticker,
        'period':     f"{args.start} ~ {args.end}",
        'n_samples':  len(df_feat),
        'n_features': len([c for c in FEATURE_COLS if c in df_feat.columns]),
        'metrics': {
            'LightGBM':   metrics_lgbm,
            'Transformer': metrics_trans,
            'Ensemble':   metrics_ens,
        }
    }
    result_path = os.path.join(args.save_dir, f"{args.ticker}_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, np.integer)): return float(obj)
                return super().default(obj)
        json.dump(result_summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n  結果を保存しました: {result_path}")

    # 予測値をCSV保存
    pred_df = pd.DataFrame({
        'Date': test_dates[:len(trans_preds)],
        'Actual': y_true,
        'LightGBM': lgbm_preds,
        'Transformer': trans_preds,
        'Ensemble': ensemble_preds,
    })
    csv_path = os.path.join(args.save_dir, f"{args.ticker}_predictions.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"  予測値を保存しました: {csv_path}")

    print("\n" + "=" * 60)
    print("  完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
