"""
AlphaSignal - メインエントリーポイント
対話式メニューから株価予測を実行する

使用方法:
    python alphasignal.py              # 対話式メニュー
    python alphasignal.py --cli        # コマンドライン引数モード（従来互換）
"""

import argparse
import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.features import (
    fetch_stock_data, create_features_with_sentiment,
    preprocess_data, FEATURE_COLS
)
from src.news_sentiment import fetch_news_data, calculate_news_impact_score
from src.lgbm_model import build_and_train_lightgbm
from src.transformer_model import build_transformer_model, train_transformer
from src.ensemble import ensemble_predictions, evaluate_model, backtest_strategy
from src.menu import (
    run_interactive_menu, menu_post_run,
    print_header, print_separator, SEPARATOR
)


# ══════════════════════════════════════════════════════════════
#  JSON シリアライザ
# ══════════════════════════════════════════════════════════════
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return super().default(obj)


# ══════════════════════════════════════════════════════════════
#  予測パイプライン本体
# ══════════════════════════════════════════════════════════════
def run_prediction(ticker: str, start: str, end: str,
                   seq_len: int, epochs: int, save_dir: str):
    """予測パイプラインを実行し結果辞書を返す"""

    os.makedirs(save_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  AlphaSignal  予測実行中                                 ║")
    print(f"║  銘柄: {ticker:<10}  期間: {start} ～ {end}   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── 1. 株価データ取得 ───────────────────────────────────────
    print("\n[1/6] 株価データ取得")
    df = fetch_stock_data(ticker, start, end)
    if len(df) < 200:
        print(f"  ✗ データ不足（{len(df)}件）。200件以上のデータが必要です。")
        return None

    # ── 2. ニュースセンチメント ─────────────────────────────────
    print("\n[2/6] ニュースデータ取得・センチメント計算")
    news_df = fetch_news_data(ticker, start, end)
    print(f"  → ニュース記事数: {len(news_df)} 件")
    news_impact_df = calculate_news_impact_score(news_df, decay_rate=0.1, window_size=5)
    print(f"  → ニュースインパクト指数計算完了 ({len(news_impact_df)} 日分)")

    # ── 3. 特徴量エンジニアリング ───────────────────────────────
    print("\n[3/6] 特徴量エンジニアリング")
    df_feat = create_features_with_sentiment(df, news_impact_df)
    print(f"  → 特徴量数: {len(FEATURE_COLS)} | サンプル数: {len(df_feat)}")

    X_lgbm, y_lgbm, X_trans, y_trans, feat_scaler, tgt_scaler = preprocess_data(
        df_feat, FEATURE_COLS, 'Target', seq_len
    )
    print(f"  → LightGBM: {X_lgbm.shape} | Transformer: {X_trans.shape}")

    # 時系列分割（80/20）
    split_l = int(len(X_lgbm) * 0.8)
    split_t = int(len(X_trans) * 0.8)
    val_l   = int(split_l * 0.9)

    X_lt, X_lv = X_lgbm[:val_l],   X_lgbm[val_l:split_l]
    y_lt, y_lv = y_lgbm[:val_l],   y_lgbm[val_l:split_l]
    X_ltest, y_ltest = X_lgbm[split_l:], y_lgbm[split_l:]
    X_ttr, y_ttr = X_trans[:split_t], y_trans[:split_t]
    X_tte, y_tte = X_trans[split_t:], y_trans[split_t:]

    # ── 4. LightGBM 学習 ────────────────────────────────────────
    print("\n[4/6] LightGBM 学習")
    t0 = time.time()
    lgbm_model = build_and_train_lightgbm(X_lt, y_lt, X_lv, y_lv)
    print(f"  → 学習時間: {time.time() - t0:.1f}秒")
    lgbm_preds_sc = lgbm_model.predict(X_ltest)

    # ── 5. Transformer 学習 ─────────────────────────────────────
    print("\n[5/6] Transformer 学習")
    t0 = time.time()
    transformer = build_transformer_model(
        input_shape=(seq_len, X_trans.shape[2]),
        head_size=64, num_heads=4, ff_dim=128,
        num_transformer_blocks=2, mlp_units=[128, 64], dropout=0.1,
    )
    transformer = train_transformer(
        transformer, X_ttr, y_ttr,
        epochs=epochs, batch_size=32, validation_split=0.1,
    )
    print(f"  → 学習時間: {time.time() - t0:.1f}秒")
    trans_preds_sc = transformer.predict(X_tte, verbose=0).flatten()

    # ── 6. アンサンブル・評価 ───────────────────────────────────
    print("\n[6/6] アンサンブル・評価")
    offset = len(lgbm_preds_sc) - len(trans_preds_sc)
    lgbm_al = lgbm_preds_sc[offset:]
    y_al    = y_ltest[offset:]

    def inv(a):
        return tgt_scaler.inverse_transform(a.reshape(-1, 1)).ravel()

    lgbm_p  = inv(lgbm_al)
    trans_p = inv(trans_preds_sc)
    y_true  = inv(y_al)

    nis_vals = None
    if 'news_impact_score' in df_feat.columns:
        nis_arr  = df_feat['news_impact_score'].values
        nis_test = nis_arr[split_l + offset:][:len(trans_p)]
        if len(nis_test) == len(trans_p):
            nis_vals = nis_test

    ens_p, _ = ensemble_predictions(lgbm_p, trans_p, y_true, nis_vals)

    # 評価
    print("\n" + SEPARATOR)
    print("  予測性能評価")
    print(SEPARATOR)
    m_lgbm  = evaluate_model(y_true, lgbm_p,  "LightGBM")
    m_trans = evaluate_model(y_true, trans_p,  "Transformer")
    m_ens   = evaluate_model(y_true, ens_p,    "Ensemble (Stack)")

    # バックテスト
    print("\n" + SEPARATOR)
    print("  バックテスト（Ensemble戦略）")
    print(SEPARATOR)
    prices = df_feat['Close'].values[split_l + offset:][:len(trans_p)]
    bt_results = {}
    if len(prices) == len(ens_p):
        bt_results = backtest_strategy(y_true, ens_p, prices)

    # 直近5日
    print("\n" + SEPARATOR)
    print("  直近5日の予測")
    print(SEPARATOR)
    test_dates = df_feat.index[split_l + offset:][:len(trans_p)]
    print(f"  {'日付':<12} {'実績':>10} {'LightGBM':>10} {'Transformer':>12} {'Ensemble':>10}")
    print("  " + "─" * 58)
    for i in range(-5, 0):
        dt = (test_dates[i].strftime('%Y-%m-%d')
              if hasattr(test_dates[i], 'strftime') else str(test_dates[i]))
        print(f"  {dt:<12} {y_true[i]:>10.2f} {lgbm_p[i]:>10.2f} "
              f"{trans_p[i]:>12.2f} {ens_p[i]:>10.2f}")

    # ── 結果保存 ────────────────────────────────────────────────
    result = {
        'ticker':     ticker,
        'period':     f"{start} ~ {end}",
        'n_samples':  len(df_feat),
        'n_features': len([c for c in FEATURE_COLS if c in df_feat.columns]),
        'metrics': {
            'LightGBM':    m_lgbm,
            'Transformer': m_trans,
            'Ensemble':    m_ens,
        },
        'backtest': bt_results,
    }

    safe_ticker = ticker.replace("^", "").replace("/", "_")
    json_path = os.path.join(save_dir, f"{safe_ticker}_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    pred_df = pd.DataFrame({
        'Date':        test_dates[:len(trans_p)],
        'Actual':      y_true,
        'LightGBM':    lgbm_p,
        'Transformer': trans_p,
        'Ensemble':    ens_p,
    })
    csv_path = os.path.join(save_dir, f"{safe_ticker}_predictions.csv")
    pred_df.to_csv(csv_path, index=False)

    print()
    print(SEPARATOR)
    print(f"  ✓ 結果 JSON : {json_path}")
    print(f"  ✓ 予測 CSV  : {csv_path}")
    print(SEPARATOR)

    return result


# ══════════════════════════════════════════════════════════════
#  エントリーポイント
# ══════════════════════════════════════════════════════════════
def parse_cli_args():
    p = argparse.ArgumentParser(
        description='AlphaSignal 株価予測システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python alphasignal.py                          # 対話式メニュー（推奨）
  python alphasignal.py --cli --ticker AAPL      # CLIモード
  python alphasignal.py --cli --ticker トヨタ    # 日本語企業名OK
        """
    )
    p.add_argument('--cli',      action='store_true', help='CLIモードで実行（非対話式）')
    p.add_argument('--ticker',   type=str, default='AAPL')
    p.add_argument('--start',    type=str, default='2020-01-01')
    p.add_argument('--end',      type=str, default='2023-12-31')
    p.add_argument('--seq_len',  type=int, default=30)
    p.add_argument('--epochs',   type=int, default=50)
    p.add_argument('--save_dir', type=str, default='results')
    return p.parse_args()


def main():
    args = parse_cli_args()

    # ── CLIモード ────────────────────────────────────────────
    if args.cli:
        from src.menu import resolve_ticker
        ticker = resolve_ticker(args.ticker)
        run_prediction(ticker, args.start, args.end,
                       args.seq_len, args.epochs, args.save_dir)
        return

    # ── 対話式メニューモード ─────────────────────────────────
    last_config = None

    while True:
        # メニューから設定を取得
        config = run_interactive_menu()
        if config is None:
            break  # ユーザーが終了選択

        last_config = config

        # 予測実行
        print()
        run_prediction(
            ticker   = config['ticker'],
            start    = config['start'],
            end      = config['end'],
            seq_len  = config['seq_len'],
            epochs   = config['epochs'],
            save_dir = config['save_dir'],
        )

        # 実行後メニュー
        choice = menu_post_run(config['ticker'], config['save_dir'])

        if choice == "1":
            # 別の銘柄で再実行 → メニューに戻る
            continue
        elif choice == "2":
            # 同じ設定で再実行
            if last_config:
                print()
                run_prediction(**last_config)
                choice2 = menu_post_run(last_config['ticker'], last_config['save_dir'])
                if choice2 == "3":
                    break
        else:
            # 終了
            break

    print()
    print("  AlphaSignal を終了しました。")
    print("  結果ファイルは results/ ディレクトリに保存されています。")
    print()


if __name__ == '__main__':
    main()
