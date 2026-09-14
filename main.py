"""
AlphaSignal - 全体統合メイン実行スクリプト
株価予測、ニュースセンチメント分析、アルファ抽出・結合、トレード診断およびデータベース管理の統合システム

使用方法:
    python main.py             # 対話式統合メニュー (デフォルト)
    python main.py --cli       # コマンドライン引数モード
"""

import argparse
import os
import sys
import json
import time
import warnings
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

# モジュールインポート
sys.path.insert(0, os.path.dirname(__file__))
from src.features import (
    fetch_stock_data, create_features_with_sentiment,
    preprocess_data, FEATURE_COLS
)
from src.news_sentiment import fetch_news_data, calculate_news_impact_with_db
from src.lgbm_model import build_and_train_lightgbm
from src.transformer_model import build_transformer_model, train_transformer
from src.ensemble import ensemble_predictions, evaluate_model, backtest_strategy
from src.alpha_engine import AlphaEngine, AlphaEngineConfig
from src.signals.alpha_combiner import AlphaCombiner
from src.database import db_manager
from src.menu import (
    display_main_menu, menu_select_company, menu_select_period,
    run_interactive_menu, menu_post_run, print_header, print_separator,
    SEPARATOR, input_with_prompt, resolve_ticker
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return super().default(obj)


# ══════════════════════════════════════════════════════════════
#  1. 統合株価予測 & バックテスト
# ══════════════════════════════════════════════════════════════
def run_prediction(ticker: str, start: str, end: str,
                   seq_len: int = 30, epochs: int = 50, save_dir: str = 'results'):
    """株価予測パイプラインを実行し結果を保存・表示する"""
    os.makedirs(save_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  AlphaSignal 予測実行中                                  ║")
    print(f"║  銘柄: {ticker:<10}  期間: {start} ～ {end}   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. 株価データ取得
    print("\n[1/6] 株価データ取得")
    df = fetch_stock_data(ticker, start, end)
    if len(df) < 200:
        print(f"  ✗ データ不足（{len(df)}件）。200件以上のデータが必要です。")
        return None

    # 2. ニュースセンチメント
    print("\n[2/6] ニュースデータ取得・センチメント計算")
    news_df = fetch_news_data(ticker, start, end)
    print(f"  → ニュース記事数: {len(news_df)} 件")
    news_impact_df = calculate_news_impact_with_db(ticker, news_df, decay_rate=0.1, window_size=5)
    print(f"  → ニュースインパクト指数計算完了 ({len(news_impact_df)} 日分)")

    # 3. 特徴量エンジニアリング
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

    # 4. LightGBM 学習
    print("\n[4/6] LightGBM 学習")
    t0 = time.time()
    lgbm_model = build_and_train_lightgbm(X_lt, y_lt, X_lv, y_lv)
    print(f"  → 学習時間: {time.time() - t0:.1f}秒")
    lgbm_preds_sc = lgbm_model.predict(X_ltest)

    # 5. Transformer 学習
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

    # 6. アンサンブル・評価
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

    # 評価表示
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
    print("  直近5日の予測結果")
    print(SEPARATOR)
    test_dates = df_feat.index[split_l + offset:][:len(trans_p)]
    print(f"  {'日付':<12} {'実績':>10} {'LightGBM':>10} {'Transformer':>12} {'Ensemble':>10}")
    print("  " + "─" * 58)
    for i in range(-5, 0):
        dt = (test_dates[i].strftime('%Y-%m-%d')
              if hasattr(test_dates[i], 'strftime') else str(test_dates[i]))
        print(f"  {dt:<12} {y_true[i]:>10.2f} {lgbm_p[i]:>10.2f} "
              f"{trans_p[i]:>12.2f} {ens_p[i]:>10.2f}")

    # 保存
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
    print(f"  ✓ 評価結果 JSON : {json_path}")
    print(f"  ✓ 予測結果 CSV  : {csv_path}")
    print(SEPARATOR)

    return result


# ══════════════════════════════════════════════════════════════
#  2. アルファ結合 (AlphaEngine)
# ══════════════════════════════════════════════════════════════
def run_alpha_engine_mode():
    """AlphaEngine を用いたアルファウェイト算定と結合"""
    print_separator("AlphaEngine: シグナル結合 & ウェイト最適化")
    print()
    ticker = menu_select_company()
    start, end = menu_select_period()

    print(f"\n[1/2] データ取得中 ({ticker}, {start} ~ {end})...")
    df = fetch_stock_data(ticker, start, end)
    if len(df) < 50:
        print("  ⚠ データが少なすぎます。")
        return

    # リターン計算および技術シグナル計算
    returns = df['Close'].pct_change().dropna()
    ma5_ret = df['Close'].pct_change(5).dropna()
    ma20_ret = df['Close'].pct_change(20).dropna()
    vol_ret = (df['Volume'].pct_change() * np.sign(returns)).dropna()

    returns_matrix = pd.DataFrame({
        'Daily_Return': returns,
        'MA5_Return': ma5_ret,
        'MA20_Return': ma20_ret,
        'Volume_Return': vol_ret
    }).dropna()

    print(f"  → 構築されたシグナル行列: 形状 {returns_matrix.shape}")

    engine = AlphaEngine(AlphaEngineConfig(d=10, l2_reg=1e-3))
    weights = engine.fit_weights(returns_matrix)

    latest_signals = returns_matrix.iloc[-1]
    combined_signal = engine.combine(latest_signals, weights)

    print("\n  ── 算定されたシグナルウェイト (L1 normalized) ──")
    for sig, w in weights.items():
        print(f"  • {sig:<15}: {w:>8.4f}")
    print("  " + "─" * 40)
    print(f"  最新合成アルファスコア (Alpha Combined): {combined_signal:>10.6f}")
    print()


# ══════════════════════════════════════════════════════════════
#  3. 残差アルファ抽出 & 合成 (AlphaCombiner)
# ══════════════════════════════════════════════════════════════
def run_alpha_combiner_mode():
    """AlphaCombiner を用いた直交化残差アルファ抽出"""
    print_separator("AlphaCombiner: 因子直交化 & 残差アルファ抽出")
    print()
    ticker = menu_select_company()
    start, end = menu_select_period()

    print(f"\n[1/2] データ取得中 ({ticker}, {start} ~ {end})...")
    df = fetch_stock_data(ticker, start, end)
    if len(df) < 50:
        print("  ⚠ データが少なすぎます。")
        return

    ret = df['Close'].pct_change()
    ret5 = df['Close'].pct_change(5)
    ret20 = df['Close'].pct_change(20)

    returns_matrix = pd.DataFrame({
        'Return_1D': ret,
        'Return_5D': ret5,
        'Return_20D': ret20,
    }).dropna()

    combiner = AlphaCombiner(method='auto')
    res = combiner.compute_pipeline(returns_matrix, d=10)

    print("\n  ── 抽出された因子と残差アルファ (Residual Alpha) ──")
    weights = res['weights']
    for sig, w in weights.items():
        print(f"  • {sig:<15}: ウェイト = {w:>8.4f}")
    print("  " + "─" * 40)
    print(f"  残差アルファ合成スコア (Combined Alpha): {res['combined_signal']:>10.6f}")
    print()


# ══════════════════════════════════════════════════════════════
#  4. 総合トレード診断・売買判定
# ══════════════════════════════════════════════════════════════
def run_trade_signal_alert():
    """予測値とアルファスコアを総合したトレード診断機能"""
    print_separator("総合トレード診断 (Trade Signal Alert)")
    print()
    ticker = menu_select_company()
    print()
    start, end = menu_select_period()

    print("\n  パイプラインを実行中...")
    result = run_prediction(ticker, start, end, seq_len=30, epochs=30, save_dir='results')
    if not result:
        return

    metrics = result['metrics']['Ensemble']
    dir_acc = metrics.get('Directional Accuracy', 0.5)
    last_act = result.get('metrics', {})

    # 直近予測情報
    safe_ticker = ticker.replace("^", "").replace("/", "_")
    csv_path = f"results/{safe_ticker}_predictions.csv"
    if os.path.exists(csv_path):
        pred_df = pd.read_csv(csv_path)
        last_row = pred_df.iloc[-1]
        prev_row = pred_df.iloc[-2]
        predicted_change = (last_row['Ensemble'] - prev_row['Actual']) / prev_row['Actual']

        print("\n" + SEPARATOR)
        print("  【トレードシグナル診断結果】")
        print(SEPARATOR)
        print(f"  対象銘柄           : {ticker}")
        print(f"  最新株価 (最終実績) : {prev_row['Actual']:.2f}")
        print(f"  予測終値 (Ensemble) : {last_row['Ensemble']:.2f}")
        print(f"  予測騰落率         : {predicted_change * 100:+.2f}%")
        print(f"  方向性精度 (検証期): {dir_acc * 100:.1f}%")

        # 売買シグナル判定
        threshold = 0.005  # 0.5%
        if predicted_change > threshold and dir_acc >= 0.50:
            signal_text = "🟢 [BUY] 買い推奨 (上昇予測)"
        elif predicted_change < -threshold and dir_acc >= 0.50:
            signal_text = "🔴 [SELL] 売り推奨 (下落予測)"
        else:
            signal_text = "🟡 [HOLD] 静観 / 保持 (方向性中立)"

        print("\n  ┌──────────────────────────────────────────────────┐")
        print(f"  │ 判定シグナル: {signal_text:<34} │")
        print("  └──────────────────────────────────────────────────┘")
        print()


# ══════════════════════════════════════════════════════════════
#  5. データベース状態確認
# ══════════════════════════════════════════════════════════════
def run_db_inspection():
    """alphasignal.db のステータスと保持データ件数、Google Drive同期状態を表示"""
    print_separator("データベース状態確認 (alphasignal.db)")
    print()
    db_path = db_manager.db_path
    if not os.path.exists(db_path):
        print(f"  ⚠ データベースファイルが存在しません: {db_path}")
        return

    file_size_kb = os.path.getsize(db_path) / 1024.0

    with db_manager._get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM stocks")
        stock_count, stock_tickers = c.fetchone()

        c.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM news_impact")
        news_count, news_tickers = c.fetchone()

    print(f"  データベースパス : {os.path.abspath(db_path)}")
    print(f"  ファイルサイズ   : {file_size_kb:.2f} KB")
    print(f"  登録株価レコード : {stock_count:,} 件 ({stock_tickers} 銘柄)")
    print(f"  ニュースインパクト: {news_count:,} 件 ({news_tickers} 銘柄)")
    print()

    # Google Drive 同期状態
    sync_status = db_manager.get_sync_status()
    print("  [Google Drive 同期ステータス]")
    if sync_status.get("enabled"):
        if sync_status.get("available"):
            print("  状態             : 有効 (接続成功)")
            print(f"  フォルダID       : {sync_status.get('folder_id')}")
            print(f"  フォルダURL      : https://drive.google.com/drive/folders/{sync_status.get('folder_id')}")
            if sync_status.get("remote_exists"):
                print(f"  Drive上ファイルID: {sync_status.get('remote_id')}")
                print(f"  Drive上最終更新  : {sync_status.get('remote_modified')}")
                remote_kb = int(sync_status.get('remote_size', 0)) / 1024.0
                print(f"  Drive上サイズ    : {remote_kb:.2f} KB")
            else:
                print("  Drive上ファイル  : 未検出")
        else:
            print("  状態             : 接続不可 (認証またはネットワーク設定を確認してください)")
    else:
        print("  状態             : 無効 (GOOGLE_DRIVE_SYNC=false)")
    print()


# ══════════════════════════════════════════════════════════════
#  エントリーポイント
# ══════════════════════════════════════════════════════════════
def parse_cli_args():
    p = argparse.ArgumentParser(
        description='AlphaSignal トレード統合システム',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cli',      action='store_true', help='CLIモードで実行（非対話）')
    p.add_argument('--ticker',   type=str, default='AAPL')
    p.add_argument('--start',    type=str, default='2020-01-01')
    p.add_argument('--end',      type=str, default='2023-12-31')
    p.add_argument('--seq_len',  type=int, default=30)
    p.add_argument('--epochs',   type=int, default=50)
    p.add_argument('--save_dir', type=str, default='results')
    return p.parse_args()


def main():
    args = parse_cli_args()

    # CLIモード
    if args.cli:
        ticker = resolve_ticker(args.ticker)
        run_prediction(ticker, args.start, args.end,
                       args.seq_len, args.epochs, args.save_dir)
        return

    # 対話式統合メニューモード
    while True:
        choice = display_main_menu()

        if choice == "1":
            config = run_interactive_menu()
            if config:
                run_prediction(**config)
                menu_post_run(config['ticker'], config['save_dir'])
        elif choice == "2":
            run_alpha_engine_mode()
            input("\n  Enterキーを押してメインメニューに戻ります...")
        elif choice == "3":
            run_alpha_combiner_mode()
            input("\n  Enterキーを押してメインメニューに戻ります...")
        elif choice == "4":
            run_trade_signal_alert()
            input("\n  Enterキーを押してメインメニューに戻ります...")
        elif choice == "5":
            run_db_inspection()
            input("\n  Enterキーを押してメインメニューに戻ります...")
        elif choice == "6":
            print("\n  AlphaSignal を終了します。\n")
            break


if __name__ == '__main__':
    main()
