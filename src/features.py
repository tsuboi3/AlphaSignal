"""
特徴量エンジニアリングモジュール
価格・テクニカル指標・センチメント特徴量の生成
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler


from src.database import db_manager


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    株価データを取得。DBを優先し、不足分をYahoo Financeから取得して保存。
    """
    print(f"  → {ticker} のデータを取得中 ({start_date} ～ {end_date})")
    
    # 1. DBから試行
    df_db = db_manager.load_stocks(ticker, start_date, end_date)
    # 単純な件数チェックではなく、期間のカバー率で判断するのが理想的だが、
    # ここでは既存ロジックに合わせて200件（main.pyの要求）を一つの目安にする
    if not df_db.empty and len(df_db) >= 200:
        print(f"  → データベースから {len(df_db)} 件取得しました。")
        return df_db

    # 2. Yahoo Financeから取得
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         auto_adjust=True, progress=False)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.dropna(inplace=True)
        
        if not df.empty:
            # DBに保存
            db_manager.save_stocks(ticker, df)
            print(f"  → データを取得しデータベースに保存しました（{len(df)} 件）")
            
    except Exception as e:
        # CSVフォールバック（オフライン・テスト用）
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', f'{ticker}_dummy.csv'
        )
        if os.path.exists(csv_path):
            print(f"  → Yahoo Finance取得失敗 ({e})")
            print(f"  → CSVファイルを使用: {csv_path}")
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = df.loc[start_date:end_date]
            # CSVから読み込んだデータもDBに保存
            if not df.empty:
                db_manager.save_stocks(ticker, df)
        else:
            raise RuntimeError(
                f"株価データ取得失敗。{csv_path} にCSVを配置してください。\n"
                f"エラー詳細: {e}"
            )
    print(f"  → {len(df)} 件取得完了")
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標・価格特徴量を生成"""
    df = df.copy()
    close = df['Close']
    volume = df['Volume']

    # 価格比率・差分
    df['Open_Close'] = df['Open'] / close
    df['High_Low'] = df['High'] / df['Low']
    df['High_Close'] = df['High'] / close
    df['Low_Close'] = df['Low'] / close
    df['Open_High'] = df['Open'] / df['High']
    df['Daily_Return'] = close.pct_change()
    df['Log_Return'] = np.log(close / close.shift(1))

    # 移動平均
    for w in [5, 10, 20]:
        df[f'SMA_{w}'] = close.rolling(window=w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    # RSI
    df['RSI'] = compute_rsi(close)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # Bollinger Bands
    df['20_SMA'] = close.rolling(window=20).mean()
    df['20_STD'] = close.rolling(window=20).std()
    df['Upper_Band'] = df['20_SMA'] + 2 * df['20_STD']
    df['Lower_Band'] = df['20_SMA'] - 2 * df['20_STD']
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['20_SMA']

    # ボリューム
    df['Volume_SMA_5'] = volume.rolling(window=5).mean()
    df['Volume_SMA_10'] = volume.rolling(window=10).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA_10']

    # 時間特徴量
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # 予測ターゲット（翌日終値）
    df['Target'] = close.shift(-1)

    df.dropna(inplace=True)
    return df


def create_features_with_sentiment(
    df: pd.DataFrame,
    news_impact_df: pd.DataFrame
) -> pd.DataFrame:
    """センチメント特徴量を既存特徴量に結合"""
    df_ext = create_features(df)

    if not news_impact_df.empty:
        news_impact_df.index = pd.to_datetime(news_impact_df.index).normalize()
        df_ext.index = pd.to_datetime(df_ext.index).normalize()
        df_ext = df_ext.merge(
            news_impact_df,
            left_index=True,
            right_index=True,
            how='left'
        )
        sentiment_cols = ['news_impact_score', 'news_impact_ma',
                          'news_volume', 'news_volume_ma']
        for col in sentiment_cols:
            if col in df_ext.columns:
                df_ext[col] = df_ext[col].fillna(0)
    else:
        df_ext['news_impact_score'] = 0.0
        df_ext['news_impact_ma'] = 0.0
        df_ext['news_volume'] = 0.0
        df_ext['news_volume_ma'] = 0.0

    df_ext.dropna(inplace=True)
    return df_ext


FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'DayOfWeek', 'DayOfMonth', 'Month', 'Year',
    'Open_Close', 'High_Low', 'High_Close', 'Low_Close', 'Open_High',
    'Daily_Return', 'Log_Return',
    'SMA_5', 'SMA_10', 'SMA_20', 'EMA_5', 'EMA_10', 'EMA_20',
    'RSI', 'MACD', 'Signal_Line', 'MACD_Hist',
    '20_SMA', '20_STD', 'Upper_Band', 'Lower_Band', 'BB_Width',
    'Volume_SMA_5', 'Volume_SMA_10', 'Volume_Ratio',
    'news_impact_score', 'news_impact_ma',
    'news_volume', 'news_volume_ma',
]


def preprocess_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    LightGBM用(2D)とTransformer用(3D)のデータを準備

    Returns:
        X_lgbm, y_lgbm: LightGBM用特徴量・ターゲット
        X_trans, y_trans: Transformer用シーケンス特徴量・ターゲット
        feat_scaler: 特徴量スケーラー
        tgt_scaler:  ターゲットスケーラー
    """
    # 利用可能な特徴量のみ選択
    available = [c for c in feature_cols if c in df.columns]
    X_raw = df[available].values.astype(np.float32)
    y_raw = df[target_col].values.reshape(-1, 1).astype(np.float32)

    feat_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()
    X_scaled = feat_scaler.fit_transform(X_raw)
    y_scaled = tgt_scaler.fit_transform(y_raw).ravel()

    # LightGBM用（2D）
    X_lgbm = X_scaled
    y_lgbm = y_scaled

    # Transformer用（3D シーケンス）
    X_trans, y_trans = [], []
    for i in range(sequence_length, len(X_scaled)):
        X_trans.append(X_scaled[i - sequence_length:i])
        y_trans.append(y_scaled[i])
    X_trans = np.array(X_trans, dtype=np.float32)
    y_trans = np.array(y_trans, dtype=np.float32)

    return X_lgbm, y_lgbm, X_trans, y_trans, feat_scaler, tgt_scaler
