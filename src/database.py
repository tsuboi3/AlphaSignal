"""
データベース管理モジュール
SQLiteを使用したデータの永続化、欠損値チェック、重複防止。
"""

import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

# モジュールレベルでの環境変数読み込み（独立実行用）
load_dotenv()

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        # 環境変数 DB_NAME があれば優先、なければ引数、それもなければデフォルト
        self.db_path = db_path or os.getenv("DB_NAME", "alphasignal.db")
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """テーブルの初期化"""
        with self._get_connection() as conn:
            # 株価データテーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            # ニュースインパクト指数テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_impact (
                    ticker TEXT,
                    date TEXT,
                    news_impact_score REAL,
                    news_impact_ma REAL,
                    news_volume REAL,
                    news_volume_ma REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.commit()

    def save_stocks(self, ticker: str, df: pd.DataFrame):
        """株価データを保存（重複排除、欠損確認）"""
        if df.empty:
            return

        # 欠損値の確認（行ごと削除またはログ出力）
        initial_len = len(df)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        if len(df) < initial_len:
            print(f"  ⚠ 警告: {initial_len - len(df)} 件の欠損データを除外しました。")

        # データの整形
        save_df = df.copy()
        save_df['ticker'] = ticker
        save_df['date'] = save_df.index.strftime('%Y-%m-%d')
        save_df = save_df.reset_index(drop=True)
        
        # カラム名の小文字化
        save_df = save_df[['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        save_df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']

        with self._get_connection() as conn:
            # 既存データとの重複を考慮した挿入 (INSERT OR REPLACE)
            save_df.to_sql('stocks', conn, if_exists='append', index=False, method=self._insert_or_replace)
            conn.commit()

    def save_news_impact(self, ticker: str, df: pd.DataFrame):
        """ニュースインパクト指数を保存"""
        if df.empty:
            return

        df = df.dropna()
        save_df = df.copy()
        save_df['ticker'] = ticker
        save_df['date'] = save_df.index.strftime('%Y-%m-%d')
        save_df = save_df.reset_index(drop=True)
        
        # 必要なカラムのみ
        cols = ['ticker', 'date', 'news_impact_score', 'news_impact_ma', 'news_volume', 'news_volume_ma']
        save_df = save_df[cols]

        with self._get_connection() as conn:
            save_df.to_sql('news_impact', conn, if_exists='append', index=False, method=self._insert_or_replace)
            conn.commit()

    def load_stocks(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """DBから株価データをロード"""
        query = "SELECT * FROM stocks WHERE ticker = ? AND date BETWEEN ? AND ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            # カラム名を大文字に戻す（既存コード互換性）
            df.columns = [col.capitalize() if col in ['open', 'high', 'low', 'close', 'volume'] else col for col in df.columns]
            return df.drop(columns=['ticker'])
        return pd.DataFrame()

    def load_news_impact(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """DBからニュースインパクト指数をロード"""
        query = "SELECT * FROM news_impact WHERE ticker = ? AND date BETWEEN ? AND ?"
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df.drop(columns=['ticker'])
        return pd.DataFrame()

    def _insert_or_replace(self, table, conn, keys, data_iter):
        """pandas to_sql 用の INSERT OR REPLACE 実行関数"""
        from sqlite3 import IntegrityError
        columns = ', '.join(keys)
        placeholders = ', '.join(['?'] * len(keys))
        sql = f"INSERT OR REPLACE INTO {table.name} ({columns}) VALUES ({placeholders})"
        conn.executemany(sql, data_iter)

    def add_column_if_not_exists(self, table_name: str, column_name: str, column_type: str):
        """
        新しいコラムを既存のテーブルに追加する（後方互換性維持）。
        """
        with self._get_connection() as conn:
            cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = [info[1] for info in cursor.fetchall()]
            if column_name not in columns:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                print(f"  → テーブル {table_name} にコラム {column_name} を追加しました。")
                conn.commit()

# シングルトン的に使用
db_manager = DatabaseManager()
