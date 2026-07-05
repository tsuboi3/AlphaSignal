"""
ニュースセンチメントモジュール
FinBERTによる感情分析とニュースインパクト指数の計算
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

from src.database import db_manager

# 環境変数の読み込み
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    ニュースデータ取得関数
    実運用ではBloomberg API / News API / Alpha Vantageを使用する
    現在はダミーデータを生成
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 営業日のみ
    news_data = []
    np.random.seed(42)

    for date in dates:
        num_articles = np.random.randint(0, 6)
        for _ in range(num_articles):
            # ダミーセンチメントスコア（実際はFinBERTで算出）
            sentiment_score = np.random.uniform(-1, 1)
            relevance_weight = np.random.uniform(0.5, 1.0)
            news_data.append({
                'date': date,
                'headline': f'{ticker} related news on {date.strftime("%Y-%m-%d")}',
                'text': 'Dummy article text for demonstration.',
                'sentiment_score_raw': sentiment_score,
                'relevance_weight': relevance_weight,
            })

    df = pd.DataFrame(news_data)
    if df.empty:
        df = pd.DataFrame(columns=['date', 'headline', 'text',
                                   'sentiment_score_raw', 'relevance_weight'])
    return df


def get_finbert_sentiment(text: str) -> float:
    """
    FinBERTによるセンチメントスコア算出
    実装例（要: transformers, torch ライブラリ）:

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive = predictions[:, 0].item()
        negative = predictions[:, 1].item()
        return positive - negative  # [-1, 1] のスコア

    現在はダミー値を返す
    """
    return np.random.uniform(-1, 1)


def calculate_news_impact_score(
    news_df: pd.DataFrame,
    decay_rate: float = 0.1,
    window_size: int = 5
) -> pd.DataFrame:
    """
    ニュースインパクト指数 (NIS) の計算
    NIS_t = Σ w_i,t * S_i,t * D(days_ago_i,t)

    Args:
        news_df: ニュースデータフレーム
        decay_rate: 時間減衰率
        window_size: 移動平均のウィンドウサイズ（日）

    Returns:
        日次ニュースインパクト指数のデータフレーム
    """
    if news_df.empty:
        return pd.DataFrame(columns=['news_impact_score', 'news_volume',
                                     'news_impact_ma'])

    news_df = news_df.copy()
    news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()
    news_df = news_df.sort_values('date')

    # ボリューム加重 × 関連性加重 × センチメントスコア の日次集約
    news_df['weighted_score'] = (
        news_df['relevance_weight'] * news_df['sentiment_score_raw']
    )

    daily = news_df.groupby('date').agg(
        raw_score=('weighted_score', 'sum'),
        article_count=('sentiment_score_raw', 'count')
    ).reset_index()
    daily.set_index('date', inplace=True)

    # ボリューム加重正規化
    daily['news_impact_score'] = (
        daily['raw_score'] / daily['article_count'].clip(lower=1)
    )

    # 指数関数的時間減衰を適用した移動平均
    weights = np.array([
        np.exp(-decay_rate * i) for i in range(window_size - 1, -1, -1)
    ])
    weights /= weights.sum()

    daily['news_impact_ma'] = (
        daily['news_impact_score']
        .rolling(window=window_size, min_periods=1)
        .apply(lambda x: np.dot(x[-len(weights):],
                                weights[-len(x):] / weights[-len(x):].sum()),
               raw=True)
    )

    # ニュース量特徴量
    daily['news_volume'] = daily['article_count']
    daily['news_volume_ma'] = daily['news_volume'].rolling(
        window=window_size, min_periods=1
    ).mean()

    # DBに保存（呼び出し側でtickerがわかる必要があるため、引数にtickerを追加するか、戻り値を受け取ってから保存するか）
    # ここでは既存の引数を変えたくないので、save_news_impactの呼び出しはメインエントリ(main.py等)で行うように設計を微調整します。
    # または、この関数にtickerを渡せるように変更します。
    
    return daily[['news_impact_score', 'news_impact_ma',
                  'news_volume', 'news_volume_ma']]

def calculate_news_impact_with_db(
    ticker: str,
    news_df: pd.DataFrame,
    decay_rate: float = 0.1,
    window_size: int = 5
) -> pd.DataFrame:
    """DB保存機能付きの計算関数"""
    daily = calculate_news_impact_score(news_df, decay_rate, window_size)
    if not daily.empty:
        db_manager.save_news_impact(ticker, daily)
    return daily
