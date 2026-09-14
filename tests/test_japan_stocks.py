import pytest
import os
import json
from src.japan_stocks import (
    load_japan_stocks,
    get_sectors,
    get_stocks_by_sector,
    generate_japan_stock_aliases,
    search_stocks
)


def test_load_japan_stocks():
    stocks = load_japan_stocks()
    assert isinstance(stocks, list)
    assert len(stocks) == 100
    first = stocks[0]
    assert "code" in first
    assert "ticker" in first
    assert "name" in first
    assert "sector" in first
    assert "focus" in first


def test_get_sectors():
    sectors = get_sectors()
    assert len(sectors) == 8
    assert "半導体・電子部品・ハイテク" in sectors
    assert "防衛・インフラ重工・機械" in sectors
    assert "総合商社" in sectors


def test_get_stocks_by_sector():
    trading_stocks = get_stocks_by_sector("総合商社")
    assert len(trading_stocks) == 5
    codes = [s["code"] for s in trading_stocks]
    assert "8058" in codes  # 三菱商事
    assert "8031" in codes  # 三井物産


def test_generate_japan_stock_aliases():
    aliases = generate_japan_stock_aliases()
    # Code resolution
    assert aliases.get("8035") == "8035.T"
    assert aliases.get("7203") == "7203.T"
    # Name resolution
    assert aliases.get("東京エレクトロン") == "8035.T"
    assert aliases.get("アドバンテスト") == "6857.T"
    # Parentheses decomposition
    assert aliases.get("コマツ") == "6301.T"
    assert aliases.get("小松製作所") == "6301.T"


def test_search_stocks():
    res = search_stocks("半導体")
    assert len(res) > 0
    res_code = search_stocks("8035")
    assert len(res_code) == 1
    assert res_code[0]["name"] == "東京エレクトロン"
