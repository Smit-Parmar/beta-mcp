"""
This server is used to get stock information using the yfinance library.
Provides comprehensive stock data including prices, financials, and company info.
"""

import yfinance as yf
from fastmcp import FastMCP
from typing import Dict, Any, List, Optional
import pandas as pd

mcp = FastMCP(name="stock-server")

@mcp.tool()
def get_stock_price(symbol: str) -> float:
    """
    Get the current stock price of a given stock symbol
    """
    stock = yf.Ticker(symbol)
    return stock.history(period="1d")["Close"].iloc[-1]

@mcp.tool()
def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive stock information including company details, financial metrics, and market data
    """
    stock = yf.Ticker(symbol)
    info = stock.info
    
    # Extract key information
    return {
        "symbol": symbol,
        "company_name": info.get('longName', 'N/A'),
        "sector": info.get('sector', 'N/A'),
        "industry": info.get('industry', 'N/A'),
        "current_price": info.get('currentPrice', 0),
        "market_cap": info.get('marketCap', 0),
        "pe_ratio": info.get('trailingPE', 0),
        "dividend_yield": info.get('dividendYield', 0),
        "volume": info.get('volume', 0),
        "avg_volume": info.get('averageVolume', 0),
        "day_high": info.get('dayHigh', 0),
        "day_low": info.get('dayLow', 0),
        "year_high": info.get('fiftyTwoWeekHigh', 0),
        "year_low": info.get('fiftyTwoWeekLow', 0),
        "beta": info.get('beta', 0),
        "description": info.get('longBusinessSummary', 'N/A')[:200] + "..." if info.get('longBusinessSummary') else 'N/A'
    }

@mcp.tool()
def get_historical_data(symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
    """
    Get historical stock data for a given symbol
    Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period, interval=interval)
    data = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "data_points": len(hist),
        "latest_close": float(hist["Close"].iloc[-1]) if not hist.empty else 0,
        "latest_volume": int(hist["Volume"].iloc[-1]) if not hist.empty else 0,
        "price_change": float(hist["Close"].iloc[-1] - hist["Close"].iloc[0]) if len(hist) > 1 else 0,
        "price_change_percent": float(((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100) if len(hist) > 1 else 0
    }
    print(data)
    # Convert to dict for JSON serialization
    return data

@mcp.tool()
def get_financial_metrics(symbol: str) -> Dict[str, Any]:
    """
    Get key financial metrics and ratios for a stock
    """
    stock = yf.Ticker(symbol)
    info = stock.info
    
    return {
        "symbol": symbol,
        "revenue": info.get('totalRevenue', 0),
        "net_income": info.get('netIncomeToCommon', 0),
        "profit_margin": info.get('profitMargins', 0),
        "debt_to_equity": info.get('debtToEquity', 0),
        "current_ratio": info.get('currentRatio', 0),
        "return_on_equity": info.get('returnOnEquity', 0),
        "return_on_assets": info.get('returnOnAssets', 0),
        "price_to_book": info.get('priceToBook', 0),
        "price_to_sales": info.get('priceToSalesTrailing12Months', 0),
        "enterprise_value": info.get('enterpriseValue', 0),
        "free_cash_flow": info.get('freeCashflow', 0)
    }

@mcp.tool()
def get_dividend_info(symbol: str) -> Dict[str, Any]:
    """
    Get dividend information for a stock
    """
    stock = yf.Ticker(symbol)
    dividends = stock.dividends
    
    if dividends.empty:
        return {
            "symbol": symbol,
            "has_dividends": False,
            "message": "No dividend data available"
        }
    
    return {
        "symbol": symbol,
        "has_dividends": True,
        "annual_dividend": float(dividends.tail(4).sum()) if len(dividends) >= 4 else float(dividends.sum()),
        "last_dividend": float(dividends.iloc[-1]),
        "last_dividend_date": dividends.index[-1].strftime('%Y-%m-%d'),
        "dividend_yield": stock.info.get('dividendYield', 0),
        "payout_ratio": stock.info.get('payoutRatio', 0)
    }

@mcp.tool()
def compare_stocks(symbols: List[str]) -> Dict[str, Any]:
    """
    Compare multiple stocks side by side
    """
    results = {}
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            
            results[symbol] = {
                "company_name": info.get('longName', 'N/A'),
                "current_price": float(current_price),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "dividend_yield": info.get('dividendYield', 0),
                "volume": info.get('volume', 0),
                "sector": info.get('sector', 'N/A')
            }
        except Exception as e:
            results[symbol] = {"error": str(e)}
    
    return results

@mcp.tool()
def get_historical_data_n_days(symbol: str, days: int) -> Dict[str, Any]:
    """
    Get historical stock data for a given symbol for the last n days
    """
    stock = yf.Ticker(symbol)
    
    # Calculate the period based on days
    if days <= 5:
        period = "5d"
    elif days <= 30:
        period = "1mo"
    elif days <= 90:
        period = "3mo"
    elif days <= 180:
        period = "6mo"
    elif days <= 365:
        period = "1y"
    else:
        period = "max"
    
    hist = stock.history(period=period, interval="1d")
    
    # Filter to get only the last n days
    if len(hist) > days:
        hist = hist.tail(days)
    
    # Prepare the data
    data = {
        "symbol": symbol,
        "requested_days": days,
        "actual_days": len(hist),
        "data_points": len(hist),
        "latest_close": float(hist["Close"].iloc[-1]) if not hist.empty else 0,
        "latest_volume": int(hist["Volume"].iloc[-1]) if not hist.empty else 0,
        "price_change": float(hist["Close"].iloc[-1] - hist["Close"].iloc[0]) if len(hist) > 1 else 0,
        "price_change_percent": float(((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100) if len(hist) > 1 else 0,
        "daily_data": []
    }
    
    # Add daily data points
    for date, row in hist.iterrows():
        daily_point = {
            "date": date.strftime('%Y-%m-%d'),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        }
        data["daily_data"].append(daily_point)
    
    return data


if __name__ == "__main__":
    mcp.run()