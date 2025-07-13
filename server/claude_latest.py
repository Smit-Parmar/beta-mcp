"""
Stock Analysis MCP Server for Indian Markets using FastMCP
=========================================================

This MCP server provides comprehensive stock market data and analysis tools
optimized for Claude's analytical capabilities using FastMCP framework.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from enum import Enum

# FastMCP imports
from fastmcp import FastMCP
from pydantic import BaseModel, Field

class TimeFrame(Enum):
    """Supported timeframes for analysis"""
    ONE_MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"

class Exchange(Enum):
    """Supported exchanges"""
    NSE = "NSE"
    BSE = "BSE"

# Pydantic models for request validation
class OHLCVRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., 'RELIANCE', 'TCS')")
    exchange: str = Field("NSE", description="Exchange (NSE or BSE)")
    timeframe: str = Field("1d", description="Timeframe for data")
    period: str = Field("1y", description="Period for data")
    include_volume: bool = Field(True, description="Include volume data")

class TechnicalAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field("NSE", description="Exchange (NSE or BSE)")
    indicators: List[str] = Field(..., description="Technical indicators to calculate")
    period: str = Field("6mo", description="Period for analysis")

class FundamentalAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    exchange: str = Field("NSE", description="Exchange (NSE or BSE)")
    data_type: str = Field("info", description="Type of fundamental data")

class ScreenerCriteria(BaseModel):
    market_cap_min: Optional[float] = None
    market_cap_max: Optional[float] = None
    pe_ratio_max: Optional[float] = None
    dividend_yield_min: Optional[float] = None
    rsi_min: Optional[float] = None
    rsi_max: Optional[float] = None
    volume_min: Optional[float] = None

class MarketScreenerRequest(BaseModel):
    criteria: ScreenerCriteria = Field(..., description="Screening criteria")
    exchange: str = Field("NSE", description="Exchange (NSE or BSE)")
    limit: int = Field(20, description="Maximum number of results")

class PortfolioStock(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    exchange: str = "NSE"

class PortfolioAnalysisRequest(BaseModel):
    stocks: List[PortfolioStock] = Field(..., description="List of stocks in portfolio")
    benchmark: str = Field("^NSEI", description="Benchmark index for comparison")

class MarketSentimentRequest(BaseModel):
    index: str = Field("^NSEI", description="Market index to analyze")
    period: str = Field("1mo", description="Period for sentiment analysis")

# Initialize FastMCP
mcp = FastMCP("Stock Analysis Server")

class StockAnalysisHelper:
    """Helper class for stock analysis operations"""
    
    @staticmethod
    def format_symbol(symbol: str, exchange: str = "NSE") -> str:
        """Format symbol for yfinance"""
        exchange_suffix = ".NS" if exchange == "NSE" else ".BO"
        if not symbol.endswith((".NS", ".BO")):
            symbol += exchange_suffix
        return symbol
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return {
            "macd": macd,
            "signal": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        return {
            "upper": sma + (std * num_std),
            "middle": sma,
            "lower": sma - (std * num_std)
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=window).mean()

@mcp.tool()
async def get_ohlcv_data(request: OHLCVRequest) -> str:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for Indian stocks.
    Supports both NSE and BSE exchanges with various timeframes.
    """
    try:
        helper = StockAnalysisHelper()
        formatted_symbol = helper.format_symbol(request.symbol, request.exchange)
        ticker = yf.Ticker(formatted_symbol)
        
        # Get historical data
        hist = ticker.history(period=request.period, interval=request.timeframe)
        
        if hist.empty:
            return json.dumps({
                "error": f"No data found for {request.symbol} on {request.exchange}",
                "symbol": request.symbol,
                "exchange": request.exchange
            })
        
        # Calculate additional metrics
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else None
        
        # Prepare comprehensive data
        data_dict = {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "timeframe": request.timeframe,
            "period": request.period,
            "data_points": len(hist),
            "last_updated": datetime.now().isoformat(),
            "current_price": current_price,
            "previous_close": previous_close,
            "high_52w": float(hist['High'].max()),
            "low_52w": float(hist['Low'].min()),
            "price_change": current_price - previous_close if previous_close else 0,
            "price_change_pct": ((current_price - previous_close) / previous_close * 100) if previous_close else 0,
            "volume_avg": float(hist['Volume'].mean()) if request.include_volume else None,
            "volume_latest": float(hist['Volume'].iloc[-1]) if request.include_volume else None,
            "ohlcv_sample": hist.tail(10).to_dict('records')  # Last 10 records as sample
        }
        
        return json.dumps(data_dict, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error fetching data for {request.symbol}: {str(e)}",
            "symbol": request.symbol,
            "exchange": request.exchange
        })

@mcp.tool()
async def technical_analysis(request: TechnicalAnalysisRequest) -> str:
    """
    Perform comprehensive technical analysis on stock data.
    Supports indicators: SMA, EMA, RSI, MACD, BB (Bollinger Bands), VWAP, ATR.
    """
    try:
        print("Technical analysis request received")
        helper = StockAnalysisHelper()
        formatted_symbol = helper.format_symbol(request.symbol, request.exchange)
        ticker = yf.Ticker(formatted_symbol)
        hist = ticker.history(period=request.period)
        
        if hist.empty:
            return json.dumps({
                "error": f"No data found for {request.symbol} on {request.exchange}",
                "symbol": request.symbol,
                "exchange": request.exchange
            })
        
        results = {
            "symbol": request.symbol,
            "exchange": request.exchange,
            "analysis_date": datetime.now().isoformat(),
            "current_price": float(hist['Close'].iloc[-1]),
            "price_change": float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]),
            "price_change_pct": float(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100),
            "indicators": {}
        }
        
        # Calculate requested indicators
        for indicator in request.indicators:
            if indicator == "SMA":
                results["indicators"]["SMA_20"] = float(hist['Close'].rolling(window=20).mean().iloc[-1])
                results["indicators"]["SMA_50"] = float(hist['Close'].rolling(window=50).mean().iloc[-1])
                results["indicators"]["SMA_200"] = float(hist['Close'].rolling(window=200).mean().iloc[-1]) if len(hist) >= 200 else None
            
            elif indicator == "EMA":
                results["indicators"]["EMA_12"] = float(hist['Close'].ewm(span=12).mean().iloc[-1])
                results["indicators"]["EMA_26"] = float(hist['Close'].ewm(span=26).mean().iloc[-1])
                results["indicators"]["EMA_50"] = float(hist['Close'].ewm(span=50).mean().iloc[-1])
            
            elif indicator == "RSI":
                rsi = helper.calculate_rsi(hist['Close'])
                results["indicators"]["RSI"] = float(rsi.iloc[-1])
                results["indicators"]["RSI_Signal"] = "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral"
            
            elif indicator == "MACD":
                macd_data = helper.calculate_macd(hist['Close'])
                results["indicators"]["MACD"] = float(macd_data["macd"].iloc[-1])
                results["indicators"]["MACD_Signal"] = float(macd_data["signal"].iloc[-1])
                results["indicators"]["MACD_Histogram"] = float(macd_data["histogram"].iloc[-1])
                results["indicators"]["MACD_Crossover"] = "Bullish" if macd_data["macd"].iloc[-1] > macd_data["signal"].iloc[-1] else "Bearish"
            
            elif indicator == "BB":
                bb_data = helper.calculate_bollinger_bands(hist['Close'])
                results["indicators"]["BB_Upper"] = float(bb_data["upper"].iloc[-1])
                results["indicators"]["BB_Middle"] = float(bb_data["middle"].iloc[-1])
                results["indicators"]["BB_Lower"] = float(bb_data["lower"].iloc[-1])
                current_price = hist['Close'].iloc[-1]
                bb_position = "Above Upper" if current_price > bb_data["upper"].iloc[-1] else "Below Lower" if current_price < bb_data["lower"].iloc[-1] else "Within Bands"
                results["indicators"]["BB_Position"] = bb_position
            
            elif indicator == "VWAP":
                typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
                vwap = (typical_price * hist['Volume']).cumsum() / hist['Volume'].cumsum()
                results["indicators"]["VWAP"] = float(vwap.iloc[-1])
                results["indicators"]["VWAP_Signal"] = "Above VWAP" if hist['Close'].iloc[-1] > vwap.iloc[-1] else "Below VWAP"
            
            elif indicator == "ATR":
                atr = helper.calculate_atr(hist['High'], hist['Low'], hist['Close'])
                results["indicators"]["ATR"] = float(atr.iloc[-1])
                results["indicators"]["ATR_Volatility"] = "High" if atr.iloc[-1] > atr.mean() * 1.5 else "Low" if atr.iloc[-1] < atr.mean() * 0.5 else "Normal"
        
        return json.dumps(results, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error in technical analysis for {request.symbol}: {str(e)}",
            "symbol": request.symbol,
            "exchange": request.exchange
        })

@mcp.tool()
async def fundamental_analysis(request: FundamentalAnalysisRequest) -> str:
    """
    Get fundamental data for Indian stocks including company info, financials, balance sheet, etc.
    """
    try:
        helper = StockAnalysisHelper()
        formatted_symbol = helper.format_symbol(request.symbol, request.exchange)
        ticker = yf.Ticker(formatted_symbol)
        
        if request.data_type == "info":
            info = ticker.info
            # Filter and organize relevant fundamental data
            fundamental_data = {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "analysis_date": datetime.now().isoformat(),
                "company_info": {
                    "name": info.get('longName', 'N/A'),
                    "sector": info.get('sector', 'N/A'),
                    "industry": info.get('industry', 'N/A'),
                    "country": info.get('country', 'N/A'),
                    "website": info.get('website', 'N/A'),
                    "employees": info.get('fullTimeEmployees', 'N/A'),
                    "business_summary": info.get('longBusinessSummary', 'N/A')[:500] if info.get('longBusinessSummary') else 'N/A'
                },
                "valuation_metrics": {
                    "market_cap": info.get('marketCap', 'N/A'),
                    "enterprise_value": info.get('enterpriseValue', 'N/A'),
                    "pe_ratio": info.get('trailingPE', 'N/A'),
                    "forward_pe": info.get('forwardPE', 'N/A'),
                    "peg_ratio": info.get('pegRatio', 'N/A'),
                    "price_to_book": info.get('priceToBook', 'N/A'),
                    "price_to_sales": info.get('priceToSalesTrailing12Months', 'N/A'),
                    "ev_to_revenue": info.get('enterpriseToRevenue', 'N/A'),
                    "ev_to_ebitda": info.get('enterpriseToEbitda', 'N/A')
                },
                "financial_health": {
                    "total_revenue": info.get('totalRevenue', 'N/A'),
                    "revenue_growth": info.get('revenueGrowth', 'N/A'),
                    "gross_profit": info.get('grossProfits', 'N/A'),
                    "profit_margin": info.get('profitMargins', 'N/A'),
                    "operating_margin": info.get('operatingMargins', 'N/A'),
                    "net_income": info.get('netIncomeToCommon', 'N/A'),
                    "debt_to_equity": info.get('debtToEquity', 'N/A'),
                    "current_ratio": info.get('currentRatio', 'N/A'),
                    "quick_ratio": info.get('quickRatio', 'N/A')
                },
                "profitability_ratios": {
                    "roe": info.get('returnOnEquity', 'N/A'),
                    "roa": info.get('returnOnAssets', 'N/A'),
                    "roic": info.get('returnOnCapital', 'N/A'),
                    "gross_margin": info.get('grossMargins', 'N/A'),
                    "ebitda_margin": info.get('ebitdaMargins', 'N/A')
                },
                "dividend_info": {
                    "dividend_yield": info.get('dividendYield', 'N/A'),
                    "dividend_rate": info.get('dividendRate', 'N/A'),
                    "payout_ratio": info.get('payoutRatio', 'N/A'),
                    "ex_dividend_date": info.get('exDividendDate', 'N/A'),
                    "dividend_date": info.get('dividendDate', 'N/A')
                },
                "trading_metrics": {
                    "beta": info.get('beta', 'N/A'),
                    "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
                    "52_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
                    "avg_volume": info.get('averageVolume', 'N/A'),
                    "shares_outstanding": info.get('sharesOutstanding', 'N/A'),
                    "float_shares": info.get('floatShares', 'N/A'),
                    "institutional_ownership": info.get('heldByInstitutions', 'N/A')
                }
            }
            
        elif request.data_type == "financials":
            financials = ticker.financials
            fundamental_data = {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "data_type": "financials",
                "financials": financials.to_dict() if not financials.empty else {"error": "No financial data available"}
            }
            
        elif request.data_type == "balance_sheet":
            balance_sheet = ticker.balance_sheet
            fundamental_data = {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "data_type": "balance_sheet",
                "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {"error": "No balance sheet data available"}
            }
            
        elif request.data_type == "cash_flow":
            cash_flow = ticker.cashflow
            fundamental_data = {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "data_type": "cash_flow",
                "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {"error": "No cash flow data available"}
            }
            
        elif request.data_type == "earnings":
            earnings = ticker.earnings
            earnings_dates = ticker.calendar
            fundamental_data = {
                "symbol": request.symbol,
                "exchange": request.exchange,
                "data_type": "earnings",
                "earnings": earnings.to_dict() if not earnings.empty else {"error": "No earnings data available"},
                "earnings_calendar": earnings_dates.to_dict() if earnings_dates is not None and not earnings_dates.empty else {"error": "No earnings calendar available"}
            }
        
        return json.dumps(fundamental_data, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error in fundamental analysis for {request.symbol}: {str(e)}",
            "symbol": request.symbol,
            "exchange": request.exchange
        })

@mcp.tool()
async def market_screener(request: MarketScreenerRequest) -> str:
    """
    Screen stocks based on various criteria. 
    Note: This is a framework - integrate with NSE/BSE APIs for production use.
    """
    # Popular Indian stocks for demonstration
    demo_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "INFY", "ITC", 
        "SBIN", "BHARTIARTL", "ASIANPAINT", "MARUTI", "BAJFINANCE", "HCLTECH", 
        "KOTAKBANK", "LT", "AXISBANK", "TITAN", "NESTLEIND", "WIPRO", "ULTRACEMCO"
    ]
    
    screener_results = {
        "screening_criteria": request.criteria.dict(),
        "exchange": request.exchange,
        "analysis_date": datetime.now().isoformat(),
        "total_scanned": len(demo_stocks),
        "results": [],
        "note": "This is a demo screener. For production, integrate with NSE/BSE real-time data APIs."
    }
    
    try:
        helper = StockAnalysisHelper()
        
        for symbol in demo_stocks[:request.limit]:
            try:
                formatted_symbol = helper.format_symbol(symbol, request.exchange)
                ticker = yf.Ticker(formatted_symbol)
                info = ticker.info
                hist = ticker.history(period="3mo")
                
                if hist.empty:
                    continue
                
                # Calculate screening metrics
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 0)
                dividend_yield = info.get('dividendYield', 0) or 0
                
                # Calculate RSI
                rsi = helper.calculate_rsi(hist['Close']).iloc[-1] if len(hist) > 14 else 50
                
                # Calculate average volume
                avg_volume = hist['Volume'].mean()
                
                # Apply screening criteria
                passes_screen = True
                
                if request.criteria.market_cap_min and market_cap < request.criteria.market_cap_min:
                    passes_screen = False
                if request.criteria.market_cap_max and market_cap > request.criteria.market_cap_max:
                    passes_screen = False
                if request.criteria.pe_ratio_max and pe_ratio > request.criteria.pe_ratio_max:
                    passes_screen = False
                if request.criteria.dividend_yield_min and dividend_yield < request.criteria.dividend_yield_min:
                    passes_screen = False
                if request.criteria.rsi_min and rsi < request.criteria.rsi_min:
                    passes_screen = False
                if request.criteria.rsi_max and rsi > request.criteria.rsi_max:
                    passes_screen = False
                if request.criteria.volume_min and avg_volume < request.criteria.volume_min:
                    passes_screen = False
                
                if passes_screen:
                    screener_results["results"].append({
                        "symbol": symbol,
                        "current_price": float(hist['Close'].iloc[-1]),
                        "market_cap": market_cap,
                        "pe_ratio": pe_ratio,
                        "dividend_yield": dividend_yield,
                        "rsi": float(rsi),
                        "avg_volume": float(avg_volume),
                        "sector": info.get('sector', 'N/A')
                    })
                    
            except Exception as e:
                continue
        
        screener_results["total_matches"] = len(screener_results["results"])
        
        # Sort by market cap descending
        screener_results["results"] = sorted(
            screener_results["results"], 
            key=lambda x: x.get('market_cap', 0), 
            reverse=True
        )
        
        return json.dumps(screener_results, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error in market screening: {str(e)}",
            "exchange": request.exchange
        })

@mcp.tool()
async def portfolio_analysis(request: PortfolioAnalysisRequest) -> str:
    """
    Analyze portfolio performance with detailed metrics and benchmark comparison.
    """
    try:
        helper = StockAnalysisHelper()
        
        portfolio_data = {
            "portfolio_summary": {
                "total_stocks": len(request.stocks),
                "total_value": 0,
                "total_invested": 0,
                "unrealized_pnl": 0,
                "portfolio_return_pct": 0,
                "analysis_date": datetime.now().isoformat()
            },
            "individual_positions": [],
            "sector_allocation": {},
            "top_performers": [],
            "worst_performers": [],
            "benchmark_comparison": {}
        }
        
        # Analyze each position
        for stock in request.stocks:
            try:
                formatted_symbol = helper.format_symbol(stock.symbol, stock.exchange)
                ticker = yf.Ticker(formatted_symbol)
                
                # Get current price and basic info
                hist = ticker.history(period="1d")
                if hist.empty:
                    continue
                
                current_price = float(hist['Close'].iloc[-1])
                info = ticker.info
                
                # Calculate position metrics
                invested_amount = stock.quantity * stock.avg_price
                current_value = stock.quantity * current_price
                pnl = current_value - invested_amount
                pnl_pct = (pnl / invested_amount) * 100 if invested_amount > 0 else 0
                
                # Get 1-year performance for more context
                year_hist = ticker.history(period="1y")
                year_return = 0
                if not year_hist.empty and len(year_hist) > 1:
                    year_return = ((year_hist['Close'].iloc[-1] / year_hist['Close'].iloc[0]) - 1) * 100
                
                position_data = {
                    "symbol": stock.symbol,
                    "exchange": stock.exchange,
                    "quantity": stock.quantity,
                    "avg_price": stock.avg_price,
                    "current_price": current_price,
                    "invested_amount": invested_amount,
                    "current_value": current_value,
                    "unrealized_pnl": pnl,
                    "return_pct": pnl_pct,
                    "year_return_pct": year_return,
                    "weight_in_portfolio": 0,  # Will be calculated later
                    "sector": info.get('sector', 'Unknown'),
                    "company_name": info.get('longName', stock.symbol)
                }
                
                portfolio_data["individual_positions"].append(position_data)
                portfolio_data["portfolio_summary"]["total_value"] += current_value
                portfolio_data["portfolio_summary"]["total_invested"] += invested_amount
                
                # Track sector allocation
                sector = info.get('sector', 'Unknown')
                if sector not in portfolio_data["sector_allocation"]:
                    portfolio_data["sector_allocation"][sector] = 0
                portfolio_data["sector_allocation"][sector] += current_value
                
            except Exception as e:
                portfolio_data["individual_positions"].append({
                    "symbol": stock.symbol,
                    "error": str(e)
                })
        
        # Calculate portfolio totals and weights
        if portfolio_data["portfolio_summary"]["total_invested"] > 0:
            portfolio_data["portfolio_summary"]["unrealized_pnl"] = (
                portfolio_data["portfolio_summary"]["total_value"] - 
                portfolio_data["portfolio_summary"]["total_invested"]
            )
            portfolio_data["portfolio_summary"]["portfolio_return_pct"] = (
                portfolio_data["portfolio_summary"]["unrealized_pnl"] / 
                portfolio_data["portfolio_summary"]["total_invested"]
            ) * 100
            
            # Calculate individual position weights
            for position in portfolio_data["individual_positions"]:
                if "current_value" in position:
                    position["weight_in_portfolio"] = (
                        position["current_value"] / portfolio_data["portfolio_summary"]["total_value"]
                    ) * 100
        
        # Calculate sector allocation percentages
        total_value = portfolio_data["portfolio_summary"]["total_value"]
        if total_value > 0:
            for sector in portfolio_data["sector_allocation"]:
                portfolio_data["sector_allocation"][sector] = {
                    "value": portfolio_data["sector_allocation"][sector],
                    "percentage": (portfolio_data["sector_allocation"][sector] / total_value) * 100
                }
        
        # Find top and worst performers
        valid_positions = [p for p in portfolio_data["individual_positions"] if "return_pct" in p]
        if valid_positions:
            portfolio_data["top_performers"] = sorted(
                valid_positions, key=lambda x: x["return_pct"], reverse=True
            )[:3]
            portfolio_data["worst_performers"] = sorted(
                valid_positions, key=lambda x: x["return_pct"]
            )[:3]
        
        # Benchmark comparison (simplified)
        try:
            benchmark_ticker = yf.Ticker(request.benchmark)
            benchmark_hist = benchmark_ticker.history(period="1y")
            if not benchmark_hist.empty:
                benchmark_return = ((benchmark_hist['Close'].iloc[-1] / benchmark_hist['Close'].iloc[0]) - 1) * 100
                portfolio_data["benchmark_comparison"] = {
                    "benchmark_symbol": request.benchmark,
                    "benchmark_return_1y": benchmark_return,
                    "portfolio_vs_benchmark": portfolio_data["portfolio_summary"]["portfolio_return_pct"] - benchmark_return,
                    "outperforming": portfolio_data["portfolio_summary"]["portfolio_return_pct"] > benchmark_return
                }
        except Exception as e:
            portfolio_data["benchmark_comparison"] = {
                "error": f"Could not fetch benchmark data: {str(e)}"
            }
        
        return json.dumps(portfolio_data, indent=2, default=str)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error in portfolio analysis: {str(e)}"
        })

@mcp.tool()
async def market_sentiment(request: MarketSentimentRequest) -> str:
    """
    Analyze market sentiment for major Indian indices with comprehensive metrics.
    """
    try:
        ticker = yf.Ticker(request.index)
        hist = ticker.history(period=request.period)
        
        if hist.empty:
            return json.dumps({
                "error": f"No data found for index {request.index}",
                "index": request.index
            })
        
        # Calculate comprehensive sentiment metrics
                # Calculate comprehensive sentiment metrics
        close_prices = hist["Close"]
        high_prices = hist["High"]
        low_prices = hist["Low"]
        volume = hist["Volume"]

        # Price performance metrics
        price_change = float(close_prices.iloc[-1] - close_prices.iloc[0])
        price_change_pct = (price_change / close_prices.iloc[0]) * 100

        # Volatility metrics
        returns = close_prices.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252) * 100)  # Annualized volatility in %

        # Momentum metric (14-day RSI)
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = float(rsi.iloc[-1])

        # Moving averages
        sma_20 = float(close_prices.rolling(window=20).mean().iloc[-1])
        sma_50 = float(close_prices.rolling(window=50).mean().iloc[-1])
        sma_200 = float(close_prices.rolling(window=200).mean().iloc[-1]) if len(close_prices) >= 200 else None

        sentiment_data = {
            "index": request.index,
            "period": request.period,
            "analysis_date": datetime.now().isoformat(),
            "current_price": float(close_prices.iloc[-1]),
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "annualized_volatility_pct": volatility,
            "rsi_14": latest_rsi,
            "rsi_signal": "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral",
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "moving_average_signal": "Bullish" if sma_20 > sma_50 > sma_200 else "Bearish" if sma_20 < sma_50 < sma_200 else "Neutral",
            "high_52w": float(high_prices.max()),
            "low_52w": float(low_prices.min()),
            "volume_avg": float(volume.mean()),
            "volume_latest": float(volume.iloc[-1])
        }

        return json.dumps(sentiment_data, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": f"Error in market sentiment analysis: {str(e)}",
            "index": request.index
        })

if __name__ == "__main__":
    mcp.run()
    # mcp.run(transport="streamable-http")