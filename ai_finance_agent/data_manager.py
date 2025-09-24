"""
Data Manager for accessing ML model data and stock information
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import threading
from functools import wraps

# Add the parent directory to path to import ML models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN'))

from config import DATA_DIR, MODELS_DIR, ALPHA_VANTAGE_API_KEY, NEWS_API_KEY, SUPPORTED_STOCKS

# Rate limiting decorator
def rate_limit(calls_per_second=1):
    """Decorator to limit API calls per second"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=3, base_delay=1):
    """Decorator to retry function calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Too Many Requests
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Rate limited. Waiting {delay} seconds before retry {attempt + 1}/{max_retries}")
                            time.sleep(delay)
                            continue
                    raise e
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Error occurred: {e}. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    raise e
            return None
        return wrapper
    return decorator

class DataManager:
    """Manages access to stock data, ML models, and news sentiment"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas') if ALPHA_VANTAGE_API_KEY else None
        self._cache = {}  # Simple in-memory cache
        self._cache_timeout = 300  # 5 minutes cache timeout
        self._lock = threading.Lock()  # Thread-safe cache access
        
    def _get_from_cache(self, key: str):
        """Get data from cache if still valid"""
        with self._lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if time.time() - timestamp < self._cache_timeout:
                    return data
                else:
                    del self._cache[key]  # Remove expired cache
            return None
    
    def _set_cache(self, key: str, data):
        """Set data in cache"""
        with self._lock:
            self._cache[key] = (data, time.time())
    
    @rate_limit(calls_per_second=0.5)  # Limit to 1 call every 2 seconds
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get stock data from yfinance with multiple methods and caching"""
        cache_key = f"stock_data_{symbol}_{period}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            print(f"Returning cached stock data for {symbol}")
            return cached_data
            
        try:
            print(f"Fetching stock data for {symbol} (rate limited)...")
            
            # Method 1: Direct yfinance download (most reliable)
            try:
                data = yf.download(symbol, period=period, progress=False)
                if not data.empty:
                    print(f"Success with yf.download for {symbol}")
                    self._set_cache(cache_key, data)
                    return data
            except Exception as e:
                print(f"yf.download failed for {symbol}: {e}")
            
            # Method 2: Ticker with session
            try:
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                ticker = yf.Ticker(symbol, session=session)
                data = ticker.history(period=period)
                
                if not data.empty:
                    print(f"Success with ticker.history for {symbol}")
                    self._set_cache(cache_key, data)
                    return data
            except Exception as e:
                print(f"Ticker method failed for {symbol}: {e}")
            
            # Method 3: Try Alpha Vantage for historical data
            if self.alpha_vantage and ALPHA_VANTAGE_API_KEY:
                try:
                    print(f"Trying Alpha Vantage historical data for {symbol}")
                    data, meta_data = self.alpha_vantage.get_daily(symbol=symbol, outputsize='full')
                    if not data.empty:
                        # Convert Alpha Vantage format to match yfinance
                        data = data.rename(columns={
                            '1. open': 'Open',
                            '2. high': 'High', 
                            '3. low': 'Low',
                            '4. close': 'Close',
                            '5. volume': 'Volume'
                        })
                        print(f"Success with Alpha Vantage for {symbol}")
                        self._set_cache(cache_key, data)
                        return data
                except Exception as e:
                    print(f"Alpha Vantage failed for {symbol}: {e}")
            
            # Method 4: Try local CSV data
            csv_data = self.get_historical_data(symbol)
            if not csv_data.empty:
                print(f"Using local CSV data for {symbol}")
                # Convert CSV format to match expected format
                if 'date' in csv_data.columns:
                    csv_data = csv_data.set_index('date')
                self._set_cache(cache_key, csv_data)
                return csv_data
            
            print(f"All methods failed for stock data: {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """Get historical data from CSV files"""
        try:
            csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                data['date'] = pd.to_datetime(data['date'])
                return data
            else:
                print(f"CSV file not found for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error reading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    @retry_with_backoff(max_retries=2, base_delay=1)
    def get_news_sentiment(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get news sentiment for a stock symbol with caching"""
        cache_key = f"news_sentiment_{symbol}_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Use the existing sentiment analysis from your codebase
            url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                print(f"News API error: {response.status_code}")
                return []
            
            data = response.json()
            articles = data.get("articles", [])
            
            sentiments = []
            for article in articles[:20]:  # Limit to 20 articles
                title = article.get("title", "")
                if title:
                    sentiment_score = self.sentiment_analyzer.polarity_scores(title)
                    sentiments.append({
                        "title": title,
                        "sentiment": sentiment_score["compound"],
                        "published": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", "")
                    })
            
            # Cache the result
            self._set_cache(cache_key, sentiments)
            return sentiments
            
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {e}")
            return []
    
    @rate_limit(calls_per_second=0.5)  # Limit to 1 call every 2 seconds  
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_stock_price(self, symbol: str) -> Dict:
        """Get current stock price and basic info with multiple fallback methods"""
        cache_key = f"stock_price_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            print(f"Returning cached data for {symbol}")
            return cached_data
            
        try:
            print(f"Fetching stock price for {symbol} (rate limited)...")
            
            # Method 1: Direct Yahoo Finance API (bypass yfinance completely)
            try:
                print(f"Trying direct Yahoo Finance API for {symbol}")
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('chart') and data['chart'].get('result'):
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        
                        current_price = meta.get('regularMarketPrice', 0)
                        prev_close = meta.get('previousClose', current_price)
                        
                        if current_price > 0:
                            change = current_price - prev_close
                            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
                            
                            result_data = {
                                "symbol": symbol,
                                "current_price": float(current_price),
                                "change": float(change),
                                "change_percent": float(change_percent),
                                "volume": int(meta.get('regularMarketVolume', 0)),
                                "market_cap": 0,
                                "pe_ratio": 0,
                                "last_updated": datetime.now().isoformat(),
                                "data_source": "yahoo_direct_api",
                                "status": "success"
                            }
                            
                            self._set_cache(cache_key, result_data)
                            print(f"Success with direct Yahoo API for {symbol}")
                            return result_data
                        
            except Exception as e:
                print(f"Direct Yahoo API failed for {symbol}: {e}")
            
            # Method 2: Try yfinance with session (only if direct API failed)
            try:
                print(f"Trying yfinance with custom session for {symbol}")
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                
                ticker = yf.Ticker(symbol, session=session)
                
                # Try multiple approaches with yfinance
                methods = [
                    ("download", lambda: yf.download(symbol, period="5d", interval="1d", progress=False)),
                    ("history_5d", lambda: ticker.history(period="5d")),
                    ("history_1mo", lambda: ticker.history(period="1mo"))
                ]
                
                hist = None
                successful_method = None
                
                for method_name, method_func in methods:
                    try:
                        print(f"Trying method: {method_name}")
                        temp_hist = method_func()
                        if not temp_hist.empty:
                            hist = temp_hist
                            successful_method = method_name
                            print(f"Success with method: {method_name}")
                            break
                    except Exception as e:
                        print(f"Method {method_name} failed: {e}")
                        continue
                
                if hist is not None and not hist.empty:
                    # Get the most recent data
                    current_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[-1] if len(hist) == 1 else hist['Open'].iloc[0]
                    
                    # Try to get volume, handle missing volume data
                    volume = 0
                    if 'Volume' in hist.columns and not pd.isna(hist['Volume'].iloc[-1]):
                        volume = int(hist['Volume'].iloc[-1])
                    
                    # Try to get additional info with better error handling
                    market_cap = 0
                    pe_ratio = 0
                    try:
                        info = ticker.info
                        if info and isinstance(info, dict):
                            market_cap = info.get("marketCap", 0)
                            pe_ratio = info.get("trailingPE", 0)
                    except Exception as info_error:
                        print(f"Could not fetch additional info for {symbol}: {info_error}")
                    
                    result = {
                        "symbol": symbol,
                        "current_price": float(current_price),
                        "change": float(current_price - open_price),
                        "change_percent": float((current_price - open_price) / open_price * 100) if open_price != 0 else 0,
                        "volume": volume,
                        "market_cap": market_cap,
                        "pe_ratio": pe_ratio,
                        "last_updated": datetime.now().isoformat(),
                        "data_source": f"yfinance_{successful_method}",
                        "status": "success"
                    }
                    
                    # Cache the successful result
                    self._set_cache(cache_key, result)
                    return result
                
            except Exception as e:
                print(f"YFinance methods failed for {symbol}: {e}")
            
            # If all primary methods failed, try fallback data
            print(f"All primary methods failed for {symbol}, trying fallback data...")
            fallback_data = self._get_fallback_data(symbol)
            if fallback_data:
                return fallback_data
            
            # Return cached data if available, even if expired
            with self._lock:
                if cache_key in self._cache:
                    data, _ = self._cache[cache_key]
                    print(f"Returning stale cached data for {symbol}")
                    data["status"] = "stale_cache"
                    return data
            
            # Last resort: return error info
            return {
                "symbol": symbol,
                "error": f"All data sources failed for {symbol}",
                "status": "error",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Unexpected error fetching stock price for {symbol}: {e}")
            # Try fallback data
            fallback_data = self._get_fallback_data(symbol)
            if fallback_data:
                return fallback_data
            
            # Return cached data if available, even if expired
            with self._lock:
                if cache_key in self._cache:
                    data, _ = self._cache[cache_key]
                    print(f"Returning stale cached data for {symbol}")
                    data["status"] = "stale_cache"
                    return data
            
            # Last resort: return error info
            return {
                "symbol": symbol,
                "error": str(e),
                "status": "error",
                "last_updated": datetime.now().isoformat()
            }
    
    def _get_fallback_data(self, symbol: str) -> Dict:
        """Get fallback data from alternative sources or local data"""
        try:
            # Method 1: Try Alpha Vantage if available
            if self.alpha_vantage and ALPHA_VANTAGE_API_KEY:
                try:
                    print(f"Trying Alpha Vantage for {symbol}")
                    data, meta_data = self.alpha_vantage.get_quote_endpoint(symbol=symbol)
                    if not data.empty:
                        price = float(data['05. price'].iloc[0])
                        change = float(data['09. change'].iloc[0])
                        
                        # Fix percentage parsing - remove '%' and convert to float
                        change_percent_str = str(data['10. change percent'].iloc[0]).replace('%', '').strip()
                        change_percent = float(change_percent_str)
                        
                        return {
                            "symbol": symbol,
                            "current_price": price,
                            "change": change,
                            "change_percent": change_percent,
                            "volume": 0,
                            "market_cap": 0,
                            "pe_ratio": 0,
                            "last_updated": datetime.now().isoformat(),
                            "data_source": "alpha_vantage",
                            "status": "fallback_success"
                        }
                except Exception as av_error:
                    print(f"Alpha Vantage failed for {symbol}: {av_error}")
            
            # Method 2: Try Yahoo Finance REST API directly (bypass yfinance library)
            try:
                print(f"Trying direct Yahoo Finance API for {symbol}")
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('chart') and data['chart'].get('result'):
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        
                        current_price = meta.get('regularMarketPrice', 0)
                        prev_close = meta.get('previousClose', current_price)
                        
                        if current_price > 0:
                            change = current_price - prev_close
                            change_percent = (change / prev_close * 100) if prev_close != 0 else 0
                            
                            return {
                                "symbol": symbol,
                                "current_price": float(current_price),
                                "change": float(change),
                                "change_percent": float(change_percent),
                                "volume": int(meta.get('regularMarketVolume', 0)),
                                "market_cap": 0,
                                "pe_ratio": 0,
                                "last_updated": datetime.now().isoformat(),
                                "data_source": "yahoo_direct_api",
                                "status": "fallback_success"
                            }
                        
            except Exception as yahoo_error:
                print(f"Direct Yahoo API failed for {symbol}: {yahoo_error}")
            
            # Method 3: Try local CSV data
            csv_data = self.get_historical_data(symbol)
            if not csv_data.empty:
                print(f"Using local CSV data for {symbol}")
                latest_price = csv_data['close'].iloc[-1] if 'close' in csv_data.columns else csv_data.iloc[-1, -1]
                prev_price = csv_data['close'].iloc[-2] if len(csv_data) > 1 and 'close' in csv_data.columns else latest_price
                
                return {
                    "symbol": symbol,
                    "current_price": float(latest_price),
                    "change": float(latest_price - prev_price),
                    "change_percent": float((latest_price - prev_price) / prev_price * 100) if prev_price != 0 else 0,
                    "volume": 0,
                    "market_cap": 0,
                    "pe_ratio": 0,
                    "last_updated": datetime.now().isoformat(),
                    "data_source": "local_csv",
                    "status": "fallback_success"
                }
            
            # Method 4: Return dummy data with warning
            print(f"No fallback data available for {symbol}, returning dummy data")
            return {
                "symbol": symbol,
                "current_price": 100.0,
                "change": 0.0,
                "change_percent": 0.0,
                "volume": 0,
                "market_cap": 0,
                "pe_ratio": 0,
                "last_updated": datetime.now().isoformat(),
                "data_source": "dummy",
                "status": "no_data_available",
                "warning": "No real data available, showing dummy values"
            }
            
        except Exception as e:
            print(f"Fallback methods failed for {symbol}: {e}")
            return {}
    
    def calculate_what_if_scenario(self, symbol: str, action: str, quantity: int, 
                                 days_ago: int = 0) -> Dict:
        """Calculate what-if scenarios for stock purchases"""
        try:
            # Get historical data
            data = self.get_stock_data(symbol, period="1y")
            if data.empty:
                return {"error": "No data available"}
            
            if days_ago > 0:
                # Get price from N days ago
                target_date = datetime.now() - timedelta(days=days_ago)
                historical_data = data[data.index.date <= target_date.date()]
                if historical_data.empty:
                    return {"error": "No historical data for that date"}
                
                price = historical_data['Close'].iloc[-1]
                date = historical_data.index[-1]
            else:
                # Current price
                price = data['Close'].iloc[-1]
                date = data.index[-1]
            
            # Calculate scenarios
            total_cost = price * quantity
            
            # Calculate current value if bought then
            current_price = data['Close'].iloc[-1]
            current_value = current_price * quantity
            profit_loss = current_value - total_cost
            profit_loss_percent = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            return {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price_at_time": float(price),
                "date": date.strftime("%Y-%m-%d"),
                "total_cost": float(total_cost),
                "current_price": float(current_price),
                "current_value": float(current_value),
                "profit_loss": float(profit_loss),
                "profit_loss_percent": float(profit_loss_percent)
            }
        except Exception as e:
            print(f"Error calculating what-if scenario: {e}")
            return {"error": str(e)}
    
    def get_ml_predictions(self, symbol: str) -> Dict:
        """Get ML model predictions for a stock"""
        try:
            # This would integrate with your existing ML models
            # For now, return a placeholder structure
            return {
                "symbol": symbol,
                "lstm_prediction": "Model prediction would go here",
                "dqn_recommendation": "Trading recommendation would go here",
                "sentiment_score": 0.0,
                "confidence": 0.0
            }
        except Exception as e:
            print(f"Error getting ML predictions: {e}")
            return {"error": str(e)}
    
    def get_market_summary(self) -> Dict:
        """Get overall market summary with improved error handling"""
        try:
            summary = {}
            total_stocks = len(SUPPORTED_STOCKS)
            successful_fetches = 0
            failed_stocks = []
            
            for i, symbol in enumerate(SUPPORTED_STOCKS):
                print(f"Fetching {symbol} ({i+1}/{total_stocks})...")
                price_info = self.get_stock_price(symbol)
                if price_info and price_info.get("status") != "error":
                    summary[symbol] = price_info
                    successful_fetches += 1
                    print(f"✓ Success: {symbol}")
                else:
                    failed_stocks.append(symbol)
                    print(f"✗ Failed: {symbol}")
                
                # Add a small delay between requests to be respectful to the API
                if i < total_stocks - 1:  # Don't sleep after the last request
                    time.sleep(0.5)  # Reduced delay
            
            return {
                "timestamp": datetime.now().isoformat(),
                "stocks": summary,
                "market_status": "open",  # This could be determined by checking market hours
                "summary_stats": {
                    "total_requested": total_stocks,
                    "successful": successful_fetches,
                    "failed": len(failed_stocks),
                    "failed_stocks": failed_stocks,
                    "success_rate": f"{(successful_fetches/total_stocks)*100:.1f}%"
                }
            }
        except Exception as e:
            print(f"Error getting market summary: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict:
        """Check the health of data sources"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "services": {}
        }
        
        # Test YFinance
        try:
            test_data = yf.download("AAPL", period="1d", progress=False)
            if not test_data.empty:
                health_status["services"]["yfinance"] = {"status": "healthy", "message": "Working normally"}
            else:
                health_status["services"]["yfinance"] = {"status": "degraded", "message": "No data returned"}
        except Exception as e:
            health_status["services"]["yfinance"] = {"status": "unhealthy", "message": str(e)}
        
        # Test Alpha Vantage
        if self.alpha_vantage and ALPHA_VANTAGE_API_KEY:
            try:
                data, meta = self.alpha_vantage.get_quote_endpoint("AAPL")
                health_status["services"]["alpha_vantage"] = {"status": "healthy", "message": "Working normally"}
            except Exception as e:
                health_status["services"]["alpha_vantage"] = {"status": "unhealthy", "message": str(e)}
        else:
            health_status["services"]["alpha_vantage"] = {"status": "not_configured", "message": "No API key"}
        
        # Test News API
        if NEWS_API_KEY:
            try:
                url = f"https://newsapi.org/v2/everything?q=AAPL&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    health_status["services"]["news_api"] = {"status": "healthy", "message": "Working normally"}
                else:
                    health_status["services"]["news_api"] = {"status": "degraded", "message": f"HTTP {response.status_code}"}
            except Exception as e:
                health_status["services"]["news_api"] = {"status": "unhealthy", "message": str(e)}
        else:
            health_status["services"]["news_api"] = {"status": "not_configured", "message": "No API key"}
        
        # Determine overall status
        statuses = [service["status"] for service in health_status["services"].values()]
        if any(status == "healthy" for status in statuses):
            health_status["overall_status"] = "operational"
        elif any(status == "degraded" for status in statuses):
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "down"
        
        return health_status
    
    def clear_cache(self):
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            print("Cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "cache_keys": list(self._cache.keys())
            }
