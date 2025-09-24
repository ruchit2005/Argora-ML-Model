"""
Advanced What-If Analysis Module
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN'))

from data_manager import DataManager

class WhatIfAnalyzer:
    """Advanced what-if scenario analyzer"""
    
    def __init__(self):
        self.data_manager = DataManager()
    
    def analyze_bulk_purchase(self, symbol: str, quantity: int, 
                            current_price: float = None) -> Dict:
        """Analyze the impact of a bulk purchase"""
        try:
            if current_price is None:
                stock_info = self.data_manager.get_stock_price(symbol)
                if not stock_info:
                    return {"error": "Could not fetch current stock price"}
                current_price = stock_info["current_price"]
            
            # Calculate total cost
            total_cost = current_price * quantity
            
            # Get historical data for analysis
            historical_data = self.data_manager.get_stock_data(symbol, period="1y")
            
            if historical_data.empty:
                return {"error": "No historical data available"}
            
            # Calculate various metrics
            price_volatility = self._calculate_volatility(historical_data)
            support_resistance = self._find_support_resistance(historical_data)
            trend_analysis = self._analyze_trend(historical_data)
            
            # Risk analysis
            risk_metrics = self._calculate_risk_metrics(historical_data, total_cost)
            
            return {
                "symbol": symbol,
                "quantity": quantity,
                "current_price": current_price,
                "total_cost": total_cost,
                "analysis": {
                    "volatility": price_volatility,
                    "support_resistance": support_resistance,
                    "trend": trend_analysis,
                    "risk_metrics": risk_metrics
                },
                "recommendations": self._generate_bulk_purchase_recommendations(
                    total_cost, price_volatility, trend_analysis
                )
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_historical_purchase(self, symbol: str, quantity: int, 
                                  days_ago: int) -> Dict:
        """Analyze what would have happened if bought N days ago"""
        try:
            # Get historical data
            historical_data = self.data_manager.get_stock_data(symbol, period="1y")
            
            if historical_data.empty:
                return {"error": "No historical data available"}
            
            # Find the price N days ago
            target_date = datetime.now() - timedelta(days=days_ago)
            historical_data['date'] = historical_data.index
            
            # Find closest trading day
            past_data = historical_data[historical_data.index <= target_date]
            if past_data.empty:
                return {"error": f"No data available for {days_ago} days ago"}
            
            past_price = past_data['Close'].iloc[-1]
            past_date = past_data.index[-1]
            current_price = historical_data['Close'].iloc[-1]
            
            # Calculate metrics
            total_cost = past_price * quantity
            current_value = current_price * quantity
            profit_loss = current_value - total_cost
            profit_loss_percent = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            # Calculate additional metrics
            price_change = current_price - past_price
            price_change_percent = (price_change / past_price) * 100
            
            # Performance analysis
            performance_metrics = self._calculate_performance_metrics(
                historical_data, past_date, current_price
            )
            
            return {
                "symbol": symbol,
                "quantity": quantity,
                "purchase_date": past_date.strftime("%Y-%m-%d"),
                "purchase_price": past_price,
                "current_price": current_price,
                "total_cost": total_cost,
                "current_value": current_value,
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "performance_metrics": performance_metrics,
                "recommendation": self._generate_historical_recommendation(
                    profit_loss_percent, performance_metrics
                )
            }
        except Exception as e:
            return {"error": f"Historical analysis failed: {str(e)}"}
    
    def analyze_portfolio_impact(self, symbol: str, quantity: int, 
                               portfolio_value: float) -> Dict:
        """Analyze the impact of adding this stock to a portfolio"""
        try:
            stock_info = self.data_manager.get_stock_price(symbol)
            if not stock_info:
                return {"error": "Could not fetch stock information"}
            
            current_price = stock_info["current_price"]
            total_cost = current_price * quantity
            
            # Calculate portfolio allocation
            allocation_percent = (total_cost / portfolio_value) * 100
            
            # Get historical data for correlation analysis
            historical_data = self.data_manager.get_stock_data(symbol, period="1y")
            
            if historical_data.empty:
                return {"error": "No historical data available"}
            
            # Calculate risk metrics
            volatility = self._calculate_volatility(historical_data)
            max_drawdown = self._calculate_max_drawdown(historical_data)
            
            # Portfolio impact analysis
            portfolio_impact = {
                "allocation_percent": allocation_percent,
                "dollar_amount": total_cost,
                "risk_contribution": volatility * (allocation_percent / 100),
                "max_drawdown_impact": max_drawdown * (allocation_percent / 100)
            }
            
            # Generate recommendations
            recommendations = self._generate_portfolio_recommendations(
                allocation_percent, volatility, portfolio_impact
            )
            
            return {
                "symbol": symbol,
                "quantity": quantity,
                "current_price": current_price,
                "total_cost": total_cost,
                "portfolio_impact": portfolio_impact,
                "risk_metrics": {
                    "volatility": volatility,
                    "max_drawdown": max_drawdown
                },
                "recommendations": recommendations
            }
        except Exception as e:
            return {"error": f"Portfolio analysis failed: {str(e)}"}
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate price volatility"""
        returns = data['Close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized volatility
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        prices = data['Close'].values
        
        # Simple support/resistance calculation
        recent_prices = prices[-20:]  # Last 20 days
        support = np.percentile(recent_prices, 25)
        resistance = np.percentile(recent_prices, 75)
        
        return {
            "support": float(support),
            "resistance": float(resistance),
            "current_price": float(prices[-1]),
            "distance_to_support": float((prices[-1] - support) / support * 100),
            "distance_to_resistance": float((resistance - prices[-1]) / prices[-1] * 100)
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze price trend"""
        prices = data['Close'].values
        
        # Short-term trend (last 5 days)
        short_trend = np.polyfit(range(5), prices[-5:], 1)[0] if len(prices) >= 5 else 0
        
        # Medium-term trend (last 20 days)
        medium_trend = np.polyfit(range(20), prices[-20:], 1)[0] if len(prices) >= 20 else 0
        
        # Long-term trend (last 60 days)
        long_trend = np.polyfit(range(60), prices[-60:], 1)[0] if len(prices) >= 60 else 0
        
        return {
            "short_term": float(short_trend),
            "medium_term": float(medium_trend),
            "long_term": float(long_trend),
            "trend_strength": "strong" if abs(long_trend) > 0.5 else "weak",
            "trend_direction": "up" if long_trend > 0 else "down"
        }
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, investment_amount: float) -> Dict:
        """Calculate various risk metrics"""
        returns = data['Close'].pct_change().dropna()
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5)
        var_amount = investment_amount * abs(var_95)
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = returns[returns <= var_95]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_95
        es_amount = investment_amount * abs(expected_shortfall)
        
        return {
            "var_95_percent": float(var_95 * 100),
            "var_95_amount": float(var_amount),
            "expected_shortfall_percent": float(expected_shortfall * 100),
            "expected_shortfall_amount": float(es_amount),
            "max_daily_loss": float(returns.min() * 100),
            "max_daily_gain": float(returns.max() * 100)
        }
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, start_date: pd.Timestamp, 
                                    current_price: float) -> Dict:
        """Calculate performance metrics for a period"""
        start_price = data.loc[start_date, 'Close']
        
        # Total return
        total_return = (current_price - start_price) / start_price * 100
        
        # Annualized return
        days_held = (datetime.now() - start_date).days
        annualized_return = ((current_price / start_price) ** (365 / days_held) - 1) * 100 if days_held > 0 else 0
        
        # Sharpe ratio (simplified)
        returns = data['Close'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            "total_return_percent": float(total_return),
            "annualized_return_percent": float(annualized_return),
            "days_held": days_held,
            "sharpe_ratio": float(sharpe_ratio)
        }
    
    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        prices = data['Close'].values
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return float(max_dd * 100)
    
    def _generate_bulk_purchase_recommendations(self, total_cost: float, 
                                              volatility: float, trend: Dict) -> List[str]:
        """Generate recommendations for bulk purchase"""
        recommendations = []
        
        if total_cost > 100000:  # Large investment
            recommendations.append("💡 Consider dollar-cost averaging for large investments")
        
        if volatility > 0.3:  # High volatility
            recommendations.append("⚠️ High volatility detected - consider position sizing")
        
        if trend["trend_direction"] == "up" and trend["trend_strength"] == "strong":
            recommendations.append("📈 Strong uptrend - good timing for entry")
        elif trend["trend_direction"] == "down":
            recommendations.append("📉 Downtrend detected - consider waiting for reversal")
        
        if not recommendations:
            recommendations.append("✅ Investment appears reasonable based on current metrics")
        
        return recommendations
    
    def _generate_historical_recommendation(self, profit_loss_percent: float, 
                                         performance_metrics: Dict) -> str:
        """Generate recommendation based on historical analysis"""
        if profit_loss_percent > 20:
            return "🎉 Excellent investment! Strong returns achieved."
        elif profit_loss_percent > 5:
            return "✅ Good investment with positive returns."
        elif profit_loss_percent > -5:
            return "⚖️ Mixed results - consider market conditions."
        else:
            return "📉 Loss incurred - review investment strategy."
    
    def _generate_portfolio_recommendations(self, allocation_percent: float, 
                                          volatility: float, portfolio_impact: Dict) -> List[str]:
        """Generate portfolio recommendations"""
        recommendations = []
        
        if allocation_percent > 20:
            recommendations.append("⚠️ High concentration risk - consider diversification")
        
        if portfolio_impact["risk_contribution"] > 0.1:
            recommendations.append("⚠️ High risk contribution to portfolio")
        
        if volatility > 0.4:
            recommendations.append("📊 High volatility stock - ensure it fits risk tolerance")
        
        if allocation_percent < 5:
            recommendations.append("✅ Conservative allocation - good for diversification")
        
        return recommendations
