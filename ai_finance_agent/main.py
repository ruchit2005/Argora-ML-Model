"""
FastAPI server for the AI Finance Agent
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from datetime import datetime

from ai_agent import FinanceAgent
from config import HOST, PORT, DEBUG

# Initialize FastAPI app
app = FastAPI(
    title="AI Finance Agent",
    description="AI-powered financial advisor with ML model integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the AI agent
finance_agent = FinanceAgent()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    confidence: float
    query_type: Optional[str] = None
    stock_symbol: Optional[str] = None
    timestamp: str
    error: Optional[str] = None

class WhatIfRequest(BaseModel):
    symbol: str
    action: str  # "bought" or "sold"
    quantity: int
    days_ago: int = 0

class WhatIfResponse(BaseModel):
    query: str
    response: str
    confidence: float
    scenario_data: Optional[Dict[str, Any]] = None
    timestamp: str
    error: Optional[str] = None

# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Finance Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Ask financial questions",
            "/whatif": "POST - Calculate what-if scenarios",
            "/health": "GET - Health check",
            "/stocks": "GET - Get supported stocks"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agent_initialized": finance_agent is not None
    }

@app.get("/stocks")
async def get_supported_stocks():
    """Get list of supported stock symbols"""
    from config import SUPPORTED_STOCKS
    return {
        "supported_stocks": SUPPORTED_STOCKS,
        "description": "List of stock symbols that the AI agent can analyze"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a general financial query
    
    Examples:
    - "What's the current price of AAPL?"
    - "Should I buy Tesla stock?"
    - "What's the sentiment around Microsoft?"
    - "Predict the future price of GOOGL"
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        result = finance_agent.process_query(request.query)
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/whatif", response_model=WhatIfResponse)
async def process_what_if_scenario(request: WhatIfRequest):
    """
    Process a what-if scenario query
    
    Examples:
    - symbol: "AAPL", action: "bought", quantity: 100, days_ago: 30
    - symbol: "TSLA", action: "bought", quantity: 50, days_ago: 7
    """
    try:
        if request.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        if request.days_ago < 0:
            raise HTTPException(status_code=400, detail="Days ago must be non-negative")
        
        result = finance_agent.process_what_if_scenario(
            request.symbol, 
            request.action, 
            request.quantity, 
            request.days_ago
        )
        return WhatIfResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing what-if scenario: {str(e)}")

@app.get("/stock/{symbol}")
async def get_stock_info(symbol: str):
    """Get current information for a specific stock"""
    try:
        stock_info = finance_agent.data_manager.get_stock_price(symbol.upper())
        if not stock_info:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found or no data available")
        
        return {
            "symbol": symbol.upper(),
            "data": stock_info,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock info: {str(e)}")

@app.get("/market/summary")
async def get_market_summary():
    """Get overall market summary"""
    try:
        summary = finance_agent.data_manager.get_market_summary()
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching market summary: {str(e)}")

@app.get("/sentiment/{symbol}")
async def get_stock_sentiment(symbol: str, days: int = Query(7, ge=1, le=30)):
    """Get news sentiment analysis for a stock"""
    try:
        sentiments = finance_agent.data_manager.get_news_sentiment(symbol.upper(), days)
        if not sentiments:
            return {
                "symbol": symbol.upper(),
                "sentiments": [],
                "average_sentiment": 0.0,
                "message": "No sentiment data available"
            }
        
        avg_sentiment = sum(s["sentiment"] for s in sentiments) / len(sentiments)
        return {
            "symbol": symbol.upper(),
            "sentiments": sentiments,
            "average_sentiment": avg_sentiment,
            "total_articles": len(sentiments),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sentiment: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )
