"""
AI Finance Agent using LangGraph
"""
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN'))

from data_manager import DataManager
from config import OPENAI_API_KEY, MODEL_TEMPERATURE, MAX_TOKENS

class FinanceAgentState:
    """State for the finance agent"""
    def __init__(self):
        self.query = ""
        self.stock_symbol = ""
        self.query_type = ""
        self.context_data = {}
        self.response = ""
        self.confidence = 0.0
        self.tools_used = []

class FinanceAgent:
    """AI Finance Agent with LangGraph"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=MODEL_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=OPENAI_API_KEY
        )
        self.data_manager = DataManager()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(FinanceAgentState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("gather_data", self._gather_data)
        workflow.add_node("analyze_data", self._analyze_data)
        workflow.add_node("generate_response", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "gather_data")
        workflow.add_edge("gather_data", "analyze_data")
        workflow.add_edge("analyze_data", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: FinanceAgentState) -> FinanceAgentState:
        """Classify the type of query"""
        query = state.query.lower()
        
        # Determine query type
        if any(word in query for word in ["what if", "if i bought", "if i had bought", "scenario"]):
            state.query_type = "what_if"
        elif any(word in query for word in ["price", "current", "now", "today"]):
            state.query_type = "current_price"
        elif any(word in query for word in ["predict", "forecast", "future", "will"]):
            state.query_type = "prediction"
        elif any(word in query for word in ["news", "sentiment", "headlines"]):
            state.query_type = "sentiment"
        elif any(word in query for word in ["buy", "sell", "hold", "recommend"]):
            state.query_type = "recommendation"
        else:
            state.query_type = "general"
        
        # Extract stock symbol
        symbols = ["AMZN", "GOOG", "MSFT", "TSLA", "NSEI", "AAPL", "NVDA", "META"]
        for symbol in symbols:
            if symbol.lower() in query or symbol in query:
                state.stock_symbol = symbol
                break
        
        return state
    
    def _gather_data(self, state: FinanceAgentState) -> FinanceAgentState:
        """Gather relevant data based on query type"""
        context_data = {}
        
        if state.stock_symbol:
            # Get stock data
            context_data["stock_info"] = self.data_manager.get_stock_price(state.stock_symbol)
            context_data["historical_data"] = self.data_manager.get_historical_data(state.stock_symbol)
            context_data["news_sentiment"] = self.data_manager.get_news_sentiment(state.stock_symbol)
            context_data["ml_predictions"] = self.data_manager.get_ml_predictions(state.stock_symbol)
        
        # Add market context
        context_data["market_summary"] = self.data_manager.get_market_summary()
        
        state.context_data = context_data
        return state
    
    def _analyze_data(self, state: FinanceAgentState) -> FinanceAgentState:
        """Analyze the gathered data"""
        # This is where you would add more sophisticated analysis
        # For now, we'll pass the data to the response generator
        return state
    
    def _generate_response(self, state: FinanceAgentState) -> FinanceAgentState:
        """Generate the final response"""
        # Create system prompt based on query type
        system_prompt = self._create_system_prompt(state.query_type)
        
        # Create user prompt with context
        user_prompt = self._create_user_prompt(state.query, state.context_data)
        
        # Generate response
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        state.response = response.content
        state.confidence = 0.8  # This could be calculated based on data quality
        
        return state
    
    def _create_system_prompt(self, query_type: str) -> str:
        """Create system prompt based on query type"""
        base_prompt = """You are an expert financial advisor AI with access to real-time stock data, ML predictions, and news sentiment analysis. 
        You can analyze stock prices, provide investment advice, and answer what-if scenarios.
        
        Always be accurate, cite specific data when available, and provide disclaimers about investment risks."""
        
        if query_type == "what_if":
            return base_prompt + """
            For what-if scenarios, calculate the exact financial impact including:
            - Purchase cost at the specified time
            - Current value
            - Profit/loss amount and percentage
            - Provide clear, actionable insights
            """
        elif query_type == "prediction":
            return base_prompt + """
            For predictions, use the ML model data and sentiment analysis to provide:
            - Short-term and long-term outlook
            - Confidence levels
            - Key factors influencing the prediction
            - Risk factors to consider
            """
        elif query_type == "recommendation":
            return base_prompt + """
            For recommendations, provide:
            - Clear buy/sell/hold recommendation
            - Reasoning based on technical and fundamental analysis
            - Risk assessment
            - Time horizon for the recommendation
            """
        else:
            return base_prompt
    
    def _create_user_prompt(self, query: str, context_data: Dict) -> str:
        """Create user prompt with context data"""
        prompt = f"User Query: {query}\n\n"
        
        # Add relevant context data
        if "stock_info" in context_data and context_data["stock_info"]:
            stock_info = context_data["stock_info"]
            prompt += f"Current Stock Information for {stock_info.get('symbol', 'N/A')}:\n"
            prompt += f"- Current Price: ${stock_info.get('current_price', 'N/A'):.2f}\n"
            prompt += f"- Change: ${stock_info.get('change', 0):.2f} ({stock_info.get('change_percent', 0):.2f}%)\n"
            prompt += f"- Volume: {stock_info.get('volume', 'N/A'):,}\n"
            prompt += f"- Market Cap: ${stock_info.get('market_cap', 0):,}\n"
            prompt += f"- P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}\n\n"
        
        if "news_sentiment" in context_data and context_data["news_sentiment"]:
            sentiments = context_data["news_sentiment"]
            if sentiments:
                avg_sentiment = sum(s["sentiment"] for s in sentiments) / len(sentiments)
                prompt += f"News Sentiment Analysis:\n"
                prompt += f"- Average Sentiment Score: {avg_sentiment:.3f}\n"
                prompt += f"- Number of Articles Analyzed: {len(sentiments)}\n"
                prompt += f"- Recent Headlines: {sentiments[0]['title']}\n\n"
        
        if "ml_predictions" in context_data and context_data["ml_predictions"]:
            predictions = context_data["ml_predictions"]
            prompt += f"ML Model Predictions:\n"
            prompt += f"- LSTM Prediction: {predictions.get('lstm_prediction', 'N/A')}\n"
            prompt += f"- DQN Recommendation: {predictions.get('dqn_recommendation', 'N/A')}\n"
            prompt += f"- Sentiment Score: {predictions.get('sentiment_score', 0):.3f}\n"
            prompt += f"- Confidence: {predictions.get('confidence', 0):.3f}\n\n"
        
        prompt += "Please provide a comprehensive response to the user's query using the above data."
        
        return prompt
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return response"""
        try:
            # Create initial state
            state = FinanceAgentState()
            state.query = query
            
            # Run the graph
            result = self.graph.invoke(state)
            
            return {
                "query": query,
                "response": result.response,
                "confidence": result.confidence,
                "query_type": result.query_type,
                "stock_symbol": result.stock_symbol,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "query": query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)}",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_what_if_scenario(self, symbol: str, action: str, quantity: int, days_ago: int = 0) -> Dict[str, Any]:
        """Process a what-if scenario query"""
        try:
            scenario = self.data_manager.calculate_what_if_scenario(symbol, action, quantity, days_ago)
            
            if "error" in scenario:
                return {
                    "query": f"What if I {action} {quantity} shares of {symbol} {days_ago} days ago",
                    "response": f"I couldn't calculate that scenario: {scenario['error']}",
                    "confidence": 0.0,
                    "error": scenario["error"]
                }
            
            # Generate natural language response
            response = f"""If you had {action} {quantity} shares of {symbol} {days_ago} days ago:
            
• Purchase Date: {scenario['date']}
• Price at Time: ${scenario['price_at_time']:.2f}
• Total Cost: ${scenario['total_cost']:,.2f}
• Current Price: ${scenario['current_price']:.2f}
• Current Value: ${scenario['current_value']:,.2f}
• Profit/Loss: ${scenario['profit_loss']:,.2f} ({scenario['profit_loss_percent']:.2f}%)

{'📈 You would have made a profit!' if scenario['profit_loss'] > 0 else '📉 You would have a loss.' if scenario['profit_loss'] < 0 else '💰 You would break even.'}"""
            
            return {
                "query": f"What if I {action} {quantity} shares of {symbol} {days_ago} days ago",
                "response": response,
                "confidence": 0.9,
                "scenario_data": scenario,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "query": f"What if I {action} {quantity} shares of {symbol} {days_ago} days ago",
                "response": f"I couldn't calculate that scenario: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
