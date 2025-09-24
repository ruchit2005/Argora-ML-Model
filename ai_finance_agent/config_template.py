"""
Configuration Template for AI Finance Agent

Copy this file to 'config.py' and fill in your API keys and settings.
Never commit config.py to version control - it's in .gitignore for security.

To use:
1. Copy this file: cp config_template.py config.py
2. Edit config.py with your actual API keys
3. The .gitignore will ensure config.py stays private
"""

# API Keys (Get these from the respective services)
OPENAI_API_KEY = ""  # Get from https://platform.openai.com/api-keys
ALPHA_VANTAGE_API_KEY = ""  # Get from https://www.alphavantage.co/support/#api-key  
NEWS_API_KEY = ""  # Get from https://newsapi.org/register

# Server Configuration
HOST = "127.0.0.1"
PORT = 8000
DEBUG = True

# Data Configuration
DATA_DIR = "../Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN/rainbow/data"
MODELS_DIR = "../Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN/rainbow/models"

# Supported Stock Symbols
SUPPORTED_STOCKS = [
    "AAPL",   # Apple Inc.
    "GOOGL",  # Alphabet Inc.
    "MSFT",   # Microsoft Corporation
    "AMZN",   # Amazon.com Inc.
    "TSLA",   # Tesla Inc.
    "META",   # Meta Platforms Inc.
    "NVDA",   # NVIDIA Corporation
    "NFLX",   # Netflix Inc.
    "DIS",    # The Walt Disney Company
    "UBER",   # Uber Technologies Inc.
    "NIO",    # NIO Inc.
    "TATA",   # Tata Motors (may not work with US APIs)
    "RELIANCE"  # Reliance Industries (may not work with US APIs)
]

# AI Configuration
AI_MODEL = "gpt-3.5-turbo"  # OpenAI model to use
MAX_TOKENS = 500
TEMPERATURE = 0.7

# Cache Configuration
CACHE_TIMEOUT = 300  # 5 minutes in seconds