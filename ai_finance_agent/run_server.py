"""
Server startup script with environment validation
"""
import os
import sys
from pathlib import Path

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment")
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import langgraph
        import langchain
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import yfinance
        import requests
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_data_access():
    """Check if data files are accessible"""
    data_dir = Path("../Real-Time-Stock-Market-Prediction-using-Ensemble-DL-and-Rainbow-DQN/rainbow/data")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure the ML model data directory is accessible")
        return False
    
    # Check for some key files
    key_files = ["AAPL.csv", "GOOG.csv", "MSFT.csv", "TSLA.csv"]
    missing_files = []
    
    for file in key_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Some data files missing: {missing_files}")
        print("The system will work but with limited data")
    
    print("✅ Data directory accessible")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting AI Finance Agent Server...")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data access
    check_data_access()
    
    print("\n✅ Environment checks passed!")
    print("🌐 Starting server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    # Import and run the server
    try:
        from main import app
        import uvicorn
        from config import HOST, PORT, DEBUG
        
        uvicorn.run(
            "main:app",
            host=HOST,
            port=PORT,
            reload=DEBUG,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
