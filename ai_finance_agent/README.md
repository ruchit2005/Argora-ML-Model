# AI Finance Agent with LangGraph

An intelligent financial advisor AI system that integrates with your existing ML models for stock market prediction and provides comprehensive financial Q&A capabilities.

## Features

- **Intelligent Financial Q&A**: Answer complex financial questions using real-time data
- **What-If Analysis**: Calculate scenarios like "What if I bought X stock Y days ago?"
- **ML Model Integration**: Access to your LSTM/GRU forecasting and Rainbow DQN models
- **Sentiment Analysis**: Real-time news sentiment analysis for stocks
- **Bulk Purchase Analysis**: Analyze the impact of large stock purchases
- **Portfolio Impact**: Understand how adding stocks affects your portfolio

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph      │    │   Data Manager  │
│   Endpoints     │◄──►│   AI Agent       │◄──►│   & ML Models   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   REST API      │    │   Query          │    │   Stock Data    │
│   Routes        │    │   Processing     │    │   & Sentiment   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. **Install Dependencies**:
```bash
cd ai_finance_agent
pip install -r requirements.txt
```

2. **Configure API Keys**:
```bash
# Copy the template configuration
cp config_template.py config.py

# Edit config.py with your API keys
# Get API keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
# - News API: https://newsapi.org/register
```

3. **Run the Server**:
```bash
python main.py
# Or use the run script
python run_server.py
```

The server will start on `http://localhost:8000`

## Security Note

- **config.py** contains your API keys and is excluded from git for security
- **config_template.py** shows the required configuration structure
- Never commit actual API keys to version control

## API Endpoints

### General Queries
```bash
POST /query
{
  "query": "What's the current price of AAPL?"
}
```

### What-If Scenarios
```bash
POST /whatif
{
  "symbol": "AAPL",
  "action": "bought",
  "quantity": 100,
  "days_ago": 30
}
```

### Stock Information
```bash
GET /stock/AAPL
GET /market/summary
GET /sentiment/AAPL?days=7
```

## Example Queries

### Financial Questions
- "What's the current price of Tesla?"
- "Should I buy Microsoft stock?"
- "What's the sentiment around Apple?"
- "Predict the future price of Google"
- "Is Amazon a good investment?"

### What-If Scenarios
- "What if I bought 100 shares of AAPL 30 days ago?"
- "What if I bought 50 shares of TSLA 7 days ago?"
- "What if I bought 1000 shares of MSFT right now?"

### Bulk Purchase Analysis
- "What if I buy 1000 shares of GOOGL in bulk?"
- "Analyze the impact of buying 500 shares of NVDA"

## Integration with Your ML Models

The system integrates with your existing ML infrastructure:

1. **LSTM/GRU Forecasting**: Uses your forecasting models for predictions
2. **Rainbow DQN**: Integrates trading recommendations from your DQN agent
3. **Sentiment Analysis**: Uses your news scraping and sentiment analysis
4. **Historical Data**: Accesses your CSV data files for analysis

## Advanced Features

### What-If Analyzer
- Historical purchase analysis
- Bulk purchase impact assessment
- Portfolio allocation analysis
- Risk metrics calculation
- Support/resistance level analysis

### Risk Analysis
- Value at Risk (VaR) calculations
- Expected Shortfall analysis
- Volatility assessment
- Maximum drawdown analysis
- Sharpe ratio calculations

### Portfolio Impact
- Allocation percentage analysis
- Risk contribution assessment
- Diversification recommendations
- Concentration risk warnings

## Usage Examples

### Python Client
```python
import requests

# General query
response = requests.post("http://localhost:8000/query", 
                        json={"query": "What's the best tech stock to buy?"})
print(response.json())

# What-if scenario
response = requests.post("http://localhost:8000/whatif", 
                        json={
                            "symbol": "AAPL",
                            "action": "bought", 
                            "quantity": 100,
                            "days_ago": 30
                        })
print(response.json())
```

### cURL Examples
```bash
# General query
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the current sentiment around Tesla?"}'

# What-if analysis
curl -X POST "http://localhost:8000/whatif" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "TSLA", "action": "bought", "quantity": 50, "days_ago": 7}'
```

## Configuration

Edit `config.py` to customize:
- API keys and endpoints
- Model parameters
- Supported stocks
- Forecasting settings

## Error Handling

The system includes comprehensive error handling:
- API key validation
- Data source fallbacks
- Model loading errors
- Network timeout handling
- Graceful degradation

## Performance

- Fast response times (< 2 seconds for most queries)
- Efficient data caching
- Parallel data fetching
- Optimized ML model integration

## Security

- Input validation and sanitization
- API key protection
- Rate limiting (configurable)
- CORS protection

## Monitoring

- Health check endpoint (`/health`)
- Performance metrics
- Error logging
- Usage analytics

## Future Enhancements

- Real-time streaming updates
- Advanced portfolio optimization
- Machine learning model retraining
- Multi-language support
- Mobile app integration

## Troubleshooting

1. **API Key Issues**: Ensure all API keys are valid and have sufficient quota
2. **Data Access**: Check that your ML model data files are accessible
3. **Network Issues**: Verify internet connectivity for real-time data
4. **Model Loading**: Ensure TensorFlow models are compatible

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify API key configuration
3. Test individual endpoints
4. Review data file permissions
