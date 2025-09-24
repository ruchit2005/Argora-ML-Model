"""
Test script for the AI Finance Agent API
"""
import requests
import json
import time
from typing import Dict, Any

class FinanceAgentTester:
    """Test client for the AI Finance Agent API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Health check passed")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_general_queries(self) -> None:
        """Test general financial queries"""
        queries = [
            "What's the current price of AAPL?",
            "Should I buy Tesla stock?",
            "What's the sentiment around Microsoft?",
            "Is Amazon a good investment?",
            "Predict the future price of Google"
        ]
        
        print("\n🧠 Testing General Queries...")
        for query in queries:
            try:
                response = self.session.post(
                    f"{self.base_url}/query",
                    json={"query": query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Query: {query}")
                    print(f"   Response: {result['response'][:100]}...")
                    print(f"   Confidence: {result['confidence']:.2f}")
                else:
                    print(f"❌ Query failed: {query} - Status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Query error: {query} - {e}")
            
            time.sleep(1)  # Rate limiting
    
    def test_what_if_scenarios(self) -> None:
        """Test what-if scenario queries"""
        scenarios = [
            {"symbol": "AAPL", "action": "bought", "quantity": 100, "days_ago": 30},
            {"symbol": "TSLA", "action": "bought", "quantity": 50, "days_ago": 7},
            {"symbol": "MSFT", "action": "bought", "quantity": 200, "days_ago": 14},
            {"symbol": "GOOGL", "action": "bought", "quantity": 25, "days_ago": 3}
        ]
        
        print("\n🔮 Testing What-If Scenarios...")
        for scenario in scenarios:
            try:
                response = self.session.post(
                    f"{self.base_url}/whatif",
                    json=scenario,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Scenario: {scenario['symbol']} - {scenario['action']} {scenario['quantity']} shares {scenario['days_ago']} days ago")
                    print(f"   Response: {result['response'][:100]}...")
                    print(f"   Confidence: {result['confidence']:.2f}")
                else:
                    print(f"❌ Scenario failed: {scenario} - Status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Scenario error: {scenario} - {e}")
            
            time.sleep(1)
    
    def test_stock_info(self) -> None:
        """Test stock information endpoints"""
        symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
        
        print("\n📊 Testing Stock Information...")
        for symbol in symbols:
            try:
                response = self.session.get(f"{self.base_url}/stock/{symbol}", timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {symbol}: ${result['data']['current_price']:.2f} ({result['data']['change_percent']:.2f}%)")
                else:
                    print(f"❌ {symbol} info failed - Status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {symbol} info error: {e}")
            
            time.sleep(1)
    
    def test_market_summary(self) -> None:
        """Test market summary endpoint"""
        print("\n📈 Testing Market Summary...")
        try:
            response = self.session.get(f"{self.base_url}/market/summary", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Market summary retrieved with {len(result.get('stocks', {}))} stocks")
                for symbol, data in result.get('stocks', {}).items():
                    print(f"   {symbol}: ${data.get('current_price', 0):.2f}")
            else:
                print(f"❌ Market summary failed - Status: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Market summary error: {e}")
    
    def test_sentiment_analysis(self) -> None:
        """Test sentiment analysis endpoint"""
        symbols = ["AAPL", "TSLA", "MSFT"]
        
        print("\n📰 Testing Sentiment Analysis...")
        for symbol in symbols:
            try:
                response = self.session.get(f"{self.base_url}/sentiment/{symbol}?days=7", timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {symbol} sentiment: {result.get('average_sentiment', 0):.3f} ({result.get('total_articles', 0)} articles)")
                else:
                    print(f"❌ {symbol} sentiment failed - Status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {symbol} sentiment error: {e}")
            
            time.sleep(1)
    
    def test_error_handling(self) -> None:
        """Test error handling"""
        print("\n🚨 Testing Error Handling...")
        
        # Test invalid query
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                json={"query": ""},
                timeout=10
            )
            if response.status_code == 400:
                print("✅ Empty query properly rejected")
            else:
                print(f"❌ Empty query not rejected - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Empty query test error: {e}")
        
        # Test invalid what-if scenario
        try:
            response = self.session.post(
                f"{self.base_url}/whatif",
                json={"symbol": "INVALID", "action": "bought", "quantity": -1, "days_ago": -1},
                timeout=10
            )
            if response.status_code in [400, 500]:
                print("✅ Invalid what-if scenario properly handled")
            else:
                print(f"❌ Invalid what-if scenario not handled - Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Invalid what-if test error: {e}")
    
    def run_all_tests(self) -> None:
        """Run all tests"""
        print("🚀 Starting AI Finance Agent API Tests...")
        print("=" * 50)
        
        # Test health first
        if not self.test_health():
            print("❌ Health check failed - stopping tests")
            return
        
        # Run all test suites
        self.test_general_queries()
        self.test_what_if_scenarios()
        self.test_stock_info()
        self.test_market_summary()
        self.test_sentiment_analysis()
        self.test_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")

def main():
    """Main test function"""
    tester = FinanceAgentTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
