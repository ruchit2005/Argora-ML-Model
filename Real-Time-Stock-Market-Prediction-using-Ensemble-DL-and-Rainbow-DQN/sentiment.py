import requests
from bs4 import BeautifulSoup
import json

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


news_headlines = []
# Create a SentimentIntensityAnalyzer object. 
sid_obj = SentimentIntensityAnalyzer() 

def print_headlines(response_text):
    soup = BeautifulSoup(response_text, 'lxml')
    headlines = soup.find_all(attrs={"itemprop": "headline"})
    for headline in headlines:
        news_headlines.append(headline.text)


def get_headers():
    return {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-IN,en-US;q=0.9,en;q=0.8",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "cookie": "_ga=GA1.2.474379061.1548476083; _gid=GA1.2.251903072.1548476083; __gads=ID=17fd29a6d34048fc:T=1548476085:S=ALNI_MaRiLYBFlMfKNMAtiW0J3b_o0XGxw",
        "origin": "https://inshorts.com",
        "referer": "https://inshorts.com/en/read/",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
        "x-requested-with": "XMLHttpRequest"
    }

def news2sentiment(stock_symbol, api_key):
    """
    Fetches news about the stock_symbol and returns sentiment scores.
    """
    news_headlines = []

    # NewsAPI endpoint
    api_key = "e3dea6dae69341f2adb88979f6cbf32f"
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&sortBy=publishedAt&language=en&apiKey={api_key}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch news: {response.status_code}")
            return []

        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            print("No news articles returned")
            return []

        # Extract headlines
        for article in articles[:50]:  # limit to 50 articles
            title = article.get("title")
            if title:
                news_headlines.append(title)

    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

    # Compute sentiment
    scores = [sid_obj.polarity_scores(headline)["compound"] for headline in news_headlines]
    return scores