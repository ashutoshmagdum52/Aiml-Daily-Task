import os
import requests
import json
import re
import time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWSDATA_API_KEY")


class NewsClassifier:
    def __init__(self, config_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_config_path = os.path.join(base_dir, config_path)

        with open(full_config_path, 'r') as f:
            self.config = json.load(f)
        self.categories = self.config['categories']

    def classify(self, text):
        text = text.lower()
        scores = {}

        for cat, keywords in self.categories.items():
            count = 0
            for word in keywords:
                if word in text:
                    count += 1
            scores[cat] = count

        best_cat = max(scores, key=scores.get)

        if scores[best_cat] == 0:
            return "GENERAL"
        return best_cat

    def extract_entities(self, text):
        # Known major companies to match against
        known_companies = [
            "Apple", "Google", "Microsoft", "Tesla", "Amazon", "Meta",
            "Netflix", "Twitter", "Samsung", "Intel", "Nvidia", "IBM",
            "Goldman Sachs", "JPMorgan", "BlackRock", "Alibaba", "Tata",
            "Reliance", "Infosys", "Wipro", "HDFC", "Adani"
        ]

        found_companies = []
        for company in known_companies:
            if company.lower() in text.lower():
                found_companies.append(company)

        entities = {
            "Companies": found_companies,
            "Currency": list(set(re.findall(r'[\$\£\€\₹¥]', text))),
            "Percentages": list(set(re.findall(r'\d+(?:\.\d+)?%', text)))
        }
        return entities


def fetch_news(keyword, count=50):
    results = []
    url = f"https://newsdata.io/api/1/latest?apikey={API_KEY}&q={keyword}"

    try:
        response = requests.get(url)

        if response.status_code == 429:
            print("Rate limit hit. Waiting 5 seconds before retrying...")
            time.sleep(5)
            response = requests.get(url)

        response.raise_for_status()
        data = response.json()
        articles = data.get("results", [])
        return articles[:count]

    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


if __name__ == "__main__":
    if not API_KEY:
        print("Error: NEWSDATA_API_KEY not found in .env file.")
    else:
        classifier = NewsClassifier("news_config.json")

        keyword = "Global Economy"
        print(f"Fetching latest news for: '{keyword}'...")

        articles = fetch_news(keyword, count=50)
        processed_news = []

        for art in articles:
            title = art.get("title", "")
            description = art.get("description", "") or ""
            full_text = f"{title} {description}"

            category = classifier.classify(full_text)
            entities = classifier.extract_entities(full_text)

            processed_news.append({
                "title": title,
                "category": category,
                "entities": entities,
                "link": art.get("link")
            })

            print(f"[{category}] {title[:60]}...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "news_data_day2.json")

        with open(output_path, "w") as f:
            json.dump(processed_news, f, indent=4)

        print(f"\nProcessed {len(processed_news)} articles.")
        print(f"Saved to: {output_path}")
