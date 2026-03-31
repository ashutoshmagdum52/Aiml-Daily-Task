import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Ahmedabad", "Pune", "Jaipur", "Lucknow"
]

def process_city(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        resp = requests.get(url)

        if resp.status_code == 404:
            print(f"City not found: {city}")
            return None

        if resp.status_code == 401:
            print("Invalid API Key. Please check your .env file.")
            return None

        resp.raise_for_status()
        data = resp.json()

        city_data = {
            "city": data["name"],
            "temp_c": round(data["main"]["temp"] - 273.15, 2),
            "humidity": data["main"]["humidity"],
            "weather": data["weather"][0]["main"],
            "last_updated": datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')
        }

        print(f"Done: {city_data['city']} | {city_data['temp_c']}°C | {city_data['weather']}")
        return city_data

    except Exception as e:
        print(f"Error for {city}: {e}")
        return None


if __name__ == "__main__":
    if not API_KEY or API_KEY == "your_api_key_here":
        print("Please update your .env file with your API key.")
    else:
        results = []
        for city in CITIES:
            city_data = process_city(city)
            if city_data is not None:
                results.append(city_data)

        with open("weather_data.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nSaved weather data for {len(results)} cities to weather_data.json")
