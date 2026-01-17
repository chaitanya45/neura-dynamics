import requests
from typing import Dict, Any, Optional

class WeatherModel:
    """
    Model for fetching weather data from OpenWeatherMap API.
    Refrains from business logic or complex decision making.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    def fetch_weather(self, city: str) -> Dict[str, Any]:
        """
        Fetches weather data for a specific city.
        
        Args:
            city (str): Name of the city.
            
        Returns:
            Dict[str, Any]: Raw weather data or error info.
        """
        if not self.api_key:
            return {"error": "API key not configured."}

        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
             return {"error": f"HTTP error occurred: {http_err}", "status_code": response.status_code}
        except Exception as err:
            return {"error": f"An error occurred: {err}"}
