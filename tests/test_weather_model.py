import pytest
from unittest.mock import Mock, patch
from src.models.weather_model import WeatherModel

def test_init():
    model = WeatherModel("test_key")
    assert model.api_key == "test_key"

def test_fetch_weather_success():
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.json.return_value = {"weather": [{"description": "sunny"}], "main": {"temp": 25}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        model = WeatherModel("fake_key")
        data = model.fetch_weather("London")
        
        assert "weather" in data
        assert data["main"]["temp"] == 25
        mock_get.assert_called_once()

def test_fetch_weather_no_key():
    model = WeatherModel("")
    data = model.fetch_weather("London")
    assert "error" in data
    assert data["error"] == "API key not configured."

def test_fetch_weather_failure():
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("Network fail")
        
        model = WeatherModel("fake_key")
        data = model.fetch_weather("London")
        
        assert "error" in data
        assert "Network fail" in data["error"]
