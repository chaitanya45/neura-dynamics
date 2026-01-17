import pytest
from unittest.mock import MagicMock, patch
from src.controllers.workflow_controller import WorkflowController
from src.graph.state import AgentState

@pytest.fixture
def mock_controller():
    with patch('src.controllers.workflow_controller.WeatherModel'), \
         patch('src.controllers.workflow_controller.RAGModel'), \
         patch('src.controllers.workflow_controller.LLMService') as MockLLM:
        
        # Instantiate controller with mocked deps
        # Since we patched the classes, init() uses the mocks
        controller = WorkflowController()
        # The instances on the controller are the return values of the class mocks
        # But we need to access the specific instance mocks
        # Re-assigning for clarity in tests or we can configure the class mocks
        return controller

def test_determine_intent_weather(mock_controller):
    # Setup mock
    mock_controller.llm_service.classify_intent.return_value = "weather"
    
    state = {"query": "Is it raining?"}
    result = mock_controller.determine_intent(state)
    
    assert result["intent"] == "weather"

def test_determine_intent_document(mock_controller):
    mock_controller.llm_service.classify_intent.return_value = "document"
    
    state = {"query": "Summarize pdf"}
    result = mock_controller.determine_intent(state)
    
    assert result["intent"] == "document"

def test_handle_weather_flow(mock_controller):
    # Setup mocks
    mock_controller.llm_service.extract_city.return_value = "London"
    mock_controller.weather_model.fetch_weather.return_value = {"temp": 10}
    mock_controller.llm_service.summarize_weather.return_value = "It is cold."
    
    state = {"query": "Weather in London"}
    result = mock_controller.handle_weather(state)
    
    assert result["response"] == "It is cold."
    assert result["weather_data"] == {"temp": 10}

def test_handle_weather_no_city(mock_controller):
    mock_controller.llm_service.extract_city.return_value = ""
    
    state = {"query": "Weather somewhere"}
    result = mock_controller.handle_weather(state)
    
    assert "could not identify" in result["response"]

def test_handle_rag_flow(mock_controller):
    mock_docs = [MagicMock(page_content="doc1")]
    mock_controller.rag_model.retrieve_context.return_value = mock_docs
    mock_controller.llm_service.rag_response.return_value = "Summary of doc"
    
    state = {"query": "What does the doc say?"}
    result = mock_controller.handle_rag(state)
    
    assert result["response"] == "Summary of doc"
    assert result["rag_context"] == mock_docs
