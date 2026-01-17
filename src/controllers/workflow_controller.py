from typing import Dict, Any
from src.utils.config import Config
from src.models.weather_model import WeatherModel
from src.models.rag_model import RAGModel
from src.services.llm_service import LLMService
from src.graph.state import AgentState

class WorkflowController:
    """
    Controller that implements the logic for each step in the graph workflow.
    Orchestrates calls to Models and Services.
    """
    def __init__(self):
        self.weather_model = WeatherModel(api_key=Config.OPENWEATHER_API_KEY)
        self.rag_model = RAGModel(qdrant_path=Config.QDRANT_PATH)
        self.llm_service = LLMService()

    def determine_intent(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyzes the query and determines the intent.
        """
        query = state.get("query", "")
        intent = self.llm_service.classify_intent(query).lower().strip()
        
        if "weather" in intent:
            intent = "weather"
        else:
            intent = "document"
            
        print(f"Decided Intent: {intent}")
        return {"intent": intent}

    def handle_weather(self, state: AgentState) -> Dict[str, Any]:
        """
        Handles the weather flow: fetches data and summarizes.
        """
        query = state.get("query", "")
        city = self.llm_service.extract_city(query)
        
        if not city:
            return {"response": "I could not identify the city for the weather request."}
            
        weather_data = self.weather_model.fetch_weather(city)
        if "error" in weather_data:
            return {"response": f"Error fetching weather: {weather_data['error']}", "weather_data": weather_data}
            
        summary = self.llm_service.summarize_weather(query, weather_data)
        return {"weather_data": weather_data, "response": summary}

    def handle_rag(self, state: AgentState) -> Dict[str, Any]:
        """
        Handles the RAG flow: retrieve and condense.
        """
        query = state.get("query", "")
        retrieved_docs = self.rag_model.retrieve_context(query)
        response = self.llm_service.rag_response(query, retrieved_docs)
        return {"rag_context": retrieved_docs, "response": response}
