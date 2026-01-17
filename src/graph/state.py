from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.documents import Document

class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    weather_data: Optional[Dict[str, Any]]
    rag_context: Optional[List[Document]]
    response: Optional[str]
    error: Optional[str]
