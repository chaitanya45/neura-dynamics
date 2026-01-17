from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

class LLMService:
    """
    Service for handling LLM interactions using LangChain.
    Includes classification, weather summarization, and RAG QA.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def classify_intent(self, query: str) -> str:
        """
        Decides if the query is about 'weather' or 'document'.
        """
        system_prompt = """You are a helpful assistant. Classify the user query into one of two categories: 'weather' or 'document'.
        - 'weather': Questions about current weather, temperature, forecast, etc. for a specific location.
        - 'document': General questions, requests for information, summarization, or questions that might need external knowledge from a loaded PDF.
        
        Return ONLY one word: 'weather' or 'document'.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # We can add explicit LangSmith tags here via config
        return chain.invoke({"query": query}, config=RunnableConfig(run_name="classify_intent"))

    def summarize_weather(self, query: str, weather_data: dict) -> str:
        """
        Summarizes weather data in response to a user query.
        """
        system_prompt = """You are a weather assistant. Given the user query and the raw weather JSON data, provide a natural language summary of the weather.
        Be concise and helpful.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Query: {query}\nData: {data}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "data": str(weather_data)}, config=RunnableConfig(run_name="summarize_weather"))

    def rag_response(self, query: str, context: list) -> str:
        """
        Generates a response based on retrieved context.
        """
        system_prompt = """You are a helpful assistant analyzing an uploaded document. 
        Use the following pieces of retrieved context to answer the user's question.
        
        If the user asks for a summary or general information about the document (Example: "tell me about this pdf"), use the provided chunks to give a high-level overview.
        If the answer is not in the context, say "I don't have enough information in the uploaded document to answer that."
        
        Context:
        {context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        # Flatten context
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context_text}, config=RunnableConfig(run_name="rag_response"))

    def extract_city(self, query: str) -> str:
        """
        Extracts the city name from a weather-related query.
        """
        system_prompt = """You are an entity extractor. Extract the city name from the user's query.
        Return ONLY the city name. If no city is found, return nothing.
        Example: "What's the weather in London?" -> London
        Example: "Forecast for Paris, France" -> Paris
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}, config=RunnableConfig(run_name="extract_city")).strip()
