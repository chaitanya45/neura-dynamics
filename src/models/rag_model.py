import os
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class RAGModel:
    """
    Model for handling PDF loading, embedding generation, and Vector DB operations.
    """
    def __init__(self, qdrant_path: str, collection_name: str = "neura_docs"):
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        
        # Ensure directory exists
        os.makedirs(self.qdrant_path, exist_ok=True)
        
        self.client = QdrantClient(path=self.qdrant_path)
        
        # Initialize collection if not exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def load_and_process_pdf(self, file_path: str) -> str:
        """
        Loads a PDF, splits it, and adds it to the vector store.
        """
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
            
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        self.vector_store.add_documents(documents=splits)
        return f"Successfully processed {len(splits)} chunks from {file_path}."

    def retrieve_context(self, query: str, k: int = 4) -> List[Any]:
        """
        Retrieves relevant documents for a query.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)
