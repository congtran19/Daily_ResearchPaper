from langchain.embeddings import HuggingFaceEmbeddings  # Hoặc from langchain_openai import OpenAIEmbeddings nếu dùng OpenAI
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model. Sử dụng HuggingFace cho local/free.
        Nếu dùng OpenAI: OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch embed list of texts."""
        return self.embeddings.embed_documents(texts)