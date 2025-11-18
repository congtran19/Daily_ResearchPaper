import os
import faiss
from typing import List
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from data.models import Paper
from data.repositories.base_repository import SQLRepository  

class VectorRepository:
    def __init__(self, index_path="vector_index/faiss_index", model_name="all-MiniLM-L6-v2", ):
        self.index_path = index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path):
            return FAISS.load_local(self.index_path, self.embeddings)

        dim = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(dim)

        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )
    def add_papers(self, documents: List[Document]):
        """
        Thêm paper vào vector store nếu chưa có trong SQL.
        Chỉ tạo Document từ paper chưa tồn tại.
        """
        self.vectorstore.add_documents(documents= documents)
        self.vectorstore.save_local(self.index_path)
        print(f"Added 1 new papers to vector store")