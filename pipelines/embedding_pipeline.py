from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from data.repositories.base_repository import SQLRepository
from data.repositories.FaissVectorRepository import VectorRepository  # Sửa import nếu cần
from services.embedding_service import EmbeddingService
from data.models import Paper
from typing import List
import hashlib  # Để tạo paper_id unique

import json
import os

class EmbeddingPipeline:
    def __init__(self):
        self.vector_repo = VectorRepository()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embedding_service = EmbeddingService()

    def _chunk_abstract(self, paper: Paper) -> List[Document]:
        chunks = self.text_splitter.split_text(paper.abstract)
        documents = []
        paper_id = hashlib.sha256(paper.url.encode()).hexdigest()

        for i, chunk_text in enumerate(chunks):
            metadata = {
                "id": paper_id,
                "chunk_index": i,
                "title": paper.title,
                "source": paper.source,
                "url": paper.url,
                "published_date": paper.published_date.isoformat() if paper.published_date else None
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))
        return documents

    def run(self, papers: List[Paper] = None, incremental: bool = True):
        if papers is None:
            if incremental:
                raise ValueError("Pass list papers mới để add incremental, tránh process hết JSON.")
        all_documents = []
        for paper in papers:
            if not paper.abstract:
                continue
            chunks = self._chunk_abstract(paper)
            all_documents.extend(chunks)
        # Nếu không có gì mới -> thoát nhẹ
        if not all_documents:
            print("👍 Không có tài liệu mới cần embed.")
            return

        # Lưu vector
        self.vector_repo.add_papers(all_documents)
        print(f"✅ Added {len(all_documents)} new chunks into vector DB.")
