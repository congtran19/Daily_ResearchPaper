
import os
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from data.repositories.FaissVectorRepository import VectorRepository
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from LLM.model_via_openrouter import OpenRouterChat
import logging
load_dotenv()

class RAGService:
    """RAG Service: FAISS + OpenRouterChat + Memory + Citations."""
    
    def __init__(
        self,
        faiss_path: str = "vector_index/faiss_index",
        top_k: int = 3,
        memory_k: int = 5,
        verbose: bool = False
    ):
        # 1. LLM
        self.llm = OpenRouterChat()
        
        # 2. Vector DB
        self.repo = VectorRepository()
        self.embeddings = self.repo.embeddings
        self.vectorstore = self.repo.vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        # 3. Memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=memory_k,
            input_key="question",
            output_key="answer"
        )
        
        # 4. Prompt (đúng format cho ConversationalRetrievalChain)
        self.prompt = ChatPromptTemplate.from_template(
            """
System: Dùng context để trả lời chính xác. Nếu có thông tin, hãy trích dẫn [arxiv_id] ở cuối câu. Nếu không biết, trả lời: "Không tìm thấy info." Trả lời ngắn gọn, dưới 3 câu.

Context: {context}

Chat History: {chat_history}

Human: {question}

Assistant: 
            """
        )
        
        # 5. RAG Chain (KHÔNG dùng input_key/output_key - đã bị loại bỏ)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            verbose=verbose,
            return_source_documents=True
        )
        
        
        print(f"✅ RAGService ready! FAISS: {faiss_path}, Top-K: {top_k}")
    
    def chat(self, question: str) -> Dict[str, Any]:
        """
        Chat RAG: question → answer + sources + citations.
        
        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'chat_history': List
            }
        """
        # Gọi chain
        result = self.chain({"question": question})
        self.debug_retrieve(question)
        # Lấy source documents
        source_docs = result.get("source_documents", [])

        # Trích xuất sources
        sources = [
            {
                "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata
            }
            for doc in source_docs
        ]
        
        # Trích dẫn arxiv_id (nếu có)
        arxiv_ids = []
        arxiv_urls = []
        for doc in source_docs:
            arxiv_id = doc.metadata.get("title")
            arxiv_url = doc.metadata.get("url")
            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_urls.append(arxiv_url)
        
        # Thêm trích dẫn vào answer
        answer = result["answer"].strip()
        if arxiv_ids:
            answer += "\n\nTrích dẫn: " + ", ".join(f"[{aid}]" for aid in arxiv_ids)
        elif not answer.lower().__contains__("không tìm thấy"):
            answer += "\n\n[Không có trích dẫn khả dụng]"
        
        return {
            "answer": answer,
            "sources": sources,
            "chat_history": self.memory.chat_memory.messages
        }
    def debug_retrieve(self, question: str):
        """Trả về list (doc, score) để xem rõ threshold."""
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            question,
            k=self.retriever.search_kwargs.get("k", 3)
        )
        
        print("\n🔍 DEBUG SCORES:")
        for i, (doc, score) in enumerate(docs_with_scores, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Score: {1/(1+score)}")
            print(f"Metadata: {doc.metadata}")
            print(f"Content: {doc.page_content[:200]}...")
        
        return docs_with_scores
    def clear_history(self):
        """Xóa lịch sử chat."""
        self.memory.clear()
        print("🧹 Chat history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy stats."""
        return {
            "faiss_path": self.repo.db_path,
            "vector_count": len(self.vectorstore.index_to_docstore_id),
            "top_k": self.retriever.search_kwargs.get("k"),
            "memory_k": self.memory.k,
            "chat_history_length": len(self.memory.chat_memory.messages)
        }