from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from openai import OpenAI
from data.repositories.FaissVectorRepository import VectorRepository
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI, ChatOllama


class RAGService:
    def __init__(
        self,
        vector_repo: Optional[VectorRepository] = None,
        llm_model: str = "gpt-4o-mini", # hoặc qwen2.5, llama3,...
        temperature: float = 0.0,
        default_k: int = 4,
    ):
        self.vector_repo = vector_repo or VectorRepository()
        self.default_k = default_k

        if self.vector_repo.vectorstore is None:
            raise RuntimeError("❌ Vector DB chưa load. Chạy embed trước.")

        # === chọn LLM chuẩn v0.2 ===
        if llm_provider.lower() == "openai":
            # ⚠️ yêu cầu export OPENAI_API_KEY
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature,
            )
        elif llm_provider.lower() == "ollama":
            self.llm = ChatOllama(
                model=llm_model,
                temperature=temperature,
            )
        else:
            raise ValueError("llm_provider phải là 'openai' hoặc 'ollama'")

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer only using the context. If missing info, say 'I don't know'.\n\n"
                "=== Context ===\n{context}\n\n"
                "=== Question ===\n{question}\n\nAnswer:\n"
            )
        )

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, d in enumerate(docs, start=1):
            title = d.metadata.get("title") if d.metadata else "No Title"
            url = d.metadata.get("url") if d.metadata else "No URL"
            parts.append(f"[{i}] {title} ({url})\n{d.page_content}")
        return "\n\n".join(parts)

    def _extract_sources(self, docs: List[Document]) -> List[Dict[str, str]]:
        sources = []
        for i, d in enumerate(docs, start=1):
            sources.append({
                "tag": f"[{i}]",
                "title": d.metadata.get("title"),
                "url": d.metadata.get("url"),
            })
        return sources

    def query(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        k = k or self.default_k
        docs = self.vector_repo.vectorstore.similarity_search(question, k=k)

        if not docs:
            return {"answer": "I don't know.", "sources": [], "docs": []}

        context = self._format_context(docs)
        prompt = self.prompt_template.format(context=context, question=question)

        # ✅ ChatOpenAI trả về message.content
        answer = self.llm.invoke(prompt).content

        return {
            "answer": answer,
            "sources": self._extract_sources(docs),
            "docs": docs,
        }
