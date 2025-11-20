
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List, Optional, Any
from langchain_core.callbacks import CallbackManagerForLLMRun
from dotenv import load_dotenv
import os

load_dotenv()

# Env vars (từ .env của bạn)
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
api_key = os.getenv("OPENROUTER_API_KEY", "sk-or-v1-...")  


class OpenRouterChat(BaseChatModel):
    """Custom LLM class to call OpenRouter API."""
    
    model_name: str = model_name  # ← Auto từ .env
    api_key: str = api_key        # ← Auto từ .env
    base_url: str = base_url      # ← Auto từ .env
    temperature: float = 0.5
    client: Optional[OpenAI] = None
    
    def __init__(self, **kwargs: Any):
        # Override từ kwargs nếu có (flexible)
        if "model_name" in kwargs:
            self.model_name = kwargs["model_name"]
        if "api_key" in kwargs:
            self.api_key = kwargs["api_key"]
        if "base_url" in kwargs:
            self.base_url = kwargs["base_url"]
            
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages -> OpenAI/OpenRouter format"""
        role_map = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
            "function": "function",
            "observation": "user",
        }
        return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Send messages to OpenRouter API and return the reply."""
        if self.client is None:
            raise ValueError("Client not initialized. Check __init__.")
        
        formatted_messages = self._convert_messages(messages)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=500,
            stop=stop,
        )
        content = resp.choices[0].message.content
        ai_message = AIMessage(content=content)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
