from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import List, Optional, Any

model_name = "openai/gpt-oss-20b:free"
base_url = "https://openrouter.ai/api/v1"
api_key = "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"


class OpenRouterChat(BaseChatModel):
    """Custom LLM class to call OpenRouter API."""
    
    model_name: str = model_name
    api_key: str = api_key
    base_url: str = base_url
    temperature: float = 0.5
    client: Optional[OpenAI] = None  # Pydantic field
    
    def __init__(self, **kwargs: Any):
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


llm = OpenRouterChat()

# FIX: memory_key="history" để match default prompt của ConversationChain
memory = ConversationBufferWindowMemory(
    memory_key="history_a", 
    return_messages=True,
    k=5
)

#  Custom prompt để tương thích hoàn hảo (tùy chọn, nhưng recommend)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history_a"),
    ("human", "{input}")
])

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt  
)

# GIẢ SỬ: ĐÃ CÓ llm, memory, prompt, conversation (từ code trước)

print("🤖 BOT SẴN SÀNG! Gõ 'quit' để thoát.\n")

while True:
    user_input = input("👤 Bạn: ")  # Nhập từ bàn phím
    if user_input.lower() == 'quit':
        print("👋 Tạm biệt!")
        break
    
    reply = conversation.predict(input=user_input)  # ← CHẠT 1 LẦN
    print(f"🤖 Bot: {reply}\n")
    
    # (Tùy chọn) In history hiện tại
    print(f"📝 History ({len(memory.chat_memory.messages)} tin): {[m.content for m in memory.chat_memory.messages]}\n")