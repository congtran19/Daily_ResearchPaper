from services.rag_service import RAGService
service = RAGService()
print("🤖 RAG ready!")
while True:
    q = input("\n👤: ").strip()
    if q.lower() == 'quit': break
    
    result = service.chat(q)
    print(f"\n🤖: {result['answer']}")