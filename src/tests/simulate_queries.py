from sentence_transformers import SentenceTransformer
from src.rag import get_chroma_collection, ask, ChatMemory

embed = SentenceTransformer("all-MiniLM-L6-v2")
collection = get_chroma_collection()
memory = ChatMemory()

test_queries = [
    "How do I request a refund?",
    "Refund?",
    "App not working",
    "Tell me a joke",
    "How do I cancel subscription?",
    "Payment issue",
    "What is the weather?",
]

for q in test_queries:
    result = ask(q, collection, embed, memory=memory)
    print(f"{q} → {result['state']} ({result['reason']})")