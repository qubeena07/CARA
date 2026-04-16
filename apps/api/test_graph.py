import uuid
from dotenv import load_dotenv
load_dotenv()  # must be before any other imports that need env vars

from database.connection import SessionLocal
from graph.graph import build_graph

db = SessionLocal()
graph = build_graph(db)

initial_state = {
    "question": "what is a transformer model?",
    "session_id": str(uuid.uuid4()),
    "query_id": str(uuid.uuid4()),
    "retrieved_documents": [],
    "rewritten_question": None,
    "retry_count": 0,
    "regeneration_count": 0,
    "generation": None,
    "hallucination_detected": False,
    "confidence_score": 0.0,
    "audit_log": [],
    "web_search_needed": False,
}

result = graph.invoke(initial_state)
print("\n=== GENERATION ===")
print(result["generation"])
print("\n=== AUDIT LOG ===")
print(len(result["audit_log"]), "audit entries")
for entry in result["audit_log"]:
    print(f"  {entry['step']}: {entry['message']}")
db.close()
