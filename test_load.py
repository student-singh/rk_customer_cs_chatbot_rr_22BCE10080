
import os
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

print("[DEBUG] test_load.py starting...", flush=True)

# Custom embeddings (same)
class SentenceTransformerEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

try:
    # Absolute paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(project_root, "chroma_db")
    print(f"[DEBUG] Testing load from: {db_path}", flush=True)

    if not os.path.exists(db_path):
        print("ERROR: chroma_db folder not found—run create_db.py first.", flush=True)
    else:
        embeddings = SentenceTransformerEmbeddings()
        print("[DEBUG] Loading Chroma vectorstore...", flush=True)
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        doc_count = vectorstore._collection.count()
        print(f"[DEBUG] Loaded docs count: {doc_count}", flush=True)
        
        # Quick retrieval test
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("[DEBUG] Running retrieval test...", flush=True)
        results = retriever.invoke("What is delivery modes available?")
        print(f"[DEBUG] Retrieved {len(results)} docs:", flush=True)
        for i, doc in enumerate(results, 1):
            print(f"  {i}: {doc.page_content[:100]}...", flush=True)
        
        if doc_count == 0:
            print("ERROR: Empty DB—re-run create_db.py with explicit persist.", flush=True)
except Exception as e:
    print(f"[ERROR] Exception: {e}", flush=True)