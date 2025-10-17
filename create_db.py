import json
import os
from langchain.schema import Document
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Custom embeddings
class SentenceTransformerEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Project root path (adjust if script not in root)
project_root = os.path.dirname(os.path.abspath(__file__))  # Assumes script in root
db_path = os.path.join(project_root, "chroma_db")

# Load FAQs
faqs_path = os.path.join(project_root, "faqs.json")
with open(faqs_path, "r", encoding="utf-8") as f:
    data = json.load(f)
docs = [Document(page_content=f"{item['question']}\n{item['answer']}") for item in data]
print(f"[DEBUG] Loaded {len(docs)} FAQ docs from {faqs_path}", flush=True)

embeddings = SentenceTransformerEmbeddings()

# Create and persist with absolute path

print(f"[DEBUG] Creating Chroma vectorstore at {db_path}", flush=True)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=db_path  # Absolute now
)
print(f"[DEBUG] Vectorstore created and persisted to {db_path}! Total docs: {vectorstore._collection.count()}", flush=True)

# Verify files created
sqlite_path = os.path.join(db_path, "chroma.sqlite3")
print(f"[DEBUG] Checking for {sqlite_path}", flush=True)
if os.path.exists(sqlite_path):
    print("SUCCESS: chroma.sqlite3 exists.", flush=True)
else:
    print("WARNING: DB not saved—check permissions/OneDrive sync.", flush=True)