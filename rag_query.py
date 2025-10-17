# import sys
# import json
# import os
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from sentence_transformers import SentenceTransformer
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage
# from dotenv import load_dotenv

# load_dotenv()

# # Custom embeddings class
# class SentenceTransformerEmbeddings:
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model = SentenceTransformer(model_name)
#     def embed_documents(self, texts):
#         return self.model.encode(texts).tolist()
#     def embed_query(self, text):
#         return self.model.encode([text])[0].tolist()

# # Absolute paths (Windows-friendly)
# project_root = os.path.dirname(os.path.abspath(__file__))
# db_path = os.path.join(project_root, "chroma_db")
# faqs_path = os.path.join(project_root, "faqs.json")

# # Load FAQs
# with open(faqs_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
# docs = [Document(page_content=f"{item['question']}\n{item['answer']}") for item in data]
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# split_docs = text_splitter.split_documents(docs)

# # Load persisted vectorstore
# if not os.path.exists(db_path):
#     print(f"[ERROR] chroma_db not found at {db_path}—run create_db.py first.", file=sys.stderr)
#     sys.exit(1)

# embeddings = SentenceTransformerEmbeddings()
# vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# # Debug doc count
# doc_count = vectorstore._collection.count()
# print(f"[DEBUG] Loaded vectorstore docs count: {doc_count} from {db_path}", file=sys.stderr)
# if doc_count == 0:
#     print("[ERROR] Empty vectorstore—re-persist.", file=sys.stderr)
#     sys.exit(1)

# # LLM setup
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite",
#     temperature=0.3,
#     max_tokens=500,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# # Fixed system prompt (no {history} var—use full_input for context)
# system_prompt = (
#     "You are a helpful customer support bot for question-answering. "
#     "The input includes conversation history for context—always refer to it (e.g., if the user says 'tell me more about this', "
#     "expand on the most recent topic like refunds or returns from the history). "
#     "Use the retrieved FAQ context to answer the question concisely (3 sentences max). "
#     "If unsure from FAQs but history helps, use history to respond. "
#     "If truly unknown, say 'I don't have details on that—let me escalate to a human.'\n\n"
#     "Retrieved context: {context}"
# )

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{input}"),
# ])

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# # Input handling
# try:
#     input_str = sys.stdin.read().strip()
#     print(f"[DEBUG] Received input: {input_str}", file=sys.stderr)
#     input_data = json.loads(input_str)
#     query = input_data["query"]
#     history = input_data.get("history", "")

#     # Summarize long history
#     if len(history) > 300:
#         summary_prompt = f"Summarize key topics from this chat history concisely: {history}"
#         summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
#         history = summary
#         print(f"[DEBUG] History summarized: {history[:100]}...", file=sys.stderr)

#     full_input = f"{history}\nUser: {query}" if history else query
#     print(f"[DEBUG] Full input to RAG: {full_input[:200]}...", file=sys.stderr)

#     # Retrieval debug
#     results = vectorstore.similarity_search_with_score(full_input, k=5)
#     print(f"[DEBUG] Retrieval scores for query: {full_input[:100]}...", file=sys.stderr)
#     for i, (doc, score) in enumerate(results, 1):
#         print(f"[DEBUG] Result {i} score={score:.4f}: {doc.page_content[:100]}...", file=sys.stderr)
#     top_score = results[0][1] if results else 1.0
#     print(f"[DEBUG] Top score: {top_score} (threshold: 0.85)", file=sys.stderr)

#     threshold = 0.85  # Raised: More lenient for vague/contextual queries

#     if top_score > threshold:
#         print("[DEBUG] Escalation triggered—using history fallback.", file=sys.stderr)
#         # Fallback: Summarize history + generate from it
#         if len(history) > 0:
#             # Extract last topic (simple: last bot message)
#             history_lines = history.split('\n')
#             last_bot_msg = next((line for line in reversed(history_lines) if line.startswith('Bot:')), "recent discussion")
#             fallback_prompt = f"Based on this history: {history}\nUser follow-up: {query}\nExpand or summarize the {last_bot_msg} concisely (3 sentences max)."
#             fallback_response = llm.invoke([HumanMessage(content=fallback_prompt)]).content
#             response = {
#                 "answer": fallback_response,
#                 "escalate": True,  # Flag for logging
#                 "suggestion": "For more details, check our Returns page."
#             }
#         else:
#             response = {
#                 "answer": "I'm sorry, I don't have enough information. Let me escalate you to a human agent.",
#                 "escalate": True,
#                 "suggestion": "In the meantime, check our FAQ page."
#             }
#     else:
#         print("[DEBUG] Proceeding to RAG chain.", file=sys.stderr)
#         # Invoke with full_input as {input}—history is embedded
#         rag_response = rag_chain.invoke({"input": full_input})
#         answer = rag_response["answer"]
#         if len(history) > 500:
#             summary_prompt = f"Summarize this conversation concisely: {history}"
#             summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
#             history = summary
#         response = {
#             "answer": answer,
#             "escalate": False,
#             "suggestion": "Is there anything else I can help with?"
#         }

#     print(json.dumps(response))
#     sys.stdout.flush()
# except json.JSONDecodeError as e:
#     print(f"[ERROR] Invalid JSON: {e}", file=sys.stderr)
#     print(json.dumps({"answer": "Invalid input.", "escalate": False, "suggestion": ""}))
#     sys.stdout.flush()
# except Exception as e:
#     print(f"[ERROR] {str(e)}", file=sys.stderr)
#     print(json.dumps({"answer": "Sorry, an error occurred.", "escalate": False, "suggestion": ""}))
#     sys.stdout.flush()

















import sys
import json
import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Custom embeddings class
class SentenceTransformerEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Absolute paths (Windows-friendly)
project_root = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_root, "chroma_db")
faqs_path = os.path.join(project_root, "faqs.json")

# Load FAQs
with open(faqs_path, "r", encoding="utf-8") as f:
    data = json.load(f)
docs = [Document(page_content=f"{item['question']}\n{item['answer']}") for item in data]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
split_docs = text_splitter.split_documents(docs)

# Load persisted vectorstore
if not os.path.exists(db_path):
    print(f"[ERROR] chroma_db not found at {db_path}—run create_db.py first.", file=sys.stderr)
    sys.exit(1)

embeddings = SentenceTransformerEmbeddings()
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Debug doc count
doc_count = vectorstore._collection.count()
print(f"[DEBUG] Loaded vectorstore docs count: {doc_count} from {db_path}", file=sys.stderr)
if doc_count == 0:
    print("[ERROR] Empty vectorstore—re-persist.", file=sys.stderr)
    sys.exit(1)

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_tokens=500,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Enhanced system prompt: Always blend history + FAQs
system_prompt = (
    "You are a helpful customer support bot. "
    "Combine the conversation history (for context and follow-ups) with the retrieved FAQ context (for accurate facts) to provide a comprehensive, concise answer (3 sentences max). "
    "For direct questions, focus on FAQ details. For follow-ups (e.g., 'tell me more about this'), reference the most recent history topic while grounding in FAQs. "
    "If history and FAQs align, blend seamlessly. If unclear, prioritize FAQs and note from history.\n\n"
    "Retrieved FAQ context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),  # full_input (history + query) goes here
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Input handling
try:
    input_str = sys.stdin.read().strip()
    print(f"[DEBUG] Received input: {input_str}", file=sys.stderr)
    input_data = json.loads(input_str)
    query = input_data["query"]
    history = input_data.get("history", "")

    # Summarize long history
    if len(history) > 300:
        summary_prompt = f"Summarize key topics from this chat history concisely: {history}"
        summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
        history = summary
        print(f"[DEBUG] History summarized: {history[:100]}...", file=sys.stderr)

    full_input = f"{history}\nUser: {query}" if history else query
    print(f"[DEBUG] Full input to RAG: {full_input[:200]}...", file=sys.stderr)

    # Always retrieve (using full_input for semantic blend)
    results = vectorstore.similarity_search_with_score(full_input, k=5)
    print(f"[DEBUG] Retrieval scores for query: {full_input[:100]}...", file=sys.stderr)
    for i, (doc, score) in enumerate(results, 1):
        print(f"[DEBUG] Result {i} score={score:.4f}: {doc.page_content[:100]}...", file=sys.stderr)
    top_score = results[0][1] if results else 1.0
    print(f"[DEBUG] Top score: {top_score} (threshold: 0.85)", file=sys.stderr)

    threshold = 0.85  # For hybrid fallback if very poor

    if top_score > threshold:
        print("[DEBUG] Poor retrieval—hybrid fallback (RAG + history refine).", file=sys.stderr)
        # Hybrid: Run RAG, then refine with extra LLM call emphasizing FAQs + history
        rag_response = rag_chain.invoke({"input": full_input})
        initial_answer = rag_response["answer"]
        
        # Refine: Blend with explicit FAQ/history prompt
        refine_prompt = f"Refine this RAG answer using FAQs and history: {initial_answer}\nHistory: {history}\nQuery: {query}\nMake it concise, factual, and contextual (3 sentences)."
        refined_response = llm.invoke([HumanMessage(content=refine_prompt)]).content
        answer = refined_response
        escalate = True  # Flag if needed
        suggestion = "For more details, check our Returns page."
    else:
        print("[DEBUG] Good retrieval—full RAG blend.", file=sys.stderr)
        # Standard RAG: Already blends via full_input + {context}
        rag_response = rag_chain.invoke({"input": full_input})
        answer = rag_response["answer"]
        escalate = False
        suggestion = "Is there anything else I can help with?"

    # Post-process if long history
    if len(history) > 500:
        summary_prompt = f"Summarize this conversation concisely: {history}"
        summary = llm.invoke([HumanMessage(content=summary_prompt)]).content
        history = summary

    response = {
        "answer": answer,
        "escalate": escalate,
        "suggestion": suggestion
    }

    print(json.dumps(response))
    sys.stdout.flush()
except json.JSONDecodeError as e:
    print(f"[ERROR] Invalid JSON: {e}", file=sys.stderr)
    print(json.dumps({"answer": "Invalid input.", "escalate": False, "suggestion": ""}))
    sys.stdout.flush()
except Exception as e:
    print(f"[ERROR] {str(e)}", file=sys.stderr)
    print(json.dumps({"answer": "Sorry, an error occurred.", "escalate": False, "suggestion": ""}))
    sys.stdout.flush()