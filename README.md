# AI Customer Support Bot (RAG-powered)

A Retrieval-Augmented Generation (RAG)-powered chatbot for simulating customer support interactions.

- **AI Core:** Python (LangChain + Gemini LLM)
- **Backend API:** Node.js/Express
- **Database:** MongoDB (session persistence)
- **Frontend:** Simple EJS chat UI

Handles FAQs from a JSON dataset, maintains conversation context, and escalates unresolved queries.

---

## Objective
Simulate customer support interactions using AI for FAQs and escalation scenarios.

## Scope of Work
- **Input:** FAQs dataset & customer queries
- **Contextual memory:** Retain previous conversation
- **Escalation simulation:** When query not answered
- **Optional frontend chat interface**

## Technical Expectations
- Backend API with REST endpoints
- LLM integration for response generation
- Database for session tracking

## LLM Usage Guidance
- Generate responses
- Summarize conversations
- Suggest next actions

---

## Features
- **RAG Pipeline:** Loads FAQs from `faqs.json`, embeds with SentenceTransformer, stores in Chroma vector DB, retrieves relevant chunks for Gemini LLM responses.
- **Context-Aware:** Conversation history from MongoDB is injected into prompts for follow-ups (e.g., "tell me more about this" references prior topics).
- **Escalation Handling:** Low similarity scores trigger human handover simulation (log/email).
- **Simple UI:** EJS-based chat interface with real-time messaging via Fetch API.
- **Modular Backend:** Separated into models, helpers (session/history/RAG), routes for easy maintenance.

---

## Tech Stack
- **Backend:** Node.js/Express (API), Mongoose (MongoDB ODM)
- **AI/ML:** Python/LangChain (RAG chain), SentenceTransformer (embeddings), Chroma (vectorstore), Google Gemini (LLM)
- **Database:** MongoDB Atlas (cloud sessions)
- **Frontend:** EJS templates, vanilla JS
- **Deployment Ready:** Dockerizable; tested on localhost.

---

## Prerequisites
- Node.js ≥18
- Python ≥3.10 (with venv)
- MongoDB Atlas account (free tier)
- Google AI Studio API key (for Gemini)

---

## Installation

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/ai-customer-support-bot.git
cd ai-customer-support-bot
```

### 2. Set Up Python (RAG Core)
From project root:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt  # Includes langchain, chromadb, sentence-transformers, etc.
```
Add to root `.env`:
```
GOOGLE_API_KEY=your-gemini-api-key
```
Run notebook: `jupyter notebook rag.ipynb` → Execute all cells to persist `./chroma_db`.
Test RAG:
```bash
echo '{"query": "What is delivery modes?"}' | python rag_query.py
```

### 3. Set Up Backend (Node.js)
```bash
cd backend
npm install
```
Edit `backend/.env`:
```
MONGO_URI=mongodb+srv://youruser:pass@cluster.mongodb.net/csbot?retryWrites=true&w=majority
PORT=3000
SESSION_SECRET=your-secret
```
Test Mongo:
```bash
node test-mongo.js # creates sample session in "csbot"
```

### 4. Run the App
```bash
cd backend
npm run dev  # Starts on http://localhost:3000
```
Open: [http://localhost:3000/chat](http://localhost:3000/chat) → Chat interface ready.

---

## Usage
- **Chat via UI:** Type queries (e.g., "tell me about return policy") → Bot responds from FAQs with context.
- **API Testing (Postman/Curl):**
  - `GET /chat`: Renders UI, sets session cookie.
  - `POST /api/chat`:
    ```json
    {
      "query": "tell me about refund policy",
      "sessionId": "your-session-id"
    }
    ```
    Response:
    ```json
    {"answer": "You can receive a full refund...", "escalate": false, "suggestion": "..."}
    ```

### Context Example
- Query 1: "tell me about refund policy" → RAG from FAQs.
- Query 2: "tell me more about this" → References prior refund response.
- Escalation: Unrelated query (e.g., "weather?") → "Let me escalate..." + logs.
- Sessions persist in MongoDB "csbot" DB—refresh browser to see history.

---

## Project Structure
```
ai-customer-support-bot/
├── faqs.json              # FAQ dataset
├── rag.ipynb              # RAG setup notebook
├── requirements.txt       # Python deps
├── chroma_db/             # Persisted vectorstore
├── rag_query.py           # RAG microservice
└── backend/               # Node.js API
    ├── .env
    ├── package.json
    ├── app.js             # Main server
    ├── models/Session.js  # Mongoose schema
    ├── helpers/           # Utils (session, history, RAG spawn)
    ├── routes/chatRoutes.js # Endpoints
    └── views/chat.ejs     # Chat UI
```

---

## LLM Usage
- **Responses:** Concise (3 sentences max) from RAG + history.
- **Summarization:** Auto-summarizes long histories (>300 chars) for prompt efficiency.
- **Suggestions:** Ends with next-action prompts (e.g., "Check Returns page").

---

## Contributing
1. Fork the repo.
2. Create branch: `git checkout -b feature/xyz`.
3. Commit: `git commit -m "Add feature"`.
4. Push: `git push origin feature/xyz`.
5. Open PR—describe changes.

Issues? Open a ticket with repro steps.

---

## License
MIT License—see LICENSE for details.

---

## Acknowledgments
- LangChain for RAG orchestration.
- Google Gemini for LLM.
- MongoDB Atlas for cloud DB.

---
# Vedio Explanation
https://drive.google.com/file/d/1TDE1eAi2LoEgiG9TYrhXBd2jcFhoC38_/view?usp=sharing

Built with ❤️ for efficient customer support. Questions? Open an issue.

