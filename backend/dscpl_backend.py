# ✅ Enhanced DSCPL Spiritual Assistant with Persona Identity, Reminders, Memory, Devotion Flow, RAG, and Local LLaMA + ELIZA-style Config

import json
import re
import os
from dotenv import load_dotenv
load_dotenv()
from uuid import uuid4
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from llama_cpp import Llama

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI(title="DSCPL - AI Spiritual Assistant", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./dscpl.db")

# Base models
class ChatMessage(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    requires_clarification: bool = False
    suggested_actions: List[str] = []

class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = {}
    prayer_topics: List[str] = []
    devotion_topics: List[str] = []

# Static biblical knowledge
BIBLICAL_KNOWLEDGE = {
    "stress": {
        "verses": ["Philippians 4:6-7", "Matthew 11:28-30", "1 Peter 5:7"],
        "guidance": "God wants us to cast our anxieties on Him.",
        "prayer_focus": "Peace and trust in God"
    },
    "fear": {
        "verses": ["Isaiah 41:10", "2 Timothy 1:7", "Joshua 1:9"],
        "guidance": "God has given us power, love, and a sound mind.",
        "prayer_focus": "Courage and faith"
    },
    "anxiety": {
        "verses": ["Philippians 4:6", "1 Peter 5:7", "Psalm 94:19"],
        "guidance": "Give all your worries to God and He will comfort you.",
        "prayer_focus": "Relief from anxious thoughts"
    }
}

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            session_id TEXT,
            message TEXT,
            response TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')

        cur.execute('''CREATE TABLE IF NOT EXISTS memory (
            user_id TEXT,
            key TEXT,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key))''')

        conn.commit()
        conn.close()

    def save(self, user_id, session_id, message, response):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('INSERT INTO conversations (user_id, session_id, message, response) VALUES (?, ?, ?, ?)',
                    (user_id, session_id, message, response))
        conn.commit()
        conn.close()

    def history(self, user_id, limit=10):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('SELECT message, response, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
        rows = cur.fetchall()
        conn.close()
        return [{"message": r[0], "response": r[1], "timestamp": r[2]} for r in rows]

    def load_memory(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('SELECT key, value FROM memory WHERE user_id = ?', (user_id,))
        rows = cur.fetchall()
        conn.close()
        return {k: v for k, v in rows}

    def save_memory(self, user_id, key, value):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''INSERT INTO memory (user_id, key, value, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP''',
                    (user_id, key, value))
        conn.commit()
        conn.close()

    def get_progress(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''SELECT COUNT(*) FROM conversations WHERE user_id = ? AND message LIKE '%prayer%' ''', (user_id,))
        prayers = cur.fetchone()[0]
        cur.execute('''SELECT COUNT(*) FROM conversations WHERE user_id = ? AND message LIKE '%devotion%' ''', (user_id,))
        devotions = cur.fetchone()[0]
        cur.execute('''SELECT MAX(timestamp) FROM conversations WHERE user_id = ?''', (user_id,))
        last = cur.fetchone()[0]
        conn.close()
        return {"prayers": prayers, "devotions": devotions, "last_activity": last or "N/A"}

class LocalLLaMAManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found at path: {self.model_path}")
            return
        try:
            logger.info(f"Loading LLaMA model from: {self.model_path}")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                temperature=0.7,
                max_tokens=512
            )
        except Exception as e:
            logger.error(f"LLaMA model loading failed: {e}")

    def generate(self, prompt: str) -> str:
        if not self.llm:
            return "I'm here to support your spiritual journey."
        try:
            output = self.llm(prompt)
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Let me try that again. I'm here for you."

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self._init_docs()

    def _init_docs(self):
        docs = [Document(page_content=v["guidance"], metadata={"topic": k}) for k, v in BIBLICAL_KNOWLEDGE.items()]
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        logger.info(f"RAG initialized with {len(chunks)} chunks")

    def retrieve(self, query: str) -> List[Document]:
        try:
            return self.vector_store.similarity_search(query, k=2)
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return []

class DSCPLAgent:
    def __init__(self):
        self.db = DatabaseManager(DATABASE_PATH)
        self.llama = LocalLLaMAManager(MODEL_PATH)
        self.rag = RAGSystem()
        self.user_profiles = {}

        # ✅ Load persona config from file
        with open("eliza_character_config.json", "r", encoding="utf-8") as f:
            self.character_config = json.load(f)

    async def respond(self, message: str, user_id: str, session_id: str) -> ChatResponse:
        memory = self.db.load_memory(user_id)
        message_lower = message.lower()

        if "my name is" in message_lower:
            name = message.split("is")[-1].strip().split()[0].capitalize()
            self.db.save_memory(user_id, "name", name)
            return ChatResponse(response=f"Nice to meet you, {name}!", session_id=session_id)

        name = memory.get("name", "friend")

        # Flows same as before (Reminder, Devotion, Prayer)
      # Push notification flow
        if memory.get("intent") == "reminder" and "reminder_time" not in memory:
            self.db.save_memory(user_id, "reminder_time", message)
            self.db.save_memory(user_id, "intent", "")  # Clear intent
            return ChatResponse(response=f"Got it! I will remind you at {message}. Stay blessed!", session_id=session_id)

        if "remind" in message_lower:
            self.db.save_memory(user_id, "intent", "reminder")
            return ChatResponse(response="Would you like to receive a push notification for this? What time should I remind you?", session_id=session_id)

        # Devotion flow
        if "devotion" in message_lower and "devotion_duration" not in memory:
            self.db.save_memory(user_id, "intent", "devotion")
            return ChatResponse(response="How many days of devotion would you like?", session_id=session_id, suggested_actions=["1 Day", "5 Days"])

        if memory.get("intent") == "devotion" and "devotion_duration" not in memory:
            self.db.save_memory(user_id, "devotion_duration", message)
            return ChatResponse(response="What topic would you like for your devotion?", session_id=session_id, suggested_actions=["Grace", "Purpose"])

        if memory.get("intent") == "devotion" and "devotion_topic" not in memory:
            self.db.save_memory(user_id, "devotion_topic", message)
            self.db.save_memory(user_id, "intent", "")  # ✅ Clear intent so LLM is used next
            return ChatResponse(response=f"Great, {name}. I’ll guide you through a {memory['devotion_duration']} devotion journey on {message.title()}.", session_id=session_id)

        # Prayer flow
        if "prayer" in message_lower and "topic" not in memory:
            self.db.save_memory(user_id, "intent", "prayer")
            return ChatResponse(response="How many days of prayer would you like?", session_id=session_id, suggested_actions=["Today Only", "3 Days"])

        if memory.get("intent") == "prayer" and "duration" not in memory:
            self.db.save_memory(user_id, "duration", message)
            return ChatResponse(response="What topic would you like for your prayer?", session_id=session_id, suggested_actions=["Healing", "Growth"])

        if memory.get("intent") == "prayer" and "topic" not in memory:
            self.db.save_memory(user_id, "topic", message)
            self.db.save_memory(user_id, "intent", "")  # ✅ Clear intent
            return ChatResponse(response=f"Thanks! Starting your {memory['duration']} prayer plan on {message}. Here's a prayer: \n\n'Lord, I come to you seeking {message.lower()}... Amen.'", session_id=session_id)

        # ✅ From here — LLM gets invoked for natural free-text messages
        rag_docs = self.rag.retrieve(message)
        rag_content = "\n".join([d.page_content for d in rag_docs])
      
        bio = "\n".join(self.character_config.get("bio", []))
        lore = "\n".join(self.character_config.get("lore", []))
        style = "\n".join(self.character_config.get("style", {}).get("all", []))

        prompt = f"""
You are {self.character_config['name']}, a spiritual AI companion.

{bio}

{lore}

You always speak in the following style:
{style}

Use this if helpful:
{rag_content}

The user you're helping is named {name}.

User: {message}
{self.character_config['name']}:
"""

        raw_reply = self.llama.generate(prompt)
        reply = re.split(r"User:", raw_reply)[0].strip()
        reply = re.sub(r'<a\s+href=[^>]+>(.*?)</a>', r'\1', reply)
        reply = re.sub(r'https?://\S+', '', reply)
        reply = re.sub(r'\[.*?\]\(.*?\)', '', reply)

        self.db.save(user_id, session_id, message, reply)

        suggested_actions = []
        if any(w in message_lower for w in ["fear", "anxiety", "worry", "panic"]):
            suggested_actions = ["Prayer", "Peace", "Faith"]
        elif any(w in message_lower for w in ["purpose", "calling", "lost", "meaning"]):
            suggested_actions = ["Devotion", "Scripture", "God's Plan"]
        elif any(w in message_lower for w in ["forgive", "hurt", "relationship"]):
            suggested_actions = ["Forgiveness", "Healing", "Prayer"]
        else:
            suggested_actions = ["Prayer", "Devotion", "Scripture"]

        clarification = any(word in message_lower for word in ["devotion", "prayer", "remind", "help"]) and len(message.split()) < 5

        return ChatResponse(response=reply, session_id=session_id, requires_clarification=clarification, suggested_actions=suggested_actions)

agent = DSCPLAgent()

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    sid = chat_message.session_id or f"sess_{datetime.now().timestamp()}"
    return await agent.respond(chat_message.message, chat_message.user_id, sid)

@app.get("/user/{user_id}/progress")
async def user_progress(user_id: str):
    return agent.db.get_progress(user_id)

@app.get("/user/{user_id}/history")
async def get_history(user_id: str):
    return {"history": agent.db.history(user_id)}

@app.post("/user/profile")
async def update_profile(profile: UserProfile):
    agent.user_profiles[profile.user_id] = profile.name
    return {"message": "Profile saved"}

@app.get("/topics")
async def get_topics():
    return {"topics": list(BIBLICAL_KNOWLEDGE.keys())}

@app.get("/new_session")
async def new_session(user_id: str):
    new_sid = f"sess_{datetime.now().timestamp()}_{uuid4().hex[:8]}"
    return {"new_session_id": new_sid}

@app.get("/user/{user_id}/session/{session_id}/history")
async def get_session_history(user_id: str, session_id: str):
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute('''SELECT message, response, timestamp FROM conversations WHERE user_id = ? AND session_id = ? ORDER BY timestamp ASC''', (user_id, session_id))
    rows = cur.fetchall()
    conn.close()
    return [{"message": r[0], "response": r[1], "timestamp": r[2]} for r in rows]

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
