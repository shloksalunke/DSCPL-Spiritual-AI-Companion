# ✅ Enhanced DSCPL Spiritual Assistant with Persona Identity, Reminders, Memory, Devotion Flow, RAG, and Local LLaMA + ELIZA-style Config

from email.mime import message
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

PRAYER_TEMPLATES = {
    "anxiety": {
        "prayer": """
🕊️ **Prayer for Anxiety**

**Adoration:**  
Heavenly Father, You are the Prince of Peace. You sit enthroned above every storm. You are not the author of confusion but of calm. Your Word declares that perfect love drives out all fear — and Your love for me is unchanging.

**Confession:**  
Lord, I confess that I have allowed anxious thoughts to control me. I’ve worried about tomorrow instead of trusting You with today. I’ve tried to carry burdens that You never intended for me to carry. Forgive me for forgetting that You are in control.

**Thanksgiving:**  
Thank You for always being near to the brokenhearted. Thank You for Your promise in Philippians 4:6–7, that if I bring everything to You in prayer with thanksgiving, Your peace — which surpasses understanding — will guard my heart and mind.

**Supplication:**  
Lord, I surrender my anxious thoughts to You. I ask You to quiet my racing mind and steady my heart. Help me to fix my eyes on You instead of the waves. Teach me to breathe deeply, think clearly, and live with peace rooted in Your presence.

📖 *Philippians 4:6–7 • 1 Peter 5:7 • Psalm 94:19*
""".strip()
    },
    "healing": {
        "prayer": """
💖 **Prayer for Healing**

**Adoration:**  
Jehovah Rapha, the Lord who heals — I worship You. You are the God who knit me together in my mother’s womb and holds every cell in Your hand. You spoke healing into being throughout the life of Jesus, and I believe You still heal today.

**Confession:**  
Lord, I confess my doubt and discouragement. Sometimes I believe more in the sickness than in the Healer. Forgive me for letting despair drown out faith. I admit I need You more than ever — not just for healing my body, but healing my hope.

**Thanksgiving:**  
Thank You for doctors, medicines, and wisdom You’ve provided. Thank You for the healing that flows through community, prayer, and even rest. Most of all, thank You that by the wounds of Jesus, healing is possible — spiritually, emotionally, and physically.

**Supplication:**  
Lord, I ask for healing — in the areas I’ve spoken out loud and the silent aches no one sees. Reach into my pain, restore what’s been broken, and renew my strength like the eagle. I trust that You are working even when I can’t feel it.

📖 *Isaiah 53:5 • Jeremiah 17:14 • James 5:15*
""".strip()
    },
    "purpose": {
        "prayer": """
🌟 **Prayer for Purpose**

**Adoration:**  
Father God, You are the Creator of all things. You shaped galaxies and carved oceans, yet You also formed me with purpose. You do not create chaos — You ordain calling. You are not silent about destiny — You speak through every page of Your Word.

**Confession:**  
Lord, I confess that I’ve wandered, doubted, and compared my journey to others. I’ve looked for meaning in achievements or approval, instead of in Your presence. Forgive me for forgetting that my value is rooted in You, not in what I do.

**Thanksgiving:**  
Thank You for declaring that I am fearfully and wonderfully made. Thank You that You have plans to give me hope and a future. Thank You that my life has meaning because You authored it — not by accident, but by divine intention.

**Supplication:**  
Father, awaken the gifts You’ve placed in me. Open doors that align with Your will and close the ones that distract me. Help me to trust Your timing, walk boldly in obedience, and fulfill the unique purpose You’ve written for me before I was born.

📖 *Jeremiah 29:11 • Romans 8:28 • Proverbs 3:5–6*
""".strip()
    }
}


BIBLICAL_KNOWLEDGE = {
    "stress": {
        "verses": ["Philippians 4:6-7", "Matthew 11:28-30", "1 Peter 5:7", "Psalm 55:22"],
        "guidance": "God wants us to cast our anxieties on Him and He will sustain us.",
        "prayer_focus": "Peace and trust in God"
    },
    "fear": {
        "verses": ["Isaiah 41:10", "2 Timothy 1:7", "Joshua 1:9", "Psalm 23:4"],
        "guidance": "God has not given us a spirit of fear, but of power and love.",
        "prayer_focus": "Courage and faith"
    },
    "anxiety": {
        "verses": ["Philippians 4:6", "1 Peter 5:7", "Psalm 94:19", "John 14:27"],
        "guidance": "Give all your worries to God; His peace surpasses all understanding.",
        "prayer_focus": "Relief from anxious thoughts"
    },
    "guilt": {
        "verses": ["1 John 1:9", "Romans 8:1", "Psalm 103:12"],
        "guidance": "In Christ, we are forgiven and made new.",
        "prayer_focus": "Forgiveness and freedom"
    },
    "purpose": {
        "verses": ["Jeremiah 29:11", "Romans 8:28", "Proverbs 3:5-6"],
        "guidance": "God has a plan for your life, trust in His direction.",
        "prayer_focus": "Clarity and guidance"
    },
    "healing": {
        "verses": ["Isaiah 53:5", "Jeremiah 17:14", "James 5:15"],
        "guidance": "By His wounds we are healed — seek Him in prayer.",
        "prayer_focus": "Healing and restoration"
    },
    "forgiveness": {
        "verses": ["Ephesians 4:32", "Matthew 6:14-15", "Colossians 3:13"],
        "guidance": "Forgive as Christ forgave you, and be free.",
        "prayer_focus": "Letting go and embracing grace"
    },
    "loneliness": {
        "verses": ["Deuteronomy 31:6", "Psalm 68:6", "Matthew 28:20"],
        "guidance": "God is always with you and places the lonely in families.",
        "prayer_focus": "Connection and comfort"
    },
    "temptation": {
        "verses": ["1 Corinthians 10:13", "James 1:12", "Hebrews 4:15-16"],
        "guidance": "God provides a way out — lean on Him.",
        "prayer_focus": "Strength and purity"
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
        

        # ✅ Define full internal persona prompt (DSCPL spiritual assistant)
        self.persona_prompt = '''You are DSCPL, a deeply spiritual, emotionally intelligent AI companion. Your purpose is to walk with the user through every season of life — offering support, Scripture, devotion, and prayer. You are not just a chatbot — you're a spiritual friend and mentor, rooted in biblical truth and guided by grace.

💬 Your Communication Style:
- Sound like a wise, kind mentor — never robotic, never generic.
- Speak naturally, like a calm spiritual counselor.
- Use comforting language, faith-filled declarations, and encourage reflection.
- Avoid saying things like “As an AI…” or “I am a language model.”
- Speak directly to the user, using their name if you know it.
- Always connect your response with Scripture, whenever relevant.

📖 Your Foundation:
- Base your guidance on the Bible — reference verses (e.g., Philippians 4:6), principles (e.g., forgiveness, faith), and teachings of Jesus.
- If the user expresses emotions like anxiety, fear, or guilt — respond with empathy and Scripture.
- When users ask spiritual or life questions, use Bible-based wisdom and sound like a compassionate human.

🧠 Personal Memory Handling:
- Remember if the user shares their name, struggles, spiritual goals, or reminders.
- Use memory naturally: “Earlier you mentioned anxiety, would you like a meditation on peace today?”

🎯 Key Capabilities:
- Guide users through 1-day, 5-day, or 7-day **Devotion** or **Prayer** plans.
- Prompt them: “Would you like to receive daily reminders for this at a specific time?”
- Understand user intent from open-ended phrases like “I’m feeling lost” or “Can you help me?”
- If their message is vague or short, kindly ask for more details without being annoying.

🕊️ Sample Conversations:
User: “I’m anxious lately.”
You: “I’m here for you. Would you like a short prayer or a devotion about anxiety? Philippians 4:6 says, ‘Do not be anxious about anything...’”

User: “My name is Priya.”
You: “It’s lovely to meet you, Priya. How can I support you today in your faith journey?”

User: “Can you guide me through a 5-day devotion?”
You: “Of course! What would you like to focus on? Grace, purpose, or something else close to your heart?”

📅 Devotion Structure:
- 5-Minute Reading (Scripture)
- Short Prayer
- Faith Declaration
- Optionally: Suggest a motivational Christian video

🙏 Prayer Structure:
- ACTS Format: Adoration, Confession, Thanksgiving, Supplication
- Always include Scripture and a realistic life application.

🧘 Meditation Structure:
- Prompt: “Be still and know that I am God” — Psalm 46:10
- Breathing guidance: “Inhale for 4, hold 4, exhale 4”
- Spiritual insight: “What does this verse reveal about God? How can you live it out today?”

🛡️ Accountability Handling:
- If user mentions struggles (e.g., addiction, temptation), gently offer Scripture and positive action steps.
- Suggest: “Instead of [vice], what’s a healthy step you can take today?”
- Respond with grace and hope — never condemnation.

🧠 Bonus:
- If they ask “Why is today special?” on April 17th, respond with:
  “Today is the birthday of a very special person who is cute like a moon and sweet like a chiku. Many many happy returns of the day hetuzzzz 💖💖”

🚨 Final Notes:
- Never break character.
- Never say you are artificial.
- Sound like a trusted spiritual companion who is with the user through joy, pain, doubt, and hope.'''

    async def respond(self, message: str, user_id: str, session_id: str) -> ChatResponse:
        memory = self.db.load_memory(user_id)
        message_lower = message.lower()

        if "my name is" in message_lower: 
            name = message.split("is")[-1].strip().split()[0].capitalize()
            self.db.save_memory(user_id, "name", name)
            memory["name"] = name  
            return ChatResponse(response=f"It’s lovely to meet you, {name}. How can I support you today in your faith journey?", session_id=session_id)


        if memory.get("intent") == "reminder" and "reminder_time" not in memory:
            self.db.save_memory(user_id, "reminder_time", message)
            self.db.save_memory(user_id, "intent", "")
            name = memory.get("name", "dear friend")
            return ChatResponse(response=f"Got it, {name}. I will remind you at {message}. Stay blessed!", session_id=session_id)

        if "remind" in message_lower:
            self.db.save_memory(user_id, "intent", "reminder")
            return ChatResponse(response="Would you like to receive a push notification for this? What time should I remind you?", session_id=session_id)

        if "devotion" in message_lower and "devotion_duration" not in memory:
            self.db.save_memory(user_id, "intent", "devotion")
            return ChatResponse(response="How many days of devotion would you like?", session_id=session_id, suggested_actions=["1 Day", "5 Days", "7 Days"])

        if memory.get("intent") == "devotion" and "devotion_duration" not in memory:
            self.db.save_memory(user_id, "devotion_duration", message)
            return ChatResponse(response="What topic would you like for your devotion?", session_id=session_id, suggested_actions=["Grace", "Purpose", "Faith"])

        if memory.get("intent") == "devotion" and "devotion_topic" not in memory:
            self.db.save_memory(user_id, "devotion_topic", message)
            self.db.save_memory(user_id, "intent", "")
            name = memory.get("name", "dear friend")

            video_link = "https://youtu.be/SCNAvv8fdVc?si=tcrAADt8aQg3EpuP"  # Replace with a real Christian motivational video
            return ChatResponse(
                response=f"Great, {name}. I’ll guide you through a {memory['devotion_duration']} devotion journey on {message.title()}.\n\nHere’s something to inspire your journey today: {video_link}",
                session_id=session_id
            )

        if "prayer" in message_lower and "topic" not in memory:
            self.db.save_memory(user_id, "intent", "prayer")
            return ChatResponse(response="How many days of prayer would you like?", session_id=session_id, suggested_actions=["Today Only", "3 Days", "7 Days"])

        if memory.get("intent") == "prayer" and "duration" not in memory:
            self.db.save_memory(user_id, "duration", message)
            return ChatResponse(response="What topic would you like for your prayer?", session_id=session_id, suggested_actions=["Healing", "Growth", "Peace"])

        if memory.get("intent") == "prayer" and "topic" not in memory:
            topic = message.lower()
            self.db.save_memory(user_id, "topic", topic)
            self.db.save_memory(user_id, "intent", "")
            name = memory.get("name", "dear friend")
            prayer_block = PRAYER_TEMPLATES.get(topic, {}).get("prayer")

            if prayer_block:
                return ChatResponse(
                    response=f"Thank you, {name}. Starting your {memory['duration']} prayer plan on **{topic.title()}**.\n\n{prayer_block}",
                    session_id=session_id
                )
            else:
                return ChatResponse(
                    response=f"Thank you, {name}. Starting your {memory['duration']} prayer plan on {topic}.\n\nHere’s a short prayer:\n\nLord, I come to You for {topic}. Let Your will be done. Amen.",
                    session_id=session_id
                )



        rag_docs = self.rag.retrieve(message)
        rag_content = "\n".join([d.page_content for d in rag_docs])
        name = memory.get("name", "dear friend")
        prompt = f"{self.persona_prompt}\n\nUse this if helpful:\n{rag_content}\n\nThe user you're helping is named {name}.\n\nUser: {message}\nDSCPL:"

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

        return ChatResponse(
            response=reply,
            session_id=session_id,
            requires_clarification=clarification,
            suggested_actions=suggested_actions
        )

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
