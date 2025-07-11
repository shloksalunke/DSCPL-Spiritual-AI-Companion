# 🙏 DSCPL - Spiritual AI Companion

**DSCPL (Daily Spiritual Companion and Personal Leader)** is an AI-powered Christian assistant that provides personalized prayer guidance, devotion plans, and scripture-rooted conversations. With its ELIZA-inspired persona and Retrieval-Augmented Generation (RAG) using local LLaMA models, it offers emotionally intelligent support and spiritual companionship.

---

## 📖 Overview

**DSCPL-Spiritual-AI-Companion** is a locally-run intelligent spiritual assistant that fosters user well-being through daily devotions, personalized prayer flows, and meaningful biblical conversations. The assistant responds like a wise spiritual mentor — grounded in scripture, compassionate in tone, and context-aware.

---

### 🙌 Why DSCPL-Spiritual-AI-Companion?

This project aims to deliver an offline-first, faith-based AI that enhances spiritual growth and consistency. It blends memory, RAG, and local LLMs into a seamless experience:

- 🌸 **Personalized Guidance**: Tracks names, prayer topics, devotion durations, and offers tailored next steps.
- ✝️ **Biblical Integration**: Suggests verses, explains biblical meanings for stress, anxiety, fear, etc.
- 🧠 **LLaMA-powered Conversations**: Uses a local GGUF LLaMA model with custom prompts inspired by ELIZA-style personality config.
- 💬 **Memory & Context-Awareness**: Remembers past sessions, user intents, and devotion states.
- 🔍 **RAG Integration**: Uses LangChain + FAISS to retrieve relevant biblical context before responding.
- 🚀 **FastAPI Backend**: High-speed async API layer using FastAPI and Uvicorn.
- 🗄️ **SQLite Memory DB**: Lightweight persistence of sessions, memory, and user interactions.

---

## ⚙️ Getting Started

### 📦 Prerequisites

Ensure the following are installed:

- Python ≥ 3.10
- pip
- A GGUF LLaMA-compatible model (e.g., `llama-2-7b.Q4_K_M.gguf`)
- (Optional) Node.js if you are using a separate frontend

---

### 🛠 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/DSCPL-Spiritual-AI-Companion.git
cd DSCPL-Spiritual-AI-Companion
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Create a .env file:

env
Copy
Edit
MODEL_PATH=./models/llama-2-7b.Q4_K_M.gguf
DATABASE_PATH=./dscpl.db
🚀 Usage
Start the backend server with:

bash
Copy
Edit
uvicorn main:app --reload
Access the API on:

arduino
Copy
Edit
http://localhost:8000
🧪 Testing
You can interact directly via:

/chat endpoint using POST

Or a connected frontend (e.g., HTML + React + JS Socket)

Basic test setup (if available):

bash
Copy
Edit
pytest
🧠 Persona & Config
This assistant loads its character config from eliza_character_config.json which defines:

💡 Personality & tone (grace-filled, compassionate, wise)

🧭 Lore and backstory

🗣️ Message examples

🎯 Preferred response style (short, spiritual, emotionally aware)

You can modify the JSON file to tweak DSCPL's behavior!

🗂 Folder Structure
bash
Copy
Edit
backend/
├── core/
├── models/
├── rag/
├── main.py
├── eliza_character_config.json
├── .env
└── requirements.txt
❤️ Feedback & Contributions
If this project helps you, please ⭐ the repo or fork it.

Feel free to open an issue or submit a PR for:

🛠 New prayer topics

✍️ Devotion plan ideas

🤝 Collaborations

Together we can bring daily faith, tech, and encouragement closer to everyone.
