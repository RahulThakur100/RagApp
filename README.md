# 🧠💬🎙️ RAG Conversational Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot with **chat memory** and **voice input/output**.  
It lets you **talk** with your data: ask questions via voice or text, get context-aware answers, and hear responses spoken back to you.

Powered by:

- [Whisper-1](https://platform.openai.com/docs/guides/speech-to-text) for speech-to-text 🎙️
- [GPT-4o-mini-tts](https://platform.openai.com/docs/guides/text-to-speech) for speech synthesis 🔊
- Retrieval-Augmented Generation (RAG) for knowledge grounding 📚

---

## 🚀 Features

- 🔍 **Knowledge-grounded answers** from your uploaded documents
- 💬 **Chat capabilities** — keeps track of conversation context
- 🎙️ **Voice input** (transcription with Whisper-1)
- 🔊 **Voice output** (natural speech with GPT-4o-mini-tts)
- 📂 Easy document ingestion (`upload_data_service.py`)

---

## 📦 Requirements

- Python 3.9+
- Dependencies in [`requirements.txt`](requirements.txt)

Install them with:

```bash
pip install -r requirements.txt

⚙️ Setup

1. Clone the repo:

    git clone https://github.com/RahulThakur100/RagApp.git
    cd RagApp


2. Create a .env file in the root directory and add your API key(s):

    OPENAI_API_KEY=your_api_key_here
    PINECONE_API_KEY = your_pinecone_api_key
    PINECONE_INDEX_NAME = pinecone_index_name


3. Run the app:

    python main.py


📂 Project Structure
RagApp/
│── main.py                # Entry point
│── rag_pipeline.py         # RAG pipeline logic
│── upload_data_service.py  # Data ingestion
│── audio_service.py        # Audio input/output
│── requirements.txt        # Python dependencies
│── .gitignore
│── README.md
```
