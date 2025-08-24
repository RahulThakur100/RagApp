# 🧠 RAG App

A simple **Retrieval-Augmented Generation (RAG)** application built with Python.  
This app lets you upload your own data, index it and query it using an LLM — making responses more accurate and grounded in your knowledge base.

---

## 🚀 Features

- Upload and index documents (`upload_data_service.py`)
- Query pipeline with retrieval + generation (`rag_pipeline.py`)
- Support for audio input (`audio_service.py`)
- Simple CLI / script interface

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
