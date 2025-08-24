import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import tempfile
from pypdf import PdfReader
import docx

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")

# Init clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ------------------------
# 🔹 Embeddings
# ------------------------


def embed_text(text: str) -> list[float]:
    """Generate embeddings for the given text using OpenAI."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# ------------------------
# 🔹 Retrieval
# ------------------------
def retrieve_context(query: str, top_k: int = 5) -> str:
    """Retrieve context from Pinecone based on the query."""
    query_vector = embed_text(query.strip().lower())

    results = index.query(vector=query_vector,
                          top_k=top_k, include_metadata=True)

    if not results.matches:
        return "Sorry, I don't know."

    context = []
    for match in results.matches:
        text = match.metadata.get("text", "")
        # if it's a list, join it
        if isinstance(text, list):
            text = "\n".join(text)
        context.append(str(text))  # make sure it's a string

    return "\n".join(context)


# ------------------------
# 🔹 Generation
# ------------------------
def generate_answer(query: str, context: str) -> str:
    """Generate final answer using GPT-4 with context"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n"
                    "The user’s question may come from speech-to-text transcription "
                    "and might include mispronunciations or spelling mistakes. "
                    "First, infer the intended question as best as possible, then "
                    "answer ONLY using the provided context.\n\n"
                    "If the answer is not in the context, say 'Sorry, I don't know.'"
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]
    )
    return response.choices[0].message.content.strip()


def run_rag_pipeline(query: str) -> str:
    """Run the entire RAG pipeline: embed → retrieve → generate"""
    context = retrieve_context(query)
    print(f"Retrieved context: {context}")
    answer = generate_answer(query, context)
    return answer
