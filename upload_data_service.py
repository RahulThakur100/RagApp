
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
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
# 🔹 File Handling (Upload API)
# ------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    reader = PdfReader(file_path)
    texts = [page.extract_text()
             for page in reader.pages if page.extract_text()]
    return "\n".join(texts)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    texts = [para.text for para in doc.paragraphs if para.text]
    return "\n".join(texts)


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap=50) -> list[str]:
    """Split text into overlapping chunks for embeddings"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def upload_file_to_pinecone(text: str, source: str):
    """Split text into chunks, embed, and upload to Pinecone"""
    chunks = split_text_into_chunks(text)
    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embed = response.data[0].embedding
        index.upsert([
            (f"{source}_{i}", embed, {"text": chunk, "source": source})
        ])
