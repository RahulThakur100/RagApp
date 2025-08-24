import shutil
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_pipeline import run_rag_pipeline
from audio_service import transcribe_audio, synthesize_speech
from upload_data_service import extract_text_from_pdf, extract_text_from_docx, upload_file_to_pinecone


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure uploads folder exists

app = FastAPI()

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# 🔹 Request Models
# ------------------------


class Question(BaseModel):
    query: str


# ------------------------
# 🔹 Chat Endpoint (Text)
# ------------------------
@app.post("/chat")
async def chat(data: Question):
    answer = run_rag_pipeline(data.query)
    return {"answer": answer}


# ------------------------
# 🔹 Audio Endpoint
# ------------------------
@app.post("/audio")
async def audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".mp3", ".wav", ".ogg", ".m4a", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # 1. Transcribe audio
    question = await transcribe_audio(file)

    print(f"Transcribed question: {question}")

    # 2. Get answer using RAG pipeline
    answer = run_rag_pipeline(question)

    print(f"Generated answer: {answer}")

    # 3. Convert to speech (only because input was audio)
    audio_url = await synthesize_speech(answer)

    return {"answer": answer, "question": question, "audio_file": audio_url}


# ------------------------
# 🔹 File Upload Endpoint
# ------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extyract text based on file type
    if suffix == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif suffix == ".docx":
        text = extract_text_from_docx(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    # Remove the temporary file
    os.remove(file_path)

    # uplaod chunks to Pinecone
    upload_file_to_pinecone(text, file.filename)

    return {"message": f"File uploaded and indexed {file.filename}"}
