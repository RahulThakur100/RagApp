import os
import uuid
from openai import OpenAI
import shutil
from fastapi import UploadFile
from pathlib import Path

UPLOAD_DIR = "uploads"

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Init clients
client = OpenAI(api_key=OPENAI_API_KEY)


async def transcribe_audio(file: UploadFile) -> str:
    """Transcribe audio file to text."""
    file_path = Path(UPLOAD_DIR) / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    os.remove(file_path)
    return transcript.text.strip()


async def synthesize_speech(text: str) -> str:
    """Convert text reply to speech and save to uploads folder.."""
    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}.mp3"
    audio_file_path = Path(UPLOAD_DIR) / unique_filename

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        input=text,
        voice="alloy"
    )

    with open(audio_file_path, "wb") as f:
        f.write(response.read())

    return f"/uploads/{unique_filename}"
