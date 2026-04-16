import os
import uuid
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from kokoro import KPipeline
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


TMP_DIR = Path("/tmp/wordstack_tts")
TMP_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 24000
MAX_INPUT_CHARS = 6000
DEFAULT_VOICE = "af_heart"


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    voice: str = Field(default=DEFAULT_VOICE, min_length=1)


app = FastAPI(title="wordstack-tts-api", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once at startup for low latency on repeated requests.
try:
    pipeline = KPipeline(lang_code="a")
except Exception as exc:
    raise RuntimeError("Failed to initialize Kokoro pipeline.") from exc


def _chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Chunk text to keep generation memory usage bounded."""
    stripped = " ".join(text.split())
    if len(stripped) <= chunk_size:
        return [stripped]

    chunks: List[str] = []
    start = 0
    while start < len(stripped):
        end = min(start + chunk_size, len(stripped))
        if end < len(stripped):
            split = stripped.rfind(" ", start, end)
            if split > start:
                end = split
        chunks.append(stripped[start:end].strip())
        start = end + 1

    return [c for c in chunks if c]


def _cleanup_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        # Avoid surfacing cleanup issues to API consumers.
        pass


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/audio/speech")
def create_speech(payload: SpeechRequest):
    text = payload.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        text_chunks = _chunk_text(text)
        all_audio: List[np.ndarray] = []

        for chunk in text_chunks:
            generator = pipeline(chunk, voice=payload.voice)
            for _, _, audio in generator:
                all_audio.append(np.asarray(audio, dtype=np.float32))

        if not all_audio:
            raise HTTPException(status_code=500, detail="No audio generated.")

        merged_audio = np.concatenate(all_audio)

        filename = f"speech_{uuid.uuid4().hex}.wav"
        output_path = TMP_DIR / filename
        sf.write(output_path, merged_audio, SAMPLE_RATE, format="WAV")

        return FileResponse(
            path=str(output_path),
            media_type="audio/wav",
            filename=filename,
            background=BackgroundTask(_cleanup_file, output_path),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
